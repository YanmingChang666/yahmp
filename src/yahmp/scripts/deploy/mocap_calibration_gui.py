"""Live MuJoCo + tkinter mocap calibration with PER-JOINT deviation sliders.

An extension of the root-only GUI in `run_yahmp_onnx_mocap._run_gui_calibration`:
in addition to root Roll/Pitch/Yaw + height, this exposes one slider per joint so
you can hand-trim each joint's offset while watching the calibrated pose in MuJoCo,
and it keeps the automatic neutral-pose capture as a button.

    raw mocap ─► LiveReference(uncalibrated) ─► _InteractiveCalib.apply ─► qpos ─► viewer

`_InteractiveCalib` (reused from `run_yahmp_onnx_mocap`) is the shared model:
`joint_offset` is the per-joint deviation, `capture_neutral()` is the automatic
adjustment, `to_dict()` is the saved JSON (same schema `MocapCalibration.load`
reads). Single-threaded, mirroring the existing GUI: a ~50 Hz tkinter `after` tick
samples the stream, applies the (live-editable) calibration, and syncs the viewer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def run_gui_joint_calibration(
  *,
  model: Any,
  data: Any,
  spec: Any,
  reference: Any,
  viewer_cfg: Any,
  joint_qpos_adr: np.ndarray,
  root_qpos_adr: int,
  out_path: str,
  height_target: float,
  seed: Optional[Any],
  root_body_name: str = "pelvis",
  title: str = "Mocap calibration — per-joint offsets",
) -> bool:
  """Open the calibration GUI. Returns True if it ran, False if no GUI is available.

  `reference` must be an UNCALIBRATED `LiveReference` (the GUI applies calibration
  itself). The caller owns the mocap source lifecycle (start/stop).
  """
  try:
    import tkinter as tk

    _probe = tk.Tk()
    _probe.withdraw()
    _probe.destroy()
  except Exception as exc:  # noqa: BLE001 - ImportError or TclError (no $DISPLAY)
    print(f"[Calibrate] slider GUI unavailable ({exc}); falling back to auto-capture.")
    return False

  import mujoco
  import mujoco.viewer as mujoco_viewer

  from yahmp.scripts.deploy.run_yahmp_onnx_mocap import _InteractiveCalib, _quat_roll_pitch_yaw
  from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import _configure_camera

  calib = _InteractiveCalib(spec, height_target, seed=seed)
  latest: dict = {"frame": None}

  root = tk.Tk()
  root.title(title)

  roll_v = tk.DoubleVar(value=float(np.degrees(calib.root_rpy_offset[0])))
  pitch_v = tk.DoubleVar(value=float(np.degrees(calib.root_rpy_offset[1])))
  yaw_v = tk.DoubleVar(value=float(np.degrees(calib.root_rpy_offset[2])))
  height_v = tk.DoubleVar(value=float(calib.root_height_target))
  enabled_v = tk.BooleanVar(value=True)
  status_v = tk.StringVar(value=calib.status)
  readout_v = tk.StringVar(value="")

  # ── Root sliders ────────────────────────────────────────────────────────────
  def _root_slider(label: str, var: "tk.DoubleVar", lo: float, hi: float, res: float) -> None:
    row = tk.Frame(root)
    row.pack(fill="x", padx=8, pady=1)
    tk.Label(row, text=label, width=10, anchor="w").pack(side="left")
    tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res, orient="horizontal",
             length=300).pack(side="left", fill="x", expand=True)

  tk.Label(root, text="Root", anchor="w", font=("TkDefaultFont", 9, "bold")).pack(fill="x", padx=8, pady=(6, 0))
  _root_slider("Roll °", roll_v, -30.0, 30.0, 0.5)
  _root_slider("Pitch °", pitch_v, -30.0, 30.0, 0.5)
  _root_slider("Yaw °", yaw_v, -60.0, 60.0, 0.5)
  _root_slider("Height m", height_v, 0.50, 1.00, 0.005)

  # ── Per-joint offset sliders (scrollable) ───────────────────────────────────
  tk.Label(root, text="Per-joint offset (deg)", anchor="w",
           font=("TkDefaultFont", 9, "bold")).pack(fill="x", padx=8, pady=(8, 0))
  outer = tk.Frame(root)
  outer.pack(fill="both", expand=True, padx=8, pady=2)
  canvas = tk.Canvas(outer, height=340, highlightthickness=0)
  vsb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
  inner = tk.Frame(canvas)
  inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
  canvas.create_window((0, 0), window=inner, anchor="nw")
  canvas.configure(yscrollcommand=vsb.set)
  canvas.pack(side="left", fill="both", expand=True)
  vsb.pack(side="right", fill="y")
  canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
  canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
  canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

  joint_vars: list = []
  for i, name in enumerate(calib.names):
    row = tk.Frame(inner)
    row.pack(fill="x", pady=0)
    tk.Label(row, text=name.replace("_joint", ""), width=22, anchor="w").pack(side="left")
    var = tk.DoubleVar(value=round(float(np.degrees(calib.joint_offset[i])), 2))
    tk.Scale(row, variable=var, from_=-30.0, to=30.0, resolution=0.5, orient="horizontal",
             length=240).pack(side="left", fill="x", expand=True)
    joint_vars.append(var)

  # ── State sync + actions ────────────────────────────────────────────────────
  def _sync_from_sliders() -> None:
    calib.root_rpy_offset = np.deg2rad([roll_v.get(), pitch_v.get(), yaw_v.get()])
    calib.root_height_target = float(height_v.get())
    calib.enabled = bool(enabled_v.get())
    calib.joint_offset = np.deg2rad(np.array([v.get() for v in joint_vars], dtype=np.float64))

  def _refresh_joint_sliders() -> None:
    for i, v in enumerate(joint_vars):
      v.set(round(float(np.degrees(calib.joint_offset[i])), 2))

  def _auto_capture() -> None:
    f = latest["frame"]
    if f is None:
      status_v.set("no mocap frame yet — stand in view")
      return
    calib.capture_neutral(f.joint_pos, f.root_pos_w, f.root_quat_w)
    roll_v.set(0.0)
    pitch_v.set(0.0)
    yaw_v.set(0.0)
    height_v.set(calib.root_height_target)
    _refresh_joint_sliders()
    status_v.set(calib.status)

  def _zero_joints() -> None:
    for v in joint_vars:
      v.set(0.0)
    status_v.set("all joint offsets zeroed")

  def _reset_rpy() -> None:
    roll_v.set(0.0)
    pitch_v.set(0.0)
    yaw_v.set(0.0)
    status_v.set("root R/P/Y reset to 0")

  def _save() -> None:
    _sync_from_sliders()
    p = Path(out_path)
    if p.parent and not p.parent.exists():
      p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(calib.to_dict(), indent=2))
    status_v.set(f"saved → {out_path}")

  btns = tk.Frame(root)
  btns.pack(fill="x", padx=8, pady=4)
  tk.Button(btns, text="Auto-capture neutral", command=_auto_capture).pack(side="left")
  tk.Button(btns, text="Zero joints", command=_zero_joints).pack(side="left", padx=4)
  tk.Button(btns, text="Reset R/P/Y", command=_reset_rpy).pack(side="left", padx=4)
  tk.Button(btns, text="Save", command=_save).pack(side="left", padx=4)
  tk.Checkbutton(root, text="calibration enabled", variable=enabled_v).pack(anchor="w", padx=8)
  tk.Label(root, textvariable=readout_v, anchor="w", justify="left",
           font=("TkFixedFont", 10)).pack(fill="x", padx=8)
  tk.Label(root, textvariable=status_v, anchor="w", fg="#0a5").pack(fill="x", padx=8, pady=4)

  # ── Viewer + tick loop (single thread) ──────────────────────────────────────
  viewer = mujoco_viewer.launch_passive(model, data)
  _configure_camera(viewer, model, root_body_name, viewer_cfg)
  print("[Calibrate] GUI open. Auto-capture neutral, then hand-trim per-joint sliders. Save when happy.")

  def _tick() -> None:
    if not viewer.is_running():
      root.destroy()
      return
    _sync_from_sliders()
    frame = reference.sample(0.0)  # raw (reference is uncalibrated in this mode)
    latest["frame"] = frame
    jp, rp, rq = calib.apply(frame.joint_pos, frame.root_pos_w, frame.root_quat_w)
    data.qpos[root_qpos_adr : root_qpos_adr + 3] = rp
    data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = rq
    data.qpos[joint_qpos_adr] = jp
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    viewer.sync()
    er, ep, ey = np.degrees(_quat_roll_pitch_yaw(rq))
    readout_v.set(
      f"command root: roll={er:+6.1f}° pitch={ep:+6.1f}° yaw={ey:+6.1f}° height={rp[2]:.3f} m"
    )
    root.after(20, _tick)

  def _on_close() -> None:
    try:
      viewer.close()
    finally:
      root.destroy()

  root.protocol("WM_DELETE_WINDOW", _on_close)
  root.after(0, _tick)
  try:
    root.mainloop()
  finally:
    try:
      viewer.close()
    except Exception:  # noqa: BLE001 - viewer may already be closed
      pass
  return True
