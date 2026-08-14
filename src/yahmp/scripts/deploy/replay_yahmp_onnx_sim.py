"""Replay a recorded YAHMP sim2real run in MuJoCo.

Reads a `*_full.csv` produced by `run_yahmp_onnx_recorder.Sim2RealRecorder`
(via `run_yahmp_onnx_real --record ...`) and animates the G1 in the same mjlab
task scene the policy was validated against, so you can *watch* what the robot
did on hardware and eyeball the sim/real gap.

It reuses `run_yahmp_onnx_mujoco`'s scene + joint-address helpers, so the joint
columns map straight onto the model (they are the exact ONNX joint names) with
no reordering. This is the YAHMP-focused analog of Beyondmimic's
`replay_chingmu_sim.py --run <name>`, minus the table-tennis ball / racket / LSTM
machinery that YAHMP has no equivalent of.

Which joint columns to play:
    --source actual   (default) measured joint positions — what the robot did
    --source target   the position setpoints sent to the motors (policy target)
    --source ref      the mocap reference the policy was tracking (the human)

Base pose: YAHMP real deploy has no pelvis world position, so the base is pinned
at a fixed height (--base-height) and only its *orientation* is animated from the
recorded IMU quaternion (--no-use-recorded-quat to disable). This still shows
lean / balance, just not translation.

One-key replay (mirrors the reference's `--run`):
    uv run python -m yahmp.scripts.deploy.replay_yahmp_onnx_sim \
        --run run_20260814_062931 --run-dir ./sim2real_data
    # {run}_full.csv is played; {run}_meta.json (if present) supplies the ONNX
    # path so the correct root body / joint set is used automatically.

Direct file:
    uv run python -m yahmp.scripts.deploy.replay_yahmp_onnx_sim \
        --csv ./sim2real_data/run_20260814_062931_full.csv --task-id Mjlab-YAHMP-Unitree-G1

Viewer controls (printed at startup): SPACE pause · ←/→ seek · ↑/↓ speed · R restart.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import (
  PolicySpec,
  _build_task_scene,
  _configure_camera,
  _joint_addresses,
  _root_addresses,
  _yahmp_task_ids,
)

# GLFW key codes used by the viewer scrubber.
_KEY_SPACE, _KEY_RIGHT, _KEY_LEFT, _KEY_UP, _KEY_DOWN, _KEY_R = 32, 262, 263, 265, 264, 82


def _euler_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
  cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
  cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
  cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
  return np.array(
    [
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ],
    dtype=np.float64,
  )


def _load_replay_csv(
  path: Path, source: str
) -> tuple[np.ndarray, list[str], np.ndarray, Optional[np.ndarray]]:
  """Return (rel_time[T], joint_names[n], q[T, n], base_quat[T, 4] wxyz | None)."""
  prefix = f"{source}_"
  with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames or []
    rows = list(reader)
  if not rows:
    raise SystemExit(f"{path} has no data rows.")

  joint_names = [c[len(prefix):] for c in header if c.startswith(prefix)]
  if not joint_names:
    raise SystemExit(
      f"No `{prefix}*` columns in {path.name}. Available sources: "
      f"{sorted({c.split('_')[0] for c in header if c.startswith(('actual_', 'target_', 'ref_'))})}"
    )

  def col(name: str) -> np.ndarray:
    return np.array(
      [float(r[name]) if r.get(name, "") not in ("", "nan") else np.nan for r in rows],
      dtype=np.float64,
    )

  times = col("rel_time")
  times = times - times[0]
  q = np.column_stack([col(f"{prefix}{n}") for n in joint_names])

  base_quat: Optional[np.ndarray] = None
  if all(f"quat_{a}" in header for a in "wxyz"):
    base_quat = np.column_stack([col(f"quat_{a}") for a in "wxyz"])
  elif all(f"euler_{a}" in header for a in ("roll", "pitch", "yaw")):
    eul = np.column_stack([col(f"euler_{a}") for a in ("roll", "pitch", "yaw")])
    base_quat = np.array([_euler_to_quat_wxyz(*e) for e in eul], dtype=np.float64)

  # Forward-fill any NaN gaps so mj_forward never sees a NaN qpos.
  for arr in (q, base_quat):
    if arr is None:
      continue
    for j in range(arr.shape[1]):
      col_j = arr[:, j]
      mask = np.isnan(col_j)
      if mask.any():
        col_j[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), col_j[~mask]) if (~mask).any() else 0.0
  return times, joint_names, q, base_quat


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Optional[Path]]:
  """Resolve the CSV path and an optional ONNX path (from --run's meta.json)."""
  if args.csv is not None:
    return args.csv.expanduser().resolve(), (
      args.onnx_path.expanduser().resolve() if args.onnx_path else None
    )
  if args.run is None:
    raise SystemExit("Provide either --run <name> or --csv <path>.")

  run_dir = args.run_dir.expanduser()
  base = args.run
  for suffix in ("_full.csv", "_full"):
    if base.endswith(suffix):
      base = base[: -len(suffix)]
  csv_path = run_dir / f"{base}_full.csv"
  if not csv_path.is_file():
    matches = sorted(run_dir.glob(f"{base}*_full.csv"))
    if not matches:
      raise SystemExit(f"No `{base}*_full.csv` under {run_dir}.")
    csv_path = matches[-1]

  onnx_path = args.onnx_path.expanduser().resolve() if args.onnx_path else None
  if onnx_path is None:
    meta_path = run_dir / f"{base}_meta.json"
    if not meta_path.is_file():
      # meta filename carries the timestamp; recover it from the resolved CSV.
      stem = csv_path.name[: -len("_full.csv")]
      meta_path = run_dir / f"{stem}_meta.json"
    if meta_path.is_file():
      try:
        meta = json.loads(meta_path.read_text())
        if meta.get("onnx_path"):
          cand = Path(meta["onnx_path"]).expanduser()
          onnx_path = cand if cand.is_file() else None
          print(f"[Replay] meta -> onnx {cand}{'' if onnx_path else '  (not found, using --root-body)'}")
      except Exception as exc:  # noqa: BLE001
        print(f"[Replay] warning: could not read {meta_path.name}: {exc}")
  return csv_path.resolve(), onnx_path


def run(
  *,
  csv_path: Path,
  task_id: str,
  source: str,
  onnx_path: Optional[Path],
  root_body: str,
  base_height: float,
  use_recorded_quat: bool,
  replay_speed: float,
  loop: bool,
  no_viewer: bool,
) -> None:
  times, joint_names, q, base_quat = _load_replay_csv(csv_path, source)
  duration = float(times[-1])
  print(
    f"[Replay] {csv_path.name}: {len(times)} frames  {duration:.2f}s  "
    f"~{len(times) / max(duration, 1e-6):.1f} Hz  source={source}  joints={len(joint_names)}"
  )
  if no_viewer:
    return

  root_body_name = root_body
  if onnx_path is not None:
    root_body_name = PolicySpec.from_onnx(onnx_path).root_body_name

  import mujoco
  import mujoco.viewer as mujoco_viewer

  model, viewer_cfg, _, _ = _build_task_scene(task_id)
  data = mujoco.MjData(model)
  joint_qpos_adr, _ = _joint_addresses(model, tuple(joint_names))
  try:
    root_qpos_adr, _ = _root_addresses(model, root_body_name)
  except Exception as exc:  # noqa: BLE001
    print(f"[Replay] warning: no free-joint root `{root_body_name}` ({exc}); base left at origin.")
    root_qpos_adr = None

  if base_quat is None and use_recorded_quat:
    print("[Replay] no quat/euler columns found; base orientation held upright.")

  # Player state (mutated by the key callback).
  state = {"paused": False, "speed": max(replay_speed, 1e-3), "t": 0.0}

  def on_key(keycode: int) -> None:
    if keycode == _KEY_SPACE:
      state["paused"] = not state["paused"]
    elif keycode == _KEY_RIGHT:
      state["t"] = min(duration, state["t"] + 2.0)
    elif keycode == _KEY_LEFT:
      state["t"] = max(0.0, state["t"] - 2.0)
    elif keycode == _KEY_UP:
      state["speed"] = min(8.0, state["speed"] * 1.5)
      print(f"[Replay] speed = {state['speed']:.2f}x")
    elif keycode == _KEY_DOWN:
      state["speed"] = max(0.05, state["speed"] / 1.5)
      print(f"[Replay] speed = {state['speed']:.2f}x")
    elif keycode == _KEY_R:
      state["t"] = 0.0

  def apply(idx: int) -> None:
    data.qpos[joint_qpos_adr] = q[idx]
    if root_qpos_adr is not None:
      data.qpos[root_qpos_adr : root_qpos_adr + 3] = (0.0, 0.0, base_height)
      quat = base_quat[idx] if (base_quat is not None and use_recorded_quat) else (1.0, 0.0, 0.0, 0.0)
      data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = quat
    mujoco.mj_forward(model, data)

  apply(0)
  print(
    "[Replay] controls: SPACE pause · ←/→ seek 2s · ↑/↓ speed · R restart"
    f"{'  (looping)' if loop else ''}"
  )
  with mujoco_viewer.launch_passive(model, data, key_callback=on_key) as viewer:
    _configure_camera(viewer, model, root_body_name, viewer_cfg)
    last = time.perf_counter()
    while viewer.is_running():
      now = time.perf_counter()
      dt = now - last
      last = now
      if not state["paused"]:
        state["t"] += dt * state["speed"]
        if state["t"] > duration:
          if loop:
            state["t"] -= duration
          else:
            state["t"] = duration
            state["paused"] = True
      idx = int(np.clip(np.searchsorted(times, state["t"], side="right") - 1, 0, len(times) - 1))
      apply(idx)
      viewer.sync()
      time.sleep(0.008)


def _build_argparser() -> argparse.ArgumentParser:
  yahmp_tasks = _yahmp_task_ids()
  p = argparse.ArgumentParser(
    description="Replay a recorded YAHMP sim2real run in MuJoCo.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  p.add_argument("--run", type=str, default=None, help="Run name; plays {run}_full.csv from --run-dir.")
  p.add_argument("--run-dir", type=Path, default=Path("."), help="Directory holding the recorded run.")
  p.add_argument("--csv", type=Path, default=None, help="Direct path to a *_full.csv (overrides --run).")
  p.add_argument(
    "--task-id",
    type=str,
    default=(yahmp_tasks[0] if yahmp_tasks else None),
    choices=yahmp_tasks or None,
    required=not yahmp_tasks,
    help="mjlab task scene to visualize in.",
  )
  p.add_argument("--source", choices=("actual", "target", "ref"), default="actual", help="Joint columns to replay.")
  p.add_argument("--onnx-path", type=Path, default=None, help="ONNX to source root body from (else meta.json).")
  p.add_argument("--root-body", type=str, default="pelvis", help="Root body name if no ONNX is available.")
  p.add_argument("--base-height", type=float, default=0.793, help="Fixed base height (m); base xy is pinned at origin.")
  p.add_argument(
    "--use-recorded-quat",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Animate base orientation from the recorded IMU quaternion.",
  )
  p.add_argument("--replay-speed", type=float, default=1.0, help="Playback speed multiplier.")
  p.add_argument("--loop", action="store_true", help="Loop playback.")
  p.add_argument("--no-viewer", action="store_true", help="Print summary only, no MuJoCo window.")
  return p


def main() -> None:
  args = _build_argparser().parse_args()
  csv_path, onnx_path = _resolve_inputs(args)
  run(
    csv_path=csv_path,
    task_id=str(args.task_id),
    source=str(args.source),
    onnx_path=onnx_path,
    root_body=str(args.root_body),
    base_height=float(args.base_height),
    use_recorded_quat=bool(args.use_recorded_quat),
    replay_speed=float(args.replay_speed),
    loop=bool(args.loop),
    no_viewer=bool(args.no_viewer),
  )


if __name__ == "__main__":
  main()
