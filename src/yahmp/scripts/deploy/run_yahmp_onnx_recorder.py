"""Sim2real data recorder + analysis tool for the YAHMP ONNX real-robot loop.

Inspired by Beyondmimic's `record_run_csv.py`, but stripped of the table-tennis
ball / LiDAR / odometry columns (which do not exist in YAHMP whole-body teleop)
and refocused on the signals that matter for sim/real parity analysis:

  * motor tracking      target vs actual joint position  (PD / gain adequacy)
  * reference tracking  mocap ref vs actual joint position (imitation quality)
  * base attitude       quaternion, euler, gyro, projected gravity (balance/drift)
  * loop timing         inference / step / period ms + overrun (real-time health)
  * raw action          policy output per action-target joint

Two artifacts per run (`{prefix}_{ts}_*`):
  * `_full.csv`   one row per control step (typically 50 Hz)
  * `_meta.json`  run configuration (onnx path, gains, dry-run, source, dt …)

Integration (already wired into `run_yahmp_onnx_real.py` via `--record`):
    from yahmp.scripts.deploy.run_yahmp_onnx_recorder import Sim2RealRecorder
    recorder = Sim2RealRecorder(prefix, spec, meta=...) if prefix else None
    # each control step, after send_targets():
    recorder.step(
        wall_time      = t0,
        ref_joint_pos  = frame.joint_pos,
        target_joint_pos = target,
        actual_joint_pos = qj,
        actual_joint_vel = dqj,
        quat           = quat,          # wxyz
        ang_vel        = gyro,
        proj_grav      = terms["projected_gravity"],
        raw_action     = raw_action,
        timing         = {"infer_ms": ..., "step_ms": ..., "period_ms": ..., "overrun": ...},
    )
    # in the finally block:
    recorder.close()

Standalone analysis of a saved file:
    uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_recorder analyze run_..._full.csv
    uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_recorder info    run_..._full.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Timing keys written to the CSV, in column order (overrun handled separately).
_TIMING_KEYS = ("infer_ms", "step_ms", "period_ms")


def _quat_roll_pitch_yaw(q: np.ndarray) -> np.ndarray:
  """Roll/pitch/yaw (rad) from a wxyz quaternion.

  Kept self-contained (rather than importing from the MuJoCo pipeline) so the
  `analyze`/`info` CLI can read logs without the sim stack installed. Matches
  `run_yahmp_onnx_mujoco._quat_roll_pitch_yaw`.
  """
  q = np.asarray(q, dtype=np.float64)
  norm = np.linalg.norm(q)
  if norm < 1.0e-12:
    return np.zeros(3, dtype=np.float64)
  w, x, y, z = q / norm
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return np.asarray((roll, pitch, yaw), dtype=np.float64)


class Sim2RealRecorder:
  """Stream one CSV row per control step; write a meta.json alongside.

  Rows are flushed each step so an unexpected interrupt (E-stop, crash) keeps
  everything recorded up to that point.
  """

  def __init__(
    self,
    prefix: str,
    joint_names: tuple[str, ...] | list[str],
    action_target_names: tuple[str, ...] | list[str] | None = None,
    meta: Optional[dict] = None,
  ) -> None:
    """
    Args:
        prefix: Output filename prefix (no extension). Real files are
            `{prefix}_{ts}_full.csv` and `{prefix}_{ts}_meta.json`.
        joint_names: Policy joint order (len == num_joints).
        action_target_names: Names for the raw-action columns. Falls back to
            positional `action_i` labels if omitted or length mismatches.
        meta: Run-config dict (onnx path, gains, dry-run, source, dt …) written
            to meta.json so runs can be told apart after the fact.
    """
    self._names = list(joint_names)
    self._n = len(self._names)
    self._action_names = list(action_target_names) if action_target_names else None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    self._full_path = f"{prefix}_{ts}_full.csv"
    self._meta_path = f"{prefix}_{ts}_meta.json"

    out_dir = os.path.dirname(self._full_path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)

    # ── Run metadata (to distinguish policies / gains after the fact) ────────
    if meta is not None:
      try:
        m = dict(meta)
        m.setdefault("timestamp", ts)
        m["program"] = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
        m.setdefault("num_joints", self._n)
        with open(self._meta_path, "w", encoding="utf-8") as mf:
          json.dump(m, mf, ensure_ascii=False, indent=2, default=str)
        print(f"[Recorder] run config -> {self._meta_path}")
      except Exception as exc:  # noqa: BLE001 - recording must never crash the loop
        print(f"[Recorder] warning: failed to write meta.json: {exc}")

    # `_action_dim` is fixed on the first step() from the raw_action length.
    self._action_dim: Optional[int] = None
    self._full_f = open(self._full_path, "w", newline="", encoding="utf-8")
    self._full_w = csv.writer(self._full_f)
    self._header_written = False

    self._step = 0
    self._t0: Optional[float] = None

    print(f"[Recorder] full data -> {self._full_path}")
    print(f"[Recorder] joints: {self._n}  ({', '.join(self._names[:4])} …)")

  # ── Header ─────────────────────────────────────────────────────────────────
  def _action_labels(self) -> list[str]:
    if self._action_names is not None and len(self._action_names) == self._action_dim:
      return list(self._action_names)
    return [f"action_{i}" for i in range(int(self._action_dim or 0))]

  def _full_header(self) -> list[str]:
    names = self._names
    h = ["step", "wall_time", "rel_time"]
    # Base attitude
    h += ["quat_w", "quat_x", "quat_y", "quat_z"]
    h += ["euler_roll", "euler_pitch", "euler_yaw"]
    h += ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
    h += ["proj_grav_x", "proj_grav_y", "proj_grav_z"]
    # Root terms of the motion command (what the policy is told the base should do)
    h += ["cmd_anchor_vx", "cmd_anchor_vy", "cmd_anchor_wyaw"]
    h += ["cmd_root_height", "cmd_root_roll", "cmd_root_pitch"]
    # Per-joint signals
    h += [f"ref_{n}" for n in names]        # mocap reference (what the human does)
    h += [f"target_{n}" for n in names]     # position commanded to the motor
    h += [f"actual_{n}" for n in names]     # measured joint position
    h += [f"vel_{n}" for n in names]        # measured joint velocity
    h += [f"err_{n}" for n in names]        # |target - actual|  (PD tracking)
    h += [f"track_err_{n}" for n in names]  # |ref - actual|     (imitation)
    # Raw policy action
    h += self._action_labels()
    # Loop timing
    h += ["t_infer_ms", "t_step_ms", "t_period_ms", "overrun"]
    return h

  # ── Per-step ─────────────────────────────────────────────────────────────
  def step(
    self,
    *,
    wall_time: float,
    ref_joint_pos: np.ndarray,
    target_joint_pos: np.ndarray,
    actual_joint_pos: np.ndarray,
    actual_joint_vel: np.ndarray,
    quat: np.ndarray,
    ang_vel: np.ndarray,
    proj_grav: np.ndarray,
    raw_action: np.ndarray,
    root_cmd: Optional[np.ndarray] = None,
    timing: Optional[dict] = None,
  ) -> None:
    """Append one control step to the CSV.

    `root_cmd` is the 6 root terms of the motion command in native order
    `[anchor_vx, anchor_vy, anchor_wyaw, root_height, root_roll, root_pitch]`
    (e.g. `command[2*N : 2*N+6]`); pass `None` to log them as `nan`.
    """
    if self._t0 is None:
      self._t0 = wall_time
    rel_time = wall_time - self._t0

    n = self._n
    ref = np.asarray(ref_joint_pos, dtype=np.float64)[:n]
    tp = np.asarray(target_joint_pos, dtype=np.float64)[:n]
    ap = np.asarray(actual_joint_pos, dtype=np.float64)[:n]
    av = np.asarray(actual_joint_vel, dtype=np.float64)[:n]
    err = np.abs(tp - ap)
    track_err = np.abs(ref - ap)

    q = np.asarray(quat, dtype=np.float64)
    euler = _quat_roll_pitch_yaw(q)
    gyro = np.asarray(ang_vel, dtype=np.float64)
    pg = np.asarray(proj_grav, dtype=np.float64)
    rc = (
      np.full(6, np.nan)
      if root_cmd is None
      else np.asarray(root_cmd, dtype=np.float64).reshape(-1)[:6]
    )
    action = np.asarray(raw_action, dtype=np.float64).reshape(-1)

    # Fix the action width and write the header on the first row.
    if not self._header_written:
      self._action_dim = int(action.shape[0])
      self._full_w.writerow(self._full_header())
      self._header_written = True

    row: list = [self._step, f"{wall_time:.6f}", f"{rel_time:.6f}"]
    row += [f"{float(q[i]):.5f}" for i in range(4)]
    row += [f"{float(euler[i]):.5f}" for i in range(3)]
    row += [f"{float(gyro[i]):.4f}" for i in range(3)]
    row += [f"{float(pg[i]):.5f}" for i in range(3)]
    row += [f"{float(rc[i]):.5f}" for i in range(6)]
    row += [f"{v:.5f}" for v in ref]
    row += [f"{v:.5f}" for v in tp]
    row += [f"{v:.5f}" for v in ap]
    row += [f"{v:.5f}" for v in av]
    row += [f"{v:.5f}" for v in err]
    row += [f"{v:.5f}" for v in track_err]
    row += [f"{v:.5f}" for v in action[: self._action_dim]]
    # Pad/truncate defensively so the row width never drifts from the header.
    if action.shape[0] < self._action_dim:
      row += ["nan"] * (self._action_dim - action.shape[0])
    if timing is not None:
      row += [f"{float(timing.get(k, float('nan'))):.3f}" for k in _TIMING_KEYS]
      row += [int(bool(timing.get("overrun", False)))]
    else:
      row += ["nan"] * len(_TIMING_KEYS) + [""]

    self._full_w.writerow(row)
    self._full_f.flush()
    self._step += 1

  # ── Close ────────────────────────────────────────────────────────────────
  def close(self) -> None:
    if not self._full_f.closed:
      self._full_f.close()
    print(f"[Recorder] done: {self._step} steps -> {self._full_path}")

  @property
  def count(self) -> int:
    return self._step


# =============================================================================
# Standalone analysis
# =============================================================================
def _load_full_csv(path: str) -> tuple[list[str], list[dict]]:
  with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames or []
    rows = list(reader)
  return header, rows


def _col_floats(rows: list[dict], col: str) -> np.ndarray:
  vals = [float(r[col]) for r in rows if r.get(col, "") not in ("", "nan")]
  return np.asarray(vals, dtype=np.float64)


def cmd_info(path: str) -> None:
  header, rows = _load_full_csv(path)
  if not rows:
    print("[Info] empty file")
    return
  t_start = float(rows[0]["rel_time"])
  t_end = float(rows[-1]["rel_time"])
  duration = t_end - t_start
  hz = len(rows) / max(duration, 1e-6)
  joint_cols = [c for c in header if c.startswith("target_")]
  overruns = sum(1 for r in rows if r.get("overrun", "") == "1")

  print("=" * 60)
  print(f"  file      : {path}")
  print(f"  steps     : {len(rows)}")
  print(f"  duration  : {duration:.1f}s   effective rate ~= {hz:.1f} Hz")
  print(f"  joints    : {len(joint_cols)}")
  if any(r.get("overrun", "") not in ("", "nan") for r in rows):
    print(f"  overruns  : {overruns}/{len(rows)} ({100 * overruns / max(len(rows), 1):.1f}%)")
  print("=" * 60)


def _print_error_table(title: str, prefix: str, header: list[str], rows: list[dict]) -> None:
  names = [c[len(prefix):] for c in header if c.startswith(prefix)]
  if not names:
    return
  print("=" * 72)
  print(f"  {title}   (degrees)")
  print(f"  {'joint':>22s}  {'mean':>9s}  {'max':>9s}  {'std':>9s}")
  print("  " + "-" * 56)
  for name in names:
    arr = _col_floats(rows, f"{prefix}{name}")
    if arr.size == 0:
      continue
    mean_d, max_d, std_d = (
      math.degrees(arr.mean()),
      math.degrees(arr.max()),
      math.degrees(arr.std()),
    )
    flag = " <- large" if mean_d > 5.0 else ""
    print(f"  {name:>22s}  {mean_d:>8.2f}°  {max_d:>8.2f}°  {std_d:>8.2f}°{flag}")


def cmd_analyze(path: str) -> None:
  header, rows = _load_full_csv(path)
  if not rows:
    print("[Analyze] empty file")
    return

  _print_error_table("motor tracking error |target - actual|", "err_", header, rows)
  _print_error_table("reference tracking error |ref - actual|", "track_err_", header, rows)

  # ── Base attitude ─────────────────────────────────────────────────────────
  print("=" * 72)
  pgz = _col_floats(rows, "proj_grav_z")
  if pgz.size:
    print(
      f"  proj_grav_z mean={pgz.mean():+.4f}  "
      f"(upright ideal ~= -1.0, offset={pgz.mean() + 1.0:+.3f})"
    )
  for axis in ("roll", "pitch"):
    arr = _col_floats(rows, f"euler_{axis}")
    if arr.size:
      print(
        f"  euler_{axis:5s} mean={math.degrees(arr.mean()):+.2f}°  "
        f"range=[{math.degrees(arr.min()):+.2f}°, {math.degrees(arr.max()):+.2f}°]  "
        f"std={math.degrees(arr.std()):.2f}°"
      )

  # ── Command root terms (calibration check: roll/pitch ~ 0 when operator neutral)
  for label, col in (("cmd_root_roll", "cmd_root_roll"), ("cmd_root_pitch", "cmd_root_pitch")):
    arr = _col_floats(rows, col)
    if arr.size:
      print(
        f"  {label:14s} mean={math.degrees(arr.mean()):+.2f}°  "
        f"std={math.degrees(arr.std()):.2f}°  (calibrated neutral ~= 0°)"
      )
  h_arr = _col_floats(rows, "cmd_root_height")
  if h_arr.size:
    print(f"  cmd_root_height mean={h_arr.mean():+.3f} m  std={h_arr.std():.3f} m")

  # ── Loop timing ───────────────────────────────────────────────────────────
  step_ms = _col_floats(rows, "t_step_ms")
  period_ms = _col_floats(rows, "t_period_ms")
  if step_ms.size:
    print(
      f"  loop step   mean={step_ms.mean():.2f} ms  max={step_ms.max():.2f} ms"
    )
  if period_ms.size:
    overruns = sum(1 for r in rows if r.get("overrun", "") == "1")
    print(
      f"  loop period mean={period_ms.mean():.2f} ms  max={period_ms.max():.2f} ms  "
      f"overruns={overruns}/{len(rows)}"
    )
  print("=" * 72)

  _plot(path, header, rows)


def _plot(path: str, header: list[str], rows: list[dict]) -> None:
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("\n  (matplotlib not installed, skipping plot)")
    return

  try:
    t = _col_floats(rows, "rel_time")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(os.path.basename(path), fontsize=10)

    # 1: per-joint motor tracking error
    ax = axes[0]
    for col in [c for c in header if c.startswith("err_")]:
      ax.plot(t, np.degrees(_col_floats(rows, col)), alpha=0.7, linewidth=0.8, label=col[len("err_"):])
    ax.axhline(5.0, color="red", linestyle="--", linewidth=0.8)
    ax.set_ylabel("|target-actual| (deg)")
    ax.set_title("motor tracking error (red = 5°)")
    ax.legend(fontsize=6, ncol=4, loc="upper right")
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    # 2: base attitude
    ax = axes[1]
    for axis, color in (("roll", "blue"), ("pitch", "green")):
      arr = _col_floats(rows, f"euler_{axis}")
      if arr.size:
        ax.plot(t, np.degrees(arr), label=f"euler_{axis}", color=color, linewidth=0.8)
    ax.set_ylabel("euler (deg)")
    ax2 = ax.twinx()
    pgz = _col_floats(rows, "proj_grav_z")
    if pgz.size:
      ax2.plot(t, pgz, label="proj_grav_z", color="orange", alpha=0.8, linewidth=0.8)
      ax2.axhline(-1.0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("proj_grav_z")
    ax.set_title("base attitude (proj_grav_z ideal ~= -1)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3: loop timing
    ax = axes[2]
    for col, color in (("t_step_ms", "blue"), ("t_period_ms", "green")):
      arr = _col_floats(rows, col)
      if arr.size:
        ax.plot(t[: arr.size], arr, label=col, color=color, linewidth=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ms")
    ax.set_title("control loop timing")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = path.replace(".csv", "_plot.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  plot saved -> {out}")
  except Exception as exc:  # noqa: BLE001
    print(f"\n  plot failed: {exc}")


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="YAHMP sim2real CSV recorder / analysis tool.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  sub = p.add_subparsers(dest="cmd", required=True)
  pa = sub.add_parser("analyze", help="tracking-error + attitude + timing stats and plot")
  pa.add_argument("csv", type=str, help="path to a *_full.csv file")
  pi = sub.add_parser("info", help="basic stats")
  pi.add_argument("csv", type=str, help="path to a *_full.csv file")
  return p


def main() -> None:
  args = _build_argparser().parse_args()
  if args.cmd == "analyze":
    cmd_analyze(args.csv)
  elif args.cmd == "info":
    cmd_info(args.csv)


if __name__ == "__main__":
  main()
