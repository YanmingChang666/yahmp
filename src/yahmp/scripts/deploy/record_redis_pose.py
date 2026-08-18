"""Record GMR-retargeted poses from redis to CSV, for comparing retarget params.

Runs in the **yahmp** env. In `--mode replay` the deploy writes the published
pose straight to MuJoCo qpos, so the redis payload *is* the robot pose — logging
it is enough to tell "standing" from "lying" and to compare retarget settings
(`--rot-format`, `--euler-order`, world rotation, …) objectively.

One CSV per retarget-server parameter set:

    # terminal A (gmr env): server with the params under test
    python retarget_server_gmr.py … --rot-format euler --euler-order ZYX

    # terminal B (yahmp env): record ~15 s to a CSV named by those params
    uv run python -m yahmp.scripts.deploy.record_redis_pose \
      --out runs/euler_ZYX.csv --seconds 15 --label "euler ZYX"

Then compare every run at once:

    uv run python -m yahmp.scripts.deploy.record_redis_pose analyze runs/*.csv

Each row is one *new* published frame (deduped by `seq`):

    step, seq, t, root_x, root_y, root_z, height_z,
    quat_w, quat_x, quat_y, quat_z, roll_deg, pitch_deg, yaw_deg, <joint angles…>

roll/pitch/yaw are read from `root_quat` **as wxyz** (the payload's stated
convention); if the server is actually emitting xyzw they will look scrambled —
which is itself a useful signal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


# ── Pure helpers (unit-testable without redis) ───────────────────────────────
def _quat_roll_pitch_yaw(q: np.ndarray) -> np.ndarray:
  """Roll/pitch/yaw (rad) from a wxyz quaternion (matches the deploy pipeline)."""
  q = np.asarray(q, dtype=np.float64)
  n = np.linalg.norm(q)
  if n < 1.0e-12:
    return np.zeros(3, dtype=np.float64)
  w, x, y, z = q / n
  roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
  yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return np.asarray((roll, pitch, yaw), dtype=np.float64)


def csv_header(joint_names: list[str]) -> list[str]:
  h = ["step", "seq", "t", "root_x", "root_y", "root_z", "height_z"]
  h += ["quat_w", "quat_x", "quat_y", "quat_z", "roll_deg", "pitch_deg", "yaw_deg"]
  h += list(joint_names)
  return h


def pose_row(step: int, msg: dict, joint_names: list[str]) -> list:
  """One CSV row from a redis payload dict (root pose + name-keyed joints)."""
  rp = np.asarray(msg["root_pos"], dtype=np.float64)
  rq = np.asarray(msg["root_quat"], dtype=np.float64)  # stated wxyz
  jp = msg.get("joint_pos") or {}
  rpy = np.degrees(_quat_roll_pitch_yaw(rq))
  row: list = [step, msg.get("seq", ""), f"{float(msg.get('t', 0.0)):.6f}"]
  row += [f"{float(rp[i]):.5f}" for i in range(3)]        # root_x, root_y, root_z
  row += [f"{float(rp[2]):.5f}"]                          # height_z (convenience dup)
  row += [f"{float(rq[i]):.6f}" for i in range(4)]        # quat wxyz
  row += [f"{float(rpy[i]):.3f}" for i in range(3)]       # roll, pitch, yaw (deg)
  row += [f"{float(jp.get(n, float('nan'))):.5f}" for n in joint_names]
  return row


# ── Record ───────────────────────────────────────────────────────────────────
def cmd_record(args: argparse.Namespace) -> None:
  import redis  # lazy: only needed to record

  client = redis.Redis(host=args.host, port=args.port, db=args.db)
  try:
    client.ping()
  except Exception as exc:  # noqa: BLE001 - clear message, no stack
    raise SystemExit(
      f"Cannot reach redis at {args.host}:{args.port}: {exc}. Start redis and the "
      "retarget server (retarget_server_gmr.py) first."
    )

  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  out = args.out or f"redis_pose_{ts}.csv"
  out_dir = os.path.dirname(out)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  meta = dict(
    label=args.label, key=args.key, host=args.host, port=args.port, db=args.db,
    seconds=args.seconds, poll_hz=args.poll_hz, timestamp=ts,
  )
  meta_path = out[:-4] + ".meta.json" if out.endswith(".csv") else out + ".meta.json"
  Path(meta_path).write_text(json.dumps(meta, indent=2))

  poll_dt = 1.0 / max(args.poll_hz, 1.0)
  joint_names: Optional[list[str]] = None
  last_seq = None
  step = 0
  waited = False
  t0 = time.perf_counter()
  print(f"[rec] logging {args.key!r} -> {out}  ({args.seconds:.0f}s, label={args.label!r})")
  with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    while args.seconds <= 0 or time.perf_counter() - t0 < args.seconds:
      raw = client.get(args.key)
      if raw is None:
        if not waited:
          print(f"[rec] waiting for a frame at {args.key!r}…")
          waited = True
        time.sleep(poll_dt)
        continue
      if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
      try:
        msg = json.loads(raw)
      except (ValueError, TypeError):
        time.sleep(poll_dt)
        continue
      seq = msg.get("seq")
      if seq is not None and seq == last_seq:
        time.sleep(poll_dt)
        continue
      last_seq = seq
      if joint_names is None:
        joint_names = sorted((msg.get("joint_pos") or {}).keys())
        w.writerow(csv_header(joint_names))
      w.writerow(pose_row(step, msg, joint_names))
      f.flush()
      step += 1
      time.sleep(poll_dt)
  print(f"[rec] done: {step} frames -> {out}")


# ── Analyze / compare ─────────────────────────────────────────────────────────
def _col(rows: list[dict], name: str) -> np.ndarray:
  vals = [float(r[name]) for r in rows if r.get(name, "") not in ("", "nan")]
  return np.asarray(vals, dtype=np.float64)


def _label_for(path: str) -> str:
  meta_path = path[:-4] + ".meta.json" if path.endswith(".csv") else path + ".meta.json"
  try:
    return json.loads(Path(meta_path).read_text()).get("label") or ""
  except Exception:  # noqa: BLE001
    return ""


def cmd_analyze(args: argparse.Namespace) -> None:
  print("=" * 100)
  print(f"  {'file':30s} {'roll°(μ±σ)':>14s} {'pitch°(μ±σ)':>14s} {'yaw°(μ)':>8s} {'height':>8s}  verdict")
  print("  " + "-" * 96)
  for path in args.csv:
    try:
      rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    except FileNotFoundError:
      print(f"  {os.path.basename(path):30s}  (not found)")
      continue
    if not rows:
      print(f"  {os.path.basename(path):30s}  (empty)")
      continue
    roll, pitch, yaw = _col(rows, "roll_deg"), _col(rows, "pitch_deg"), _col(rows, "yaw_deg")
    h = _col(rows, "height_z")
    # Standing ⇒ pelvis ~upright (small roll/pitch) at a plausible pelvis height.
    upright = (
      roll.size and abs(roll.mean()) < 15.0 and abs(pitch.mean()) < 15.0
      and 0.55 < h.mean() < 0.95
    )
    verdict = "STANDS ✓" if upright else "lying/tilted ✗"
    lbl = _label_for(path)
    name = os.path.basename(path) + (f"  [{lbl}]" if lbl else "")
    print(
      f"  {name:30.30s} {roll.mean():+7.1f}±{roll.std():4.1f} {pitch.mean():+7.1f}±{pitch.std():4.1f} "
      f"{yaw.mean():+7.1f} {h.mean():8.3f}  {verdict}"
    )
  print("=" * 100)
  print("  standing target: roll≈0, pitch≈0, height≈0.75 m.  large roll/pitch or low height ⇒ still lying.")


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Record/compare GMR redis poses as CSV.")
  sub = p.add_subparsers(dest="cmd")
  pr = sub.add_parser("record", help="poll the redis key and log poses to CSV (default)")
  pr.add_argument("--out", type=str, default="", help="CSV path (default redis_pose_<ts>.csv). Name it by the params under test.")
  pr.add_argument("--label", type=str, default="", help="Free-text label for this run (stored in a sidecar .meta.json).")
  pr.add_argument("--seconds", type=float, default=15.0, help="Record duration (<=0 = until Ctrl-C).")
  pr.add_argument("--key", type=str, default="yahmp:mocap:pose")
  pr.add_argument("--host", type=str, default="localhost")
  pr.add_argument("--port", type=int, default=6379)
  pr.add_argument("--db", type=int, default=0)
  pr.add_argument("--poll-hz", type=float, default=100.0)
  pa = sub.add_parser("analyze", help="compare one or more recorded CSVs")
  pa.add_argument("csv", nargs="+", help="CSV files to compare")
  return p


def main() -> None:
  argv = sys.argv[1:]
  if not argv or argv[0] not in ("record", "analyze", "-h", "--help"):
    argv = ["record", *argv]  # `record` is the default subcommand
  args = _build_argparser().parse_args(argv)
  if args.cmd == "analyze":
    cmd_analyze(args)
  else:
    cmd_record(args)


if __name__ == "__main__":
  main()
