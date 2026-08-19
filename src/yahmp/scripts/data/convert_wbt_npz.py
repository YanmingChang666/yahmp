"""Convert a `whole_body_tracking` (Isaac Lab) motion npz into a YAHMP-native npz.

`whole_body_tracking/scripts/csv_to_npz.py` writes `joint_pos` and the `body_*`
arrays by reading back `robot.data.joint_pos` / `robot.data.body_pos_w`, which are
in **Isaac Lab DOF / body order**. YAHMP consumes an npz clip positionally, in its
own (mjlab) joint order — the order in `config/chingmu_joint_order.json`, which is
also the order of the source LAFAN1/Unitree CSV. The two orders differ, so feeding
a raw whole_body_tracking npz scrambles the per-joint reference (arms get
leg/wrist targets → they twist into the body and the robot cannot stand).

This rewrites the clip into YAHMP joint order and drops the Isaac-Lab-ordered
`body_*` arrays, keeping only `root_pos` + `root_quat_w` (pelvis = Isaac body 0).
The YAHMP loader then reconstructs `body_pos_w`/`body_quat_w`/`body_names` with its
own G1 forward kinematics, so everything lands in YAHMP order.

Usage:
    uv run python -m yahmp.scripts.data.convert_wbt_npz \
        --input  whole_body_tracking_motions/yuanditabu.npz \
        --output assets/motions/wbt/yuanditabu.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# YAHMP / mjlab G1 joint order (== whole_body_tracking CSV column order,
# == config/chingmu_joint_order.json).
YAHMP_JOINTS: tuple[str, ...] = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

# Isaac Lab DOF order emitted by `robot.data.joint_pos` for the G1 in csv_to_npz.py
# (breadth-first by kinematic depth, left/right/waist interleaved). Verified
# empirically by matching yuanditabu.npz frame 0 against its source CSV.
ISAAC_JOINTS: tuple[str, ...] = (
  "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
  "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
  "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
  "left_knee_joint", "right_knee_joint",
  "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint", "right_ankle_pitch_joint",
  "left_shoulder_roll_joint", "right_shoulder_roll_joint",
  "left_ankle_roll_joint", "right_ankle_roll_joint",
  "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
  "left_elbow_joint", "right_elbow_joint",
  "left_wrist_roll_joint", "right_wrist_roll_joint",
  "left_wrist_pitch_joint", "right_wrist_pitch_joint",
  "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

# joint_pos_yahmp[:, i] = joint_pos_isaac[:, ISAAC_TO_YAHMP[i]]
ISAAC_TO_YAHMP: list[int] = [ISAAC_JOINTS.index(n) for n in YAHMP_JOINTS]


def convert(inp: Path, outp: Path) -> None:
  data = np.load(inp)
  if set(ISAAC_JOINTS) != set(YAHMP_JOINTS):
    raise AssertionError("Joint-name sets differ; check the order tables.")

  jp = np.asarray(data["joint_pos"], dtype=np.float32)
  if jp.shape[1] != 29:
    raise ValueError(f"Expected 29 DOF in {inp}, got {jp.shape[1]}.")
  jp = jp[:, ISAAC_TO_YAHMP]

  out: dict[str, np.ndarray] = {
    "fps": np.asarray([float(np.asarray(data["fps"]).reshape(-1)[0])], dtype=np.float64),
    "joint_pos": jp,
    # Pelvis is Isaac body 0; body_quat_w is already wxyz (YAHMP default).
    "root_pos": np.asarray(data["body_pos_w"][:, 0, :], dtype=np.float64),
    "root_quat_w": np.asarray(data["body_quat_w"][:, 0, :], dtype=np.float64),
  }
  if "joint_vel" in data:
    out["joint_vel"] = np.asarray(data["joint_vel"], dtype=np.float32)[:, ISAAC_TO_YAHMP]

  outp.parent.mkdir(parents=True, exist_ok=True)
  np.savez(outp, **out)
  print(f"[convert] {inp}  ->  {outp}")
  print(f"[convert] frames={jp.shape[0]}  fps={out['fps'][0]:.1f}  joints=29 (YAHMP order)")


def _parse() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--input", type=Path, required=True, help="whole_body_tracking npz (Isaac Lab order).")
  p.add_argument("--output", type=Path, required=True, help="Destination YAHMP-native npz.")
  return p.parse_args()


def main() -> None:
  a = _parse()
  convert(a.input.expanduser(), a.output.expanduser())


if __name__ == "__main__":
  main()
