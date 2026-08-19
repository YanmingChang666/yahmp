"""Shared glue for driving the TWIST2 policy from live mocap (sim2sim + sim2real).

The stock TWIST2 runner (`run_twist2_onnx_mujoco.py`) tracks a pre-recorded clip.
`run_twist2_onnx_mocap.py` (MuJoCo) and `run_twist2_onnx_real.py` (Unitree G1)
instead drive the same policy from the live ChingMu stream, reusing two building
blocks unchanged:

  * TWIST2 observation + control  — `run_twist2_onnx_mujoco.py`
  * mocap source + `LiveReference`— `run_yahmp_onnx_mocap.py`

`LiveReference` is a drop-in replacement for a motion clip: `.sample(t) ->
MotionFrame`. The only friction is that a TWIST2 ONNX carries no YAHMP metadata,
so `PolicySpec.from_onnx` cannot be used. `Twist2Spec` is a tiny shim exposing
exactly the fields the mocap plumbing reads (`joint_names`, `default_joint_pos`,
`root_body_name`, `control_dt`).

The observation assembled here is byte-for-byte identical to
`run_twist2_onnx_mujoco._build_observation` — the proprio block is just sourced
from plain arrays (MjData in sim, encoder/IMU reads on hardware) instead of MjData
directly, so sim2sim and sim2real cannot drift.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from yahmp.scripts.deploy.run_twist2_onnx_mujoco import (
  ANKLE_DOF_INDICES,
  OBS_DIM_NO_FUTURE,
  OBS_DIM_WITH_CURRENT_AS_FUTURE,
  TWIST2_DEFAULT_DOF_POS,
  TWIST2_G1_JOINT_NAMES,
  Twist2ControlProfile,
  _quat_to_euler_wxyz,
  _twist2_mimic_command,
)
from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import MotionFrame

PELVIS_BODY_NAME = "pelvis"


@dataclass(frozen=True)
class Twist2Spec:
  """Minimal `PolicySpec` shim — only the fields the mocap plumbing reads.

  `LiveReference`/`MocapState`/`NpzMockSource`/`MocapCalibration` need
  `joint_names`, `default_joint_pos`, and `root_body_name`; the runners also use
  `control_dt` for loop timing and the target smoother.
  """

  control_dt: float
  joint_names: tuple[str, ...] = TWIST2_G1_JOINT_NAMES
  root_body_name: str = PELVIS_BODY_NAME
  default_joint_pos: np.ndarray = field(
    default_factory=lambda: TWIST2_DEFAULT_DOF_POS.copy()
  )


def resolve_future_block(input_dim: int) -> bool:
  """True if the ONNX expects the trailing current-as-future mimic block."""
  if input_dim == OBS_DIM_WITH_CURRENT_AS_FUTURE:
    return True
  if input_dim == OBS_DIM_NO_FUTURE:
    return False
  raise ValueError(
    "Unsupported TWIST2 ONNX input dimension. Expected "
    f"{OBS_DIM_WITH_CURRENT_AS_FUTURE} (current-as-future) or "
    f"{OBS_DIM_NO_FUTURE} (no future), got {input_dim}."
  )


def twist2_proprio_from_state(
  dof_pos: np.ndarray,
  dof_vel: np.ndarray,
  quat: np.ndarray,
  ang_vel: np.ndarray,
  last_action: np.ndarray,
  *,
  zero_ankle_vel: bool,
) -> np.ndarray:
  """92-dim TWIST2 proprio block from raw robot state (matches `_twist2_proprio`)."""
  rpy = _quat_to_euler_wxyz(np.asarray(quat, dtype=np.float64))
  obs_dof_vel = np.asarray(dof_vel, dtype=np.float64).copy()
  if zero_ankle_vel:
    obs_dof_vel[list(ANKLE_DOF_INDICES)] = 0.0
  return np.concatenate(
    (
      np.asarray(ang_vel, dtype=np.float64) * 0.25,
      rpy[:2],
      np.asarray(dof_pos, dtype=np.float64) - TWIST2_DEFAULT_DOF_POS,
      obs_dof_vel * 0.05,
      np.asarray(last_action, dtype=np.float64),
    )
  ).astype(np.float32)


def build_twist2_obs(
  *,
  frame: MotionFrame,
  dof_pos: np.ndarray,
  dof_vel: np.ndarray,
  quat: np.ndarray,
  ang_vel: np.ndarray,
  last_action: np.ndarray,
  history: deque,
  include_future_block: bool,
  zero_ankle_vel: bool,
) -> tuple[np.ndarray, np.ndarray]:
  """Full TWIST2 observation. Returns `(obs, current)` (current = mimic+proprio).

  Identical layout to `run_twist2_onnx_mujoco._build_observation`; the reference
  `frame` supplies the 35-dim mimic command, the robot state the 92-dim proprio.
  """
  mimic = _twist2_mimic_command(frame)
  proprio = twist2_proprio_from_state(
    dof_pos, dof_vel, quat, ang_vel, last_action, zero_ankle_vel=zero_ankle_vel
  )
  current = np.concatenate((mimic, proprio)).astype(np.float32)
  obs_parts = [current, np.asarray(history, dtype=np.float32).reshape(-1)]
  if include_future_block:
    obs_parts.append(mimic)
  return np.concatenate(obs_parts).astype(np.float32), current


def decode_pd_target(
  raw_action: np.ndarray,
  profile: Twist2ControlProfile,
  clip_actions: float,
) -> np.ndarray:
  """Policy action -> per-joint position target (TWIST2 default + scaled action)."""
  action = np.clip(np.asarray(raw_action, dtype=np.float64), -clip_actions, clip_actions)
  return TWIST2_DEFAULT_DOF_POS + action * profile.action_scale


def recorder_root_cmd(frame: MotionFrame) -> np.ndarray:
  """`[vx, vy, wyaw, height, roll, pitch]` for the CSV recorder, from the mimic."""
  mimic = _twist2_mimic_command(frame)  # [vx,vy, height, roll,pitch, wyaw, joints...]
  return np.array(
    [mimic[0], mimic[1], mimic[5], mimic[2], mimic[3], mimic[4]], dtype=np.float64
  )
