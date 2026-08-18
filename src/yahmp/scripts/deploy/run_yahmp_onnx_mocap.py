"""Real-time mocap teleoperation of a YAHMP ONNX policy in MuJoCo (sim2sim).

Unlike `run_yahmp_onnx_mujoco.py` (which tracks a pre-recorded `.npz` clip),
this script builds the policy's motion **command** LIVE from a ChingMu mocap
stream. The simulated G1 is driven by the ONNX policy, which tries to track the
human motion the mocap system retargets in real time.

Pipeline (reused verbatim from `run_yahmp_onnx_mujoco.py`):

    robot state (from MuJoCo sim) ─┐
                                   ├─► observation ─► ONNX ─► action ─► sim
    command (from mocap, LIVE) ────┘

The only new pieces here are:
  * `ChingMuMocapSource` — reads root pose + retargeted joint angles over VRPN
    (adapted from the TTRL `CMVrpnWrapper`), and
  * `LiveReference`      — turns the latest streamed pose into a `MotionFrame`,
    finite-differencing joint/root velocities.

To go **sim2real**, replace the MuJoCo read/write with the Unitree SDK exactly
as described in `DEPLOYMENT.md` §4 — the mocap/command half is identical.

------------------------------------------------------------------------------
Device-free test (no mocap hardware): stream an existing clip as if it were live
    uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
      --task-id Mjlab-YAHMP-Unitree-G1 \
      --onnx-path assets/models/g1_yahmp.onnx \
      --source npz \
      --npz-clip assets/motions/g1_omomo_amass_clean/<motion>.npz

Live ChingMu teleoperation (VRPN library vendored at third_party/chingmu/,
loaded by default; override with --chingmu-dll)
    uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
      --task-id Mjlab-YAHMP-Unitree-G1 \
      --onnx-path assets/models/g1_yahmp.onnx \
      --source chingmu \
      --chingmu-host MCAvatar@192.168.123.112 \
      --sensor-root 0 --sensor-joint-first 1 \
      --joint-order config/chingmu_joint_order.json
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

# Reuse the exact deployment pipeline — do NOT reimplement observation/action.
from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import (
  MotionFrame,
  PolicySpec,
  _actuator_ids,
  _append_history,
  _build_observation,
  _build_task_scene,
  _configure_camera,
  _create_onnx_session,
  _initialize_history,
  _initialize_state,
  _joint_addresses,
  _quat_conj,
  _quat_mul,
  _quat_normalize,
  _quat_roll_pitch_yaw,
  _root_addresses,
  _term_values,
)
from yahmp.scripts.deploy.run_yahmp_onnx_recorder import Sim2RealRecorder


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
  """Roll/pitch/yaw (rad) → wxyz quaternion, inverse of `_quat_roll_pitch_yaw`.

  Builds R = Rz(yaw)·Ry(pitch)·Rx(roll) so that feeding the result back through
  `_quat_roll_pitch_yaw` returns the same angles — this makes the calibration's
  root R/P/Y offset read back exactly on the command's root_roll/pitch.
  """
  cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
  cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
  cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
  return np.array(
    [
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ],
    dtype=np.float64,
  )

# Vendored ChingMu VRPN library (repo `third_party/chingmu/`). Resolved relative
# to this file so it works from any CWD; override with `--chingmu-dll`.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_VENDORED_CHINGMU_DLL = _REPO_ROOT / "third_party" / "chingmu" / "libCMVrpn.so"


# ══════════════════════════════════════════════════════════════════════════════
# Shared latest-pose slot (thread-safe hand-off from the mocap thread)
# ══════════════════════════════════════════════════════════════════════════════
class MocapState:
  """Latest streamed pose. Velocities are optional; `LiveReference` finite-
  differences whatever is missing."""

  def __init__(self, num_joints: int) -> None:
    self._lock = threading.Lock()
    self.root_pos = np.zeros(3, dtype=np.float64)
    self.root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz
    self.joint_pos = np.zeros(num_joints, dtype=np.float64)
    self.joint_vel: Optional[np.ndarray] = None
    self.root_lin_vel: Optional[np.ndarray] = None
    self.root_ang_vel: Optional[np.ndarray] = None
    self.detected = False

  def update(
    self,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
    *,
    joint_vel: Optional[np.ndarray] = None,
    root_lin_vel: Optional[np.ndarray] = None,
    root_ang_vel: Optional[np.ndarray] = None,
  ) -> None:
    with self._lock:
      self.root_pos = np.asarray(root_pos, dtype=np.float64).copy()
      self.root_quat = _quat_normalize(np.asarray(root_quat, dtype=np.float64))
      self.joint_pos = np.asarray(joint_pos, dtype=np.float64).copy()
      self.joint_vel = None if joint_vel is None else np.asarray(joint_vel, np.float64).copy()
      self.root_lin_vel = (
        None if root_lin_vel is None else np.asarray(root_lin_vel, np.float64).copy()
      )
      self.root_ang_vel = (
        None if root_ang_vel is None else np.asarray(root_ang_vel, np.float64).copy()
      )
      self.detected = True

  def snapshot(self):
    with self._lock:
      return (
        self.root_pos.copy(),
        self.root_quat.copy(),
        self.joint_pos.copy(),
        None if self.joint_vel is None else self.joint_vel.copy(),
        None if self.root_lin_vel is None else self.root_lin_vel.copy(),
        None if self.root_ang_vel is None else self.root_ang_vel.copy(),
        self.detected,
      )


def _root_ang_vel_world(q_prev: np.ndarray, q_cur: np.ndarray, dt: float) -> np.ndarray:
  """World-frame angular velocity from two root orientations (wxyz)."""
  if dt <= 0.0:
    return np.zeros(3, dtype=np.float64)
  # Relative rotation in the world frame: q_rel = q_cur * q_prev^{-1}.
  q_rel = _quat_mul(_quat_normalize(q_cur), _quat_conj(_quat_normalize(q_prev)))
  w = float(np.clip(q_rel[0], -1.0, 1.0))
  xyz = q_rel[1:]
  s = float(np.linalg.norm(xyz))
  if s < 1e-9:
    return np.zeros(3, dtype=np.float64)
  angle = 2.0 * np.arctan2(s, w)  # shortest-arc angle
  axis = xyz / s
  return (axis * angle / dt).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Neutral-pose calibration (operator-standing-still → robot default stance)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class MocapCalibration:
  """Maps the operator's relaxed neutral pose to the robot's own default stance.

  Captured once with `capture_calibration` (operator stands still), then applied
  every frame inside `LiveReference.sample`. Without it, a retargeted "neutral"
  human pose is *not* the robot's stable configuration — e.g. it commands nearly
  straight knees where the robot needs a deep bend, so the robot cannot balance.

    * joint bias   calibrated = raw − joint_offset
                   joint_offset = mean(neutral_joint) − default_joint_pos
                   (constant, so it does NOT perturb finite-differenced vels)
    * root tilt    calibrated_quat = rpy(root_rpy_offset) ⊗ conj(root_quat_neutral) ⊗ raw
                   → neutral pelvis orientation becomes identity, then a constant
                   world-frame R/P/Y correction is added on top. So the command's
                   root_roll/pitch ≈ root_rpy_offset when the operator stands
                   straight — dial it to remove a residual lean / facing error.
    * root height  calibrated_z = raw_z − root_height_neutral + root_height_target

  Assumes a **Z-up** mocap world (ChingMu default): the neutral inverse alone
  aligns the operator to the robot frame, with no extra axis swap.
  """

  joint_offset: np.ndarray        # (N,) policy joint order, radians
  root_quat_neutral: np.ndarray   # (4,) wxyz
  root_height_neutral: float
  root_height_target: float
  root_rpy_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,) rad

  @classmethod
  def load(cls, path: str | Path, spec: PolicySpec) -> "MocapCalibration":
    data = json.loads(Path(path).read_text())
    off = np.asarray(data["joint_offset_rad"], dtype=np.float64)
    names = data.get("joint_names")
    if names is not None and list(names) != list(spec.joint_names):
      missing = [n for n in spec.joint_names if n not in names]
      if missing:
        raise ValueError(f"Calibration is missing joints {missing}.")
      off = off[[list(names).index(n) for n in spec.joint_names]]
    if off.shape != (len(spec.joint_names),):
      raise ValueError(
        f"joint_offset_rad has length {off.shape}, expected ({len(spec.joint_names)},)."
      )
    return cls(
      joint_offset=off,
      root_quat_neutral=_quat_normalize(
        np.asarray(data.get("root_quat_neutral_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
      ),
      root_height_neutral=float(data.get("root_height_neutral", 0.0)),
      root_height_target=float(data.get("root_height_target", 0.0)),
      root_rpy_offset=np.asarray(data.get("root_rpy_offset_rad", [0.0, 0.0, 0.0]), dtype=np.float64),
    )

  def apply(
    self, joint_pos: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jp = joint_pos - self.joint_offset
    rp = root_pos.copy()
    rp[2] = rp[2] - self.root_height_neutral + self.root_height_target
    rq = _quat_mul(_quat_conj(self.root_quat_neutral), _quat_normalize(root_quat))
    if np.any(self.root_rpy_offset):
      rq = _quat_mul(_quat_from_euler(*self.root_rpy_offset), rq)
    return jp, rp, _quat_normalize(rq)


# ══════════════════════════════════════════════════════════════════════════════
# LiveReference: latest mocap pose  ->  MotionFrame  (drop-in `clip` replacement)
# ══════════════════════════════════════════════════════════════════════════════
class LiveReference:
  """Presents the same `.sample(time_s) -> MotionFrame` surface that
  `run_yahmp_onnx_mujoco.MotionClip` does, but sourced from the live stream.

  * Joints are remapped from the source order into `spec.joint_names` order.
  * Missing velocities are finite-differenced against the previous sample.
  * `time_s` is ignored (there is only ever a "now"); for the Future policy the
    same current frame is returned for every step offset (see NOTE below).
  """

  def __init__(
    self,
    state: MocapState,
    spec: PolicySpec,
    source_joint_order: Optional[list[str]],
    *,
    vel_smoothing: float = 0.0,
    calibration: Optional[MocapCalibration] = None,
    track_legs_only: bool = False,
    track_gain: float = 1.0,
  ) -> None:
    self._state = state
    self._spec = spec
    self._alpha = float(np.clip(vel_smoothing, 0.0, 0.99))
    self._calib = calibration
    # Command blend toward default: ref = default + gain*(mocap - default).
    # gain<1 trades imitation accuracy for stability (policy favors balance).
    self._gain = float(np.clip(track_gain, 0.0, 1.0))

    # `--track-legs-only`: hold every non-leg joint (waist + arms) at the robot
    # default and only track the legs (hip/knee/ankle) from mocap, so an off-
    # default upper-body command can't shift the CoM and tip the policy over.
    self._default_jp = np.asarray(spec.default_joint_pos, dtype=np.float64)
    self._hold_idx = (
      np.asarray(
        [
          i
          for i, name in enumerate(spec.joint_names)
          if not any(k in name for k in ("hip", "knee", "ankle"))
        ],
        dtype=np.int32,
      )
      if track_legs_only
      else np.empty(0, dtype=np.int32)
    )

    if source_joint_order is None:
      if len(spec.joint_names) != state.joint_pos.shape[0]:
        raise ValueError(
          "No --joint-order given and source joint count "
          f"({state.joint_pos.shape[0]}) != policy joint count "
          f"({len(spec.joint_names)}). Provide a joint-order mapping."
        )
      self._remap = np.arange(len(spec.joint_names), dtype=np.int32)
      print("[LiveReference] WARNING: assuming identity joint order (unverified).")
    else:
      missing = [n for n in spec.joint_names if n not in source_joint_order]
      if missing:
        raise ValueError(f"Joints missing from --joint-order: {missing}")
      self._remap = np.asarray(
        [source_joint_order.index(n) for n in spec.joint_names], dtype=np.int32
      )

    self._prev: Optional[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = None
    self._vel_jp = np.zeros(len(spec.joint_names), dtype=np.float64)
    self._vel_lin = np.zeros(3, dtype=np.float64)
    self._vel_ang = np.zeros(3, dtype=np.float64)

  def wait_for_detection(self, timeout_s: float = 10.0) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
      if self._state.snapshot()[-1]:
        return
      time.sleep(0.02)
    raise TimeoutError("No mocap frame detected within timeout.")

  def sample(self, time_s: float) -> MotionFrame:
    del time_s  # live: always "now"
    root_pos, root_quat, joints_src, jvel, lvel, avel, _ = self._state.snapshot()
    joint_pos = joints_src[self._remap]

    # Neutral-pose calibration: apply BEFORE finite-differencing so velocities
    # (and the prev-sample cache) stay consistent with the calibrated frame. The
    # joint bias is constant, so joint/root velocities are unaffected.
    if self._calib is not None:
      joint_pos, root_pos, root_quat = self._calib.apply(joint_pos, root_pos, root_quat)

    # Reduce tracking: pull the joint reference toward default (and scale the base
    # roll/pitch the same way), so wild/noisy mocap can't drag the policy over.
    if self._gain != 1.0:
      joint_pos = self._default_jp + self._gain * (joint_pos - self._default_jp)
      roll, pitch, yaw = _quat_roll_pitch_yaw(root_quat)
      root_quat = _quat_from_euler(self._gain * roll, self._gain * pitch, yaw)

    now = time.perf_counter()
    if self._prev is None:
      joint_vel = np.zeros_like(joint_pos) if jvel is None else jvel[self._remap]
      root_lin_vel = np.zeros(3) if lvel is None else lvel
      root_ang_vel = np.zeros(3) if avel is None else avel
    else:
      t_prev, jp_prev, rp_prev, rq_prev = self._prev
      dt = max(now - t_prev, 1e-4)
      fd_jv = (joint_pos - jp_prev) / dt if jvel is None else jvel[self._remap]
      fd_lv = (root_pos - rp_prev) / dt if lvel is None else lvel
      fd_av = _root_ang_vel_world(rq_prev, root_quat, dt) if avel is None else avel
      a = self._alpha
      self._vel_jp = a * self._vel_jp + (1 - a) * fd_jv
      self._vel_lin = a * self._vel_lin + (1 - a) * fd_lv
      self._vel_ang = a * self._vel_ang + (1 - a) * fd_av
      joint_vel, root_lin_vel, root_ang_vel = self._vel_jp, self._vel_lin, self._vel_ang

    self._prev = (now, joint_pos.copy(), root_pos.copy(), root_quat.copy())

    # Legs-only: pin the held (non-leg) joints to default with zero velocity, so
    # the command + residual base both say "hold arms/waist at default".
    if self._hold_idx.size:
      joint_pos = joint_pos.copy()
      joint_vel = np.asarray(joint_vel, dtype=np.float64).copy()
      joint_pos[self._hold_idx] = self._default_jp[self._hold_idx]
      joint_vel[self._hold_idx] = 0.0

    # MotionFrame feeds the policy's `command` term (dim 2N+6 = 64 for g1_yahmp):
    #   joint_pos/joint_vel -> ref joint angles/vels (N=29 each)
    #   root_* -> anchor planar vel (2), yaw rate (1), height (1), roll/pitch (2)
    return MotionFrame(
      joint_pos=joint_pos.astype(np.float64),          # (N,)  = 29
      joint_vel=np.asarray(joint_vel, dtype=np.float64),      # (N,)  = 29
      root_pos_w=root_pos.astype(np.float64),          # (3,)  world position (only z=height used)
      root_quat_w=root_quat.astype(np.float64),        # (4,)  wxyz (roll/pitch + frame for vels)
      root_lin_vel_w=np.asarray(root_lin_vel, dtype=np.float64),  # (3,)  world lin vel (xy used)
      root_ang_vel_w=np.asarray(root_ang_vel, dtype=np.float64),  # (3,)  world ang vel (yaw used)
    )

  # NOTE: the Future variant (`Mjlab-YAHMP-Future-Unitree-G1`) asks for frames at
  # `time + offset*control_dt`, which do not exist in a live stream. `_command_value`
  # only calls `.sample()`, so future steps here collapse to the current pose — an
  # approximation. Prefer the base `Mjlab-YAHMP-Unitree-G1` policy for teleop.


def capture_calibration(
  reference: LiveReference,
  spec: PolicySpec,
  *,
  seconds: float,
  height_target: float,
  out_path: str | Path,
) -> MocapCalibration:
  """Average the live stream over a held neutral pose and write a calibration JSON.

  Run with the operator standing still in a relaxed neutral pose. `reference`
  must be an **uncalibrated** `LiveReference` so the raw (remapped) values are
  captured; the resulting `joint_offset = mean(neutral) − default_joint_pos`
  recenters neutral onto the robot's default stance.
  """
  default = np.asarray(spec.default_joint_pos, dtype=np.float64)
  n = len(spec.joint_names)
  jp_acc = np.zeros(n, dtype=np.float64)
  q_acc = np.zeros(4, dtype=np.float64)
  z_acc = 0.0
  q_ref: Optional[np.ndarray] = None
  frames = 0

  reference.wait_for_detection(timeout_s=15.0)
  print(f"[Calibrate] hold the NEUTRAL standing pose — capturing {seconds:.1f}s…")
  t0 = time.perf_counter()
  while time.perf_counter() - t0 < seconds:
    f = reference.sample(0.0)
    jp_acc += f.joint_pos
    z_acc += float(f.root_pos_w[2])
    q = _quat_normalize(np.asarray(f.root_quat_w, dtype=np.float64))
    if q_ref is None:
      q_ref = q
    if float(np.dot(q, q_ref)) < 0.0:  # keep all samples in one hemisphere
      q = -q
    q_acc += q
    frames += 1
    time.sleep(max(spec.control_dt, 1e-3))

  if frames == 0:
    raise RuntimeError("No mocap frames captured during calibration.")

  jp_mean = jp_acc / frames
  q_mean = _quat_normalize(q_acc / frames)
  z_mean = z_acc / frames
  offset = jp_mean - default

  out = {
    "_comment": (
      "Neutral-pose mocap calibration (Z-up). calibrated = raw - joint_offset; "
      "root orientation zeroed via conj(root_quat_neutral); height retargeted."
    ),
    "joint_names": list(spec.joint_names),
    "joint_offset_rad": [float(v) for v in offset],
    "root_quat_neutral_wxyz": [float(v) for v in q_mean],
    "root_height_neutral": float(z_mean),
    "root_height_target": float(height_target),
    "root_rpy_offset_rad": [0.0, 0.0, 0.0],
    "captured_frames": int(frames),
  }
  out_path = Path(out_path)
  if out_path.parent and not out_path.parent.exists():
    out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(out, indent=2))

  worst = sorted(
    zip(spec.joint_names, offset), key=lambda kv: -abs(kv[1])
  )[:6]
  print(f"[Calibrate] captured {frames} frames -> {out_path}")
  print(f"[Calibrate] root height neutral={z_mean:+.3f} m  target={height_target:+.3f} m")
  print("[Calibrate] largest joint offsets (subtracted from the stream):")
  for name, off in worst:
    print(f"             {name:>24s}  {np.degrees(off):+7.1f}°")
  return MocapCalibration.load(out_path, spec)


# ══════════════════════════════════════════════════════════════════════════════
# Source A — ChingMu VRPN (adapted from TTRL data_src.CMVrpnWrapper)
# ══════════════════════════════════════════════════════════════════════════════
class ChingMuMocapSource:
  """Reads retargeted robot root pose + per-joint angles from a ChingMu VRPN
  stream and pushes them into a `MocapState`.

  Assumes the ChingMu software streams an already-retargeted G1 skeleton where
  each joint is one sensor reporting its angle in `quat[0]` (same convention as
  the TTRL pipeline). Positions are millimetres; root quat is [qx,qy,qz,qw].
  """

  def __init__(
    self,
    state: MocapState,
    *,
    dll_path: str,
    host: str,
    sensor_root: int,
    sensor_joint_first: int,
    num_joints: int,
    pos_scale: float = 0.001,
    encoding: str = "gbk",
  ) -> None:
    from ctypes import (
      CDLL,
      CFUNCTYPE,
      POINTER,
      Structure,
      c_char_p,
      c_double,
      c_int,
      c_long,
    )

    class _Timeval(Structure):
      _fields_ = [("tv_sec", c_long), ("tv_usec", c_long)]

    class _VrpnTracker(Structure):
      _fields_ = [
        ("msg_time", _Timeval),
        ("sensor", c_int),
        ("frameCounter", c_int),
        ("pos", c_double * 3),
        ("quat", c_double * 4),  # qx, qy, qz, qw
      ]

    self._state = state
    self._host = bytes(host, encoding)
    self._sensor_root = int(sensor_root)
    self._sensor_joint_first = int(sensor_joint_first)
    self._num_joints = int(num_joints)
    self._scale = float(pos_scale)

    self._root_pos = np.zeros(3, dtype=np.float64)
    self._root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    self._joints = np.zeros(self._num_joints, dtype=np.float64)
    self._got_root = False

    self._dll = CDLL(dll_path)
    self._TrackerCb = CFUNCTYPE(None, c_char_p, POINTER(_VrpnTracker))
    self._cb = self._TrackerCb(self._on_tracker)
    self._running = False
    self._thread: Optional[threading.Thread] = None

  def _on_tracker(self, _ptr, b) -> None:
    d = b.contents
    if d.sensor == self._sensor_root:
      self._root_pos = np.array(
        [d.pos[0] * self._scale, d.pos[1] * self._scale, d.pos[2] * self._scale]
      )
      # ChingMu [qx,qy,qz,qw] -> MuJoCo [qw,qx,qy,qz]
      self._root_quat = np.array([d.quat[3], d.quat[0], d.quat[1], d.quat[2]])
      self._got_root = True
    idx = d.sensor - self._sensor_joint_first
    if 0 <= idx < self._num_joints:
      self._joints[idx] = d.quat[0]  # single-axis joint angle
    if self._got_root:
      self._state.update(self._root_pos, self._root_quat, self._joints)

  def start(self) -> None:
    self._dll.CMVrpnStartExtern()
    try:
      self._dll.CMVrpnEnableLog(False)
    except Exception:
      pass
    self._dll.CMPluginConnectServer(self._host)
    self._dll.CMPluginRegisterTrackerData(self._host, None, self._cb)
    self._running = True
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()
    print(f"[ChingMu] streaming from {self._host.decode(errors='replace')}")

  def _loop(self) -> None:
    while self._running:
      self._dll.CMPluginRegisterTrackerData(self._host, None, self._cb)
      time.sleep(0.001)

  def stop(self) -> None:
    self._running = False
    if self._thread:
      self._thread.join(timeout=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Source B — NPZ replay (device-free plumbing test)
# ══════════════════════════════════════════════════════════════════════════════
class NpzMockSource:
  """Streams a YAHMP `.npz` clip into a `MocapState` as if it were live, so the
  whole teleop path can be exercised without any mocap hardware."""

  def __init__(self, state: MocapState, spec: PolicySpec, clip_path: Path) -> None:
    from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import MotionClip

    self._state = state
    self._clip = MotionClip(clip_path, root_body_name=spec.root_body_name)
    self._running = False
    self._thread: Optional[threading.Thread] = None

  def start(self) -> None:
    self._running = True
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()
    print("[NpzMock] streaming clip as synthetic mocap")

  def _loop(self) -> None:
    t0 = time.perf_counter()
    while self._running:
      t = time.perf_counter() - t0
      f = self._clip.sample(t)
      # Publish positions only; LiveReference finite-differences velocities,
      # mirroring what a real device delivers.
      self._state.update(f.root_pos_w, f.root_quat_w, f.joint_pos)
      time.sleep(0.005)

  def stop(self) -> None:
    self._running = False
    if self._thread:
      self._thread.join(timeout=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Source C — Redis (GMR-retargeted robot pose, TWIST2-style two-process teleop)
# ══════════════════════════════════════════════════════════════════════════════
class RedisMocapSource:
  """Reads GMR-retargeted robot poses from redis (published by
  `retarget_server_gmr.py`) and pushes them into a `MocapState`, so the policy
  loop is byte-for-byte identical to any other source. This is the deploy half
  of the TWIST2-style two-process retargeting architecture:

      ChingMu human skeleton ─► retarget_server_gmr (GMR, `gmr` env) ─► redis
                                                                          │
      MocapState ◄── RedisMocapSource ◄─────────────────────────────────┘

  Redis payload (JSON string at `key`), robot-space, MuJoCo conventions:
      {"seq": int, "t": float,
       "root_pos":  [x, y, z],           # pelvis world pos (m), ground-offset by GMR
       "root_quat": [w, x, y, z],
       "joint_pos": {joint_name: rad, ...},
       "joint_vel": {joint_name: rad/s, ...} | null,
       "detected":  bool}

  Joints are resolved **by name** into `joint_names` order, so ordering is
  unambiguous and no `--joint-order` is needed (or should be passed) for
  `--source redis`. `client` is injectable for testing.
  """

  def __init__(
    self,
    state: MocapState,
    joint_names: list[str] | tuple[str, ...],
    *,
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    key: str = "yahmp:mocap:pose",
    poll_hz: float = 200.0,
    client=None,
  ) -> None:
    self._state = state
    self._names = list(joint_names)
    self._key = key
    self._poll_dt = 1.0 / max(poll_hz, 1.0)
    if client is None:
      import redis  # lazy: only required for --source redis

      client = redis.Redis(host=host, port=port, db=db)
    self._client = client
    self._running = False
    self._thread: Optional[threading.Thread] = None
    self._last_seq: Optional[int] = None

  def _read_once(self) -> bool:
    """Poll the key once; push into MocapState if a new detected frame is there."""
    raw = self._client.get(self._key)
    if raw is None:
      return False
    if isinstance(raw, (bytes, bytearray)):
      raw = raw.decode("utf-8")
    try:
      msg = json.loads(raw)
    except (ValueError, TypeError):
      return False  # skip a torn/partial write; next poll gets a clean frame
    if not msg.get("detected", True):
      return False
    seq = msg.get("seq")
    if seq is not None and seq == self._last_seq:
      return False  # no new frame since last poll
    self._last_seq = seq

    root_pos = np.asarray(msg["root_pos"], dtype=np.float64)
    root_quat = np.asarray(msg["root_quat"], dtype=np.float64)  # wxyz
    jp = msg["joint_pos"]
    try:
      joint_pos = np.asarray([jp[n] for n in self._names], dtype=np.float64)
    except KeyError as exc:
      raise KeyError(
        f"redis joint_pos is missing policy joint {exc}. The retarget server must "
        f"publish all {len(self._names)} joints in spec.joint_names."
      ) from exc
    jv = msg.get("joint_vel")
    joint_vel = (
      np.asarray([jv.get(n, 0.0) for n in self._names], dtype=np.float64) if jv else None
    )
    self._state.update(root_pos, root_quat, joint_pos, joint_vel=joint_vel)
    return True

  def start(self) -> None:
    try:
      self._client.ping()
    except Exception as exc:  # noqa: BLE001 - surface a clear message, not a stack
      raise SystemExit(
        f"Cannot reach redis at key {self._key!r}: {exc}. Start the retarget "
        "server (retarget_server_gmr.py) and a redis server first."
      ) from exc
    self._running = True
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()
    print(f"[Redis] polling {self._key!r} @ {1 / self._poll_dt:.0f} Hz for GMR poses")

  def _loop(self) -> None:
    while self._running:
      try:
        self._read_once()
      except KeyError:
        raise
      except Exception:  # noqa: BLE001 - transient redis hiccup; keep last pose
        pass
      time.sleep(self._poll_dt)

  def stop(self) -> None:
    self._running = False
    if self._thread:
      self._thread.join(timeout=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Kinematic full-body replay (no policy) — mocap → MuJoCo qpos, for validation
# ══════════════════════════════════════════════════════════════════════════════
def _run_kinematic_replay(
  model,
  data,
  spec: PolicySpec,
  reference: "LiveReference",
  viewer_cfg,
  joint_qpos_adr: np.ndarray,
  root_qpos_adr: int,
  source,
) -> None:
  """Write the live mocap pose straight into the model's qpos and render — no
  policy, no physics. Mirrors `replay_chingmu_sim.py --robot-src vrpn`.

  Use this to visually verify the joint-order map and frame calibration: the G1
  model should mimic the human limb-for-limb. Once correct, switch to
  `--mode teleop` to close the policy loop.
  """
  import mujoco.viewer as mujoco_viewer

  print(
    "[INFO] Kinematic replay (no policy). Verify the G1 mirrors the human "
    "limb-for-limb, then rerun with --mode teleop."
  )
  with mujoco_viewer.launch_passive(model, data) as viewer:
    _configure_camera(viewer, model, spec.root_body_name, viewer_cfg)
    try:
      while viewer.is_running():
        t0 = time.perf_counter()
        frame = reference.sample(0.0)
        data.qpos[root_qpos_adr : root_qpos_adr + 3] = frame.root_pos_w
        data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = frame.root_quat_w
        data.qpos[joint_qpos_adr] = frame.joint_pos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(max(0.0, 0.02 - (time.perf_counter() - t0)))  # ~50 Hz
    finally:
      source.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Interactive calibration (kinematic viewer + live terminal panel, no policy)
# ══════════════════════════════════════════════════════════════════════════════
class _InteractiveCalib:
  """Mutable calibration state edited live from the viewer's key callback.

  The MuJoCo window is driven kinematically by the *calibrated* pose, so you see
  the effect instantly; the terminal panel shows per-joint raw / offset /
  calibrated / default angles. `apply()` mirrors `MocapCalibration.apply` but can
  be toggled off to compare against the raw stream.
  """

  KEYS = "[N]capture  [C]toggle  [ [ / ] ]select  [ - / = ]nudge  [0]zero  [R]root  [S]save  [P]print"

  def __init__(self, spec: PolicySpec, height_target: float, seed: Optional["MocapCalibration"] = None):
    self.names = list(spec.joint_names)
    self.n = len(self.names)
    self.default = np.asarray(spec.default_joint_pos, dtype=np.float64)
    self.joint_offset = np.zeros(self.n, dtype=np.float64)
    self.root_quat_neutral = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    self.root_height_neutral = 0.0
    self.root_height_target = float(height_target)
    self.root_rpy_offset = np.zeros(3, dtype=np.float64)  # [roll, pitch, yaw] rad
    self.enabled = True
    self.sel = 0
    self.step = np.deg2rad(2.0)
    self.status = "ready — stand NEUTRAL and press N"
    if seed is not None:
      self.joint_offset = np.asarray(seed.joint_offset, dtype=np.float64).copy()
      self.root_quat_neutral = np.asarray(seed.root_quat_neutral, dtype=np.float64).copy()
      self.root_height_neutral = float(seed.root_height_neutral)
      self.root_height_target = float(seed.root_height_target)
      self.root_rpy_offset = np.asarray(getattr(seed, "root_rpy_offset", np.zeros(3)), dtype=np.float64).copy()
      self.status = "seeded from existing calibration"

  def apply(self, joint_pos, root_pos, root_quat):
    if not self.enabled:
      return joint_pos, root_pos, root_quat
    jp = joint_pos - self.joint_offset
    rp = np.asarray(root_pos, dtype=np.float64).copy()
    rp[2] = rp[2] - self.root_height_neutral + self.root_height_target
    rq = _quat_mul(_quat_conj(self.root_quat_neutral), _quat_normalize(root_quat))
    if np.any(self.root_rpy_offset):
      rq = _quat_mul(_quat_from_euler(*self.root_rpy_offset), rq)
    return jp, rp, _quat_normalize(rq)

  def capture_neutral(self, raw_joints, raw_root_pos, raw_root_quat):
    self.joint_offset = np.asarray(raw_joints, dtype=np.float64) - self.default
    self.root_quat_neutral = _quat_normalize(np.asarray(raw_root_quat, dtype=np.float64))
    self.root_height_neutral = float(raw_root_pos[2])
    self.root_rpy_offset = np.zeros(3, dtype=np.float64)
    self.status = "captured neutral → offsets set so this pose == robot default"

  def zero_root(self, raw_root_quat):
    self.root_quat_neutral = _quat_normalize(np.asarray(raw_root_quat, dtype=np.float64))
    self.status = "root orientation re-zeroed to current pose"

  def select(self, d):
    self.sel = (self.sel + int(d)) % self.n
    self.status = f"selected {self.names[self.sel]}"

  def nudge(self, d):
    self.joint_offset[self.sel] += float(d) * self.step
    self.status = f"{self.names[self.sel]} offset = {np.degrees(self.joint_offset[self.sel]):+.1f}°"

  def zero_selected(self):
    self.joint_offset[self.sel] = 0.0
    self.status = f"{self.names[self.sel]} offset zeroed"

  def to_dict(self):
    return {
      "_comment": (
        "Interactive neutral-pose mocap calibration (Z-up). calibrated = raw - "
        "joint_offset; root zeroed via conj(root_quat_neutral); height retargeted."
      ),
      "joint_names": list(self.names),
      "joint_offset_rad": [float(v) for v in self.joint_offset],
      "root_quat_neutral_wxyz": [float(v) for v in self.root_quat_neutral],
      "root_height_neutral": float(self.root_height_neutral),
      "root_height_target": float(self.root_height_target),
      "root_rpy_offset_rad": [float(v) for v in self.root_rpy_offset],
    }

  def render_panel(self, raw) -> str:
    raw = np.zeros(self.n) if raw is None else np.asarray(raw, dtype=np.float64)
    cal = raw - self.joint_offset
    hdr = [
      "YAHMP mocap calibration  —  kinematic viewer (no policy)",
      f"  calibration: {'ON ' if self.enabled else 'OFF'}   "
      f"height_target={self.root_height_target:.3f}m   neutral_z={self.root_height_neutral:.3f}m",
      f"  {self.KEYS}",
      f"  {'sel':>3} {'joint':>24} {'raw°':>8} {'off°':>8} {'cal°':>8} {'def°':>8}",
      "  " + "─" * 63,
    ]
    body = [
      f"  {'>>' if i == self.sel else '':>3} {name:>24} "
      f"{np.degrees(raw[i]):>8.1f} {np.degrees(self.joint_offset[i]):>8.1f} "
      f"{np.degrees(cal[i]):>8.1f} {np.degrees(self.default[i]):>8.1f}"
      for i, name in enumerate(self.names)
    ]
    return "\n".join(hdr + body + [f"  status: {self.status}"])


def _run_interactive_calibration(
  model,
  data,
  spec: PolicySpec,
  reference: "LiveReference",
  viewer_cfg,
  joint_qpos_adr: np.ndarray,
  root_qpos_adr: int,
  source,
  *,
  out_path: str,
  height_target: float,
  seed: Optional["MocapCalibration"],
) -> None:
  """Live kinematic calibration: adjust offsets in the viewer, save a JSON.

  `reference` must be **uncalibrated** — calibration is applied here so it can be
  toggled on/off for comparison. Nothing is energized; this only writes qpos.
  """
  import mujoco.viewer as mujoco_viewer

  calib = _InteractiveCalib(spec, height_target, seed=seed)
  latest: dict = {"frame": None, "raw": None}

  def save() -> None:
    p = Path(out_path)
    if p.parent and not p.parent.exists():
      p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(calib.to_dict(), indent=2))
    calib.status = f"saved → {out_path}"

  def key_callback(key: int) -> None:
    ch = chr(key) if 0 <= key < 256 else ""
    f = latest["frame"]
    if ch == "N" and f is not None:
      calib.capture_neutral(f.joint_pos, f.root_pos_w, f.root_quat_w)
    elif ch == "C":
      calib.enabled = not calib.enabled
      calib.status = f"calibration {'ON' if calib.enabled else 'OFF'}"
    elif ch == "]":
      calib.select(+1)
    elif ch == "[":
      calib.select(-1)
    elif ch in ("=", "+"):
      calib.nudge(+1)
    elif ch in ("-", "_"):
      calib.nudge(-1)
    elif ch == "0":
      calib.zero_selected()
    elif ch == "R" and f is not None:
      calib.zero_root(f.root_quat_w)
    elif ch == "S":
      save()
    elif ch == "P":
      print("\n" + json.dumps(calib.to_dict(), indent=2))

  print(
    "[Calibrate] interactive kinematic calibration. Stand NEUTRAL, press N, then "
    "move to check tracking. Keys:\n            " + _InteractiveCalib.KEYS
  )
  with mujoco_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    _configure_camera(viewer, model, spec.root_body_name, viewer_cfg)
    last_panel = 0.0
    try:
      while viewer.is_running():
        t0 = time.perf_counter()
        frame = reference.sample(0.0)  # raw (reference is uncalibrated in this mode)
        latest["frame"] = frame
        latest["raw"] = frame.joint_pos
        jp, rp, rq = calib.apply(frame.joint_pos, frame.root_pos_w, frame.root_quat_w)
        data.qpos[root_qpos_adr : root_qpos_adr + 3] = rp
        data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = rq
        data.qpos[joint_qpos_adr] = jp
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        viewer.sync()
        if t0 - last_panel > 0.2:  # refresh the terminal panel ~5 Hz
          sys.stdout.write("\x1b[2J\x1b[H" + calib.render_panel(latest["raw"]) + "\n")
          sys.stdout.flush()
          last_panel = t0
        time.sleep(max(0.0, 0.02 - (time.perf_counter() - t0)))  # ~50 Hz
    finally:
      source.stop()


def _run_gui_calibration(
  model,
  data,
  spec: PolicySpec,
  reference: "LiveReference",
  viewer_cfg,
  joint_qpos_adr: np.ndarray,
  root_qpos_adr: int,
  source,
  *,
  out_path: str,
  height_target: float,
  seed: Optional["MocapCalibration"],
) -> None:
  """Slider GUI (tkinter) to dial the root Roll/Pitch/Yaw + height, live in MuJoCo.

  A single-thread design: tkinter's mainloop drives a ~50 Hz `after` tick that
  samples the stream, applies the current calibration, and syncs the passive
  viewer. Sliders adjust the world-frame root R/P/Y correction; "Capture Neutral"
  sets the joint biases + zeroes the root so legs match the robot default.
  """
  import tkinter as tk

  import mujoco.viewer as mujoco_viewer

  calib = _InteractiveCalib(spec, height_target, seed=seed)
  latest: dict = {"frame": None}

  root = tk.Tk()
  root.title("YAHMP mocap calibration — root R/P/Y")
  roll_v = tk.DoubleVar(value=float(np.degrees(calib.root_rpy_offset[0])))
  pitch_v = tk.DoubleVar(value=float(np.degrees(calib.root_rpy_offset[1])))
  yaw_v = tk.DoubleVar(value=float(np.degrees(calib.root_rpy_offset[2])))
  height_v = tk.DoubleVar(value=float(calib.root_height_target))
  enabled_v = tk.BooleanVar(value=True)
  status_v = tk.StringVar(value=calib.status)
  readout_v = tk.StringVar(value="")

  def _slider(label, var, lo, hi, res):
    row = tk.Frame(root)
    row.pack(fill="x", padx=8, pady=1)
    tk.Label(row, text=label, width=9, anchor="w").pack(side="left")
    tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res, orient="horizontal", length=320).pack(
      side="left", fill="x", expand=True
    )

  _slider("Roll °", roll_v, -30.0, 30.0, 0.5)
  _slider("Pitch °", pitch_v, -30.0, 30.0, 0.5)
  _slider("Yaw °", yaw_v, -60.0, 60.0, 0.5)
  _slider("Height m", height_v, 0.50, 1.00, 0.005)

  def _sync_from_sliders():
    calib.root_rpy_offset = np.deg2rad([roll_v.get(), pitch_v.get(), yaw_v.get()])
    calib.root_height_target = float(height_v.get())
    calib.enabled = bool(enabled_v.get())

  def _capture():
    f = latest["frame"]
    if f is not None:
      calib.capture_neutral(f.joint_pos, f.root_pos_w, f.root_quat_w)
      roll_v.set(0.0)
      pitch_v.set(0.0)
      yaw_v.set(0.0)
      height_v.set(calib.root_height_target)
      status_v.set(calib.status)

  def _reset_rpy():
    roll_v.set(0.0)
    pitch_v.set(0.0)
    yaw_v.set(0.0)
    status_v.set("root R/P/Y reset to 0")

  def _save():
    _sync_from_sliders()
    p = Path(out_path)
    if p.parent and not p.parent.exists():
      p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(calib.to_dict(), indent=2))
    status_v.set(f"saved → {out_path}")

  btns = tk.Frame(root)
  btns.pack(fill="x", padx=8, pady=6)
  tk.Button(btns, text="Capture Neutral (joints+root)", command=_capture).pack(side="left")
  tk.Button(btns, text="Reset R/P/Y", command=_reset_rpy).pack(side="left", padx=4)
  tk.Button(btns, text="Save", command=_save).pack(side="left", padx=4)
  tk.Checkbutton(root, text="calibration enabled", variable=enabled_v).pack(anchor="w", padx=8)
  tk.Label(root, textvariable=readout_v, anchor="w", justify="left", font=("TkFixedFont", 10)).pack(
    fill="x", padx=8
  )
  tk.Label(root, textvariable=status_v, anchor="w", fg="#0a5").pack(fill="x", padx=8, pady=4)

  viewer = mujoco_viewer.launch_passive(model, data)
  _configure_camera(viewer, model, spec.root_body_name, viewer_cfg)
  print("[Calibrate] GUI open. Stand NEUTRAL → Capture Neutral, then dial Roll/Pitch/Yaw. Save when happy.")

  def _tick():
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
      f"command root:  roll={er:+6.1f}°  pitch={ep:+6.1f}°  yaw={ey:+6.1f}°   height={rp[2]:.3f} m"
    )
    root.after(20, _tick)

  def _on_close():
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
    source.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Target smoothing (mirrors run_yahmp_onnx_real.TargetSmoother) + shared step
# ══════════════════════════════════════════════════════════════════════════════
class TargetSmoother:
  """EMA + slew-rate limit on the per-joint position target — a copy of the one
  in `run_yahmp_onnx_real.py` (kept here to avoid a circular import).

  A YAHMP policy can emit a high-frequency action limit cycle that ideal MuJoCo
  actuators reproduce as visible limb shaking (while real hardware's latency /
  friction filters it out). This breaks that limit cycle by smoothing the
  *commanded* target only — the observation fed back to the policy is untouched,
  so sim/real observation parity is preserved. `set_params` allows live tuning
  from the GUI.

    * EMA         y = ema·target + (1 - ema)·y_prev   (ema=1.0 → off/passthrough)
    * rate limit  |y - y_prev| <= max_rate·control_dt  (max_rate=0 → off)
  """

  def __init__(self, *, init: np.ndarray, ema: float, max_rate: float, control_dt: float) -> None:
    self._prev = np.asarray(init, dtype=np.float64).copy()
    self._control_dt = float(control_dt)
    self.set_params(ema=ema, max_rate=max_rate)

  def set_params(self, *, ema: float, max_rate: float) -> None:
    if not 0.0 < ema <= 1.0:
      raise ValueError(f"--target-ema must be in (0, 1], got {ema}.")
    if max_rate < 0.0:
      raise ValueError(f"--target-max-rate must be >= 0, got {max_rate}.")
    self._ema = float(ema)
    self._max_step = float(max_rate) * self._control_dt if max_rate > 0.0 else 0.0

  @property
  def enabled(self) -> bool:
    return self._ema < 1.0 or self._max_step > 0.0

  def step(self, target: np.ndarray) -> np.ndarray:
    y = self._ema * np.asarray(target, dtype=np.float64) + (1.0 - self._ema) * self._prev
    if self._max_step > 0.0:
      y = self._prev + np.clip(y - self._prev, -self._max_step, self._max_step)
    self._prev = y
    return y


def _teleop_step(ctx: dict) -> tuple[MotionFrame, float]:
  """One teleop control step, shared by the headless loop and the tuning GUI.

  obs → ONNX → decode target (mirrors `_apply_action`) → `TargetSmoother` →
  `data.ctrl` → `mj_step` → optional record. `ctx` bundles the static handles and
  the mutable loop state (`previous_action`, `history`, `prev_loop_start`,
  `recorder`); it is mutated in place. Returns `(frame, wall_t0)`.
  """
  spec, model, data, reference = ctx["spec"], ctx["model"], ctx["data"], ctx["reference"]
  jqp, jqv, rqp = ctx["joint_qpos_adr"], ctx["joint_qvel_adr"], ctx["root_qpos_adr"]
  idx = ctx["action_target_joint_indices"]
  wall_t0 = time.perf_counter()

  frame = reference.sample(0.0)
  terms = _term_values(model, data, spec, reference, 0.0, frame, jqp, jqv, ctx["previous_action"])
  obs = _build_observation(spec, terms, ctx["history"])
  t_infer0 = time.perf_counter()
  raw_action = (
    ctx["session"].run([ctx["output_name"]], {ctx["input_name"]: obs[None, :].astype(np.float32)})[0]
    .reshape(-1)
    .astype(np.float32)
  )
  infer_ms = (time.perf_counter() - t_infer0) * 1e3

  # Capture the state the policy acted on (pre-step) for the recorder.
  rec = ctx["recorder"]
  if rec is not None:
    actual_joint_pos = np.asarray(data.qpos[jqp], dtype=np.float64).copy()
    actual_joint_vel = np.asarray(data.qvel[jqv], dtype=np.float64).copy()
    base_quat = np.asarray(data.qpos[rqp + 3 : rqp + 7], dtype=np.float64).copy()

  # Decode action → per-joint position target (mirrors `_apply_action`), then
  # smooth the commanded target only (obs above is untouched → sim/real parity).
  processed = raw_action.astype(np.float64) * spec.action_scale + spec.action_offset
  target = ctx["default_joint_pos"].copy()
  if spec.action_semantics == "residual_joint_position":
    target[idx] = frame.joint_pos[idx] + processed
  else:
    target[idx] = processed
  target = ctx["smoother"].step(target)
  data.ctrl[ctx["action_actuator_ids"]] = target[idx]

  for _ in range(ctx["steps_per_control"]):
    mujoco.mj_step(model, data)

  if rec is not None:
    step_ms = (time.perf_counter() - wall_t0) * 1e3
    rec.step(
      wall_time=wall_t0,
      ref_joint_pos=frame.joint_pos,
      target_joint_pos=target,
      actual_joint_pos=actual_joint_pos,
      actual_joint_vel=actual_joint_vel,
      quat=base_quat,
      ang_vel=terms["base_ang_vel"],
      proj_grav=terms["projected_gravity"],
      raw_action=raw_action,
      root_cmd=terms["command"][2 * ctx["n"] : 2 * ctx["n"] + 6],
      timing=dict(
        infer_ms=infer_ms,
        step_ms=step_ms,
        period_ms=(wall_t0 - ctx["prev_loop_start"]) * 1e3 if ctx["prev_loop_start"] is not None else float("nan"),
        overrun=step_ms > spec.control_dt * 1e3,
      ),
    )
    if ctx["record_steps"] > 0 and rec.count >= ctx["record_steps"]:
      print("[Recorder] reached --record-steps limit, stopping recording.")
      rec.close()
      ctx["recorder"] = None

  ctx["prev_loop_start"] = wall_t0
  ctx["previous_action"] = raw_action
  next_frame = reference.sample(0.0)
  next_terms = _term_values(model, data, spec, reference, 0.0, next_frame, jqp, jqv, ctx["previous_action"])
  _append_history(spec, ctx["history"], next_terms)
  return frame, wall_t0


# ══════════════════════════════════════════════════════════════════════════════
# Live tuning GUI (policy in the loop) — dial root R/P/Y + height + smoothing
# ══════════════════════════════════════════════════════════════════════════════
def _run_gui_teleop(
  ctx: dict,
  viewer_cfg,
  source,
  *,
  calibration: "MocapCalibration",
  out_path: str,
  target_ema: float,
  target_max_rate: float,
) -> None:
  """Slider GUI to tune root Roll/Pitch/Yaw + height AND target smoothing LIVE
  while the policy runs. Each ~50 Hz tick runs a full `_teleop_step`, so you watch
  the tuned params drive the actual policy-controlled robot — the exact setup you
  verify before sim2real. R/P/Y + height mutate the loaded `MocapCalibration`
  (LiveReference re-reads it every frame); EMA / max-rate mutate the
  `TargetSmoother`. Save writes the `--mocap-calibration` JSON and prints the
  matching `--target-ema` / `--target-max-rate` flags for the deploy command.
  """
  import tkinter as tk

  import mujoco.viewer as mujoco_viewer

  spec, model, data = ctx["spec"], ctx["model"], ctx["data"]
  smoother = ctx["smoother"]

  root = tk.Tk()
  root.title("YAHMP teleop — live tuning (root R/P/Y + smoothing)")
  roll_v = tk.DoubleVar(value=float(np.degrees(calibration.root_rpy_offset[0])))
  pitch_v = tk.DoubleVar(value=float(np.degrees(calibration.root_rpy_offset[1])))
  yaw_v = tk.DoubleVar(value=float(np.degrees(calibration.root_rpy_offset[2])))
  height_v = tk.DoubleVar(value=float(calibration.root_height_target))
  ema_v = tk.DoubleVar(value=float(target_ema))
  rate_v = tk.DoubleVar(value=float(target_max_rate))
  status_v = tk.StringVar(value="policy running — dial params, Save when stable")
  readout_v = tk.StringVar(value="")

  def _slider(label, var, lo, hi, res):
    row = tk.Frame(root)
    row.pack(fill="x", padx=8, pady=1)
    tk.Label(row, text=label, width=11, anchor="w").pack(side="left")
    tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res, orient="horizontal", length=320).pack(
      side="left", fill="x", expand=True
    )

  _slider("Roll °", roll_v, -30.0, 30.0, 0.5)
  _slider("Pitch °", pitch_v, -30.0, 30.0, 0.5)
  _slider("Yaw °", yaw_v, -60.0, 60.0, 0.5)
  _slider("Height m", height_v, 0.50, 1.00, 0.005)
  _slider("Target EMA", ema_v, 0.05, 1.0, 0.05)
  _slider("Max rate r/s", rate_v, 0.0, 20.0, 0.5)

  def _sync_from_sliders():
    calibration.root_rpy_offset = np.deg2rad([roll_v.get(), pitch_v.get(), yaw_v.get()])
    calibration.root_height_target = float(height_v.get())
    smoother.set_params(ema=float(ema_v.get()), max_rate=float(rate_v.get()))

  def _reset_rpy():
    roll_v.set(0.0)
    pitch_v.set(0.0)
    yaw_v.set(0.0)
    status_v.set("root R/P/Y reset to 0")

  def _save():
    _sync_from_sliders()
    out = {
      "_comment": "Neutral-pose calibration; root R/P/Y + height tuned live via the teleop GUI.",
      "joint_names": list(spec.joint_names),
      "joint_offset_rad": [float(v) for v in calibration.joint_offset],
      "root_quat_neutral_wxyz": [float(v) for v in calibration.root_quat_neutral],
      "root_height_neutral": float(calibration.root_height_neutral),
      "root_height_target": float(calibration.root_height_target),
      "root_rpy_offset_rad": [float(v) for v in calibration.root_rpy_offset],
    }
    p = Path(out_path)
    if p.parent and not p.parent.exists():
      p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    flags = f"--target-ema {ema_v.get():.2f} --target-max-rate {rate_v.get():.1f}"
    status_v.set(f"saved → {out_path}   |  deploy smoothing: {flags}")
    print(f"[Tune] saved calibration → {out_path}")
    print(f"[Tune] for sim2real, add to the deploy command:  {flags}")

  btns = tk.Frame(root)
  btns.pack(fill="x", padx=8, pady=6)
  tk.Button(btns, text="Save calibration", command=_save).pack(side="left")
  tk.Button(btns, text="Reset R/P/Y", command=_reset_rpy).pack(side="left", padx=4)
  tk.Label(root, textvariable=readout_v, anchor="w", justify="left", font=("TkFixedFont", 10)).pack(
    fill="x", padx=8
  )
  tk.Label(root, textvariable=status_v, anchor="w", fg="#0a5").pack(fill="x", padx=8, pady=4)

  viewer = mujoco_viewer.launch_passive(model, data)
  _configure_camera(viewer, model, spec.root_body_name, viewer_cfg)
  print("[Tune] GUI open. Policy is running — dial Roll/Pitch/Yaw/Height + smoothing, Save when happy.")

  def _tick():
    if not viewer.is_running():
      root.destroy()
      return
    _sync_from_sliders()
    frame, _ = _teleop_step(ctx)
    viewer.sync()
    er, ep, ey = np.degrees(_quat_roll_pitch_yaw(frame.root_quat_w))
    bq = data.qpos[ctx["root_qpos_adr"] + 3 : ctx["root_qpos_adr"] + 7]
    br, bp, by = np.degrees(_quat_roll_pitch_yaw(bq))
    readout_v.set(
      f"cmd root r/p/y = {er:+6.1f}/{ep:+6.1f}/{ey:+6.1f}°   "
      f"base r/p/y = {br:+6.1f}/{bp:+6.1f}/{by:+6.1f}°   h={frame.root_pos_w[2]:.3f} m"
    )
    root.after(int(spec.control_dt * 1000), _tick)

  def _on_close():
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
    if ctx["recorder"] is not None and ctx["recorder"].count > 0:
      ctx["recorder"].close()


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
def run(
  *,
  onnx_path: Path,
  task_id: str,
  mode: str,
  source_kind: str,
  ort_provider: str,
  npz_clip: Optional[Path],
  chingmu_kwargs: dict,
  joint_order: Optional[list[str]],
  vel_smoothing: float,
  mocap_calibration: str = "",
  calibrate_height_target: float = 0.793,
  track_legs_only: bool = False,
  track_gain: float = 1.0,
  redis_kwargs: Optional[dict] = None,
  record: str = "",
  record_steps: int = 0,
  target_ema: float = 1.0,
  target_max_rate: float = 0.0,
  tune: bool = False,
) -> None:
  redis_kwargs = redis_kwargs or {}
  spec = PolicySpec.from_onnx(onnx_path)
  spec.validate()

  model, viewer_cfg, scene_physics_dt, scene_control_dt = _build_task_scene(task_id)
  if not np.isclose(scene_physics_dt, spec.physics_dt):
    raise ValueError("Task scene physics_dt does not match exported policy.")
  if not np.isclose(scene_control_dt, spec.control_dt):
    raise ValueError("Task scene control_dt does not match exported policy.")

  data = mujoco.MjData(model)
  joint_qpos_adr, joint_qvel_adr = _joint_addresses(model, spec.joint_names)
  action_actuator_ids = _actuator_ids(model, spec.action_target_names)
  action_target_joint_indices = np.asarray(
    [spec.joint_names.index(n) for n in spec.action_target_names], dtype=np.int32
  )
  root_qpos_adr, root_qvel_adr = _root_addresses(model, spec.root_body_name)

  # ── Start the chosen mocap source ─────────────────────────────────────────
  state = MocapState(num_joints=len(spec.joint_names) if joint_order is None else len(joint_order))
  if source_kind == "npz":
    if npz_clip is None:
      raise ValueError("--source npz requires --npz-clip.")
    source = NpzMockSource(state, spec, npz_clip)
  elif source_kind == "chingmu":
    source = ChingMuMocapSource(state, **chingmu_kwargs)
  elif source_kind == "redis":
    source = RedisMocapSource(state, spec.joint_names, **redis_kwargs)
  else:
    raise ValueError(f"Unknown --source {source_kind!r}.")
  source.start()

  seed_exists = bool(mocap_calibration) and Path(mocap_calibration).is_file()
  calibration = MocapCalibration.load(mocap_calibration, spec) if seed_exists else None
  if calibration is not None:
    print(f"[INFO] mocap calibration loaded from {mocap_calibration}")
  elif mocap_calibration and mode != "calibrate":
    raise SystemExit(f"--mocap-calibration file not found: {mocap_calibration}")
  # In calibrate mode the tool applies calibration itself (toggleable), so the
  # reference must stay raw; replay/teleop bake it into the reference.
  reference = LiveReference(
    state,
    spec,
    joint_order,
    vel_smoothing=vel_smoothing,
    calibration=(None if mode == "calibrate" else calibration),
    track_legs_only=track_legs_only,
    track_gain=track_gain,
  )
  if track_legs_only:
    print("[INFO] track-legs-only: waist + arms held at default; tracking legs only.")
  if track_gain != 1.0:
    print(f"[INFO] track-gain={track_gain:.2f}: command blended toward default (reduced tracking).")
  reference.wait_for_detection(timeout_s=15.0)

  if record and mode != "teleop":
    print(f"[Recorder] --record is ignored in --mode {mode} (only --mode teleop logs a run).")

  # ── Interactive calibration: tune root R/P/Y live in the viewer, save JSON ──
  if mode == "calibrate":
    if not mocap_calibration:
      raise SystemExit("--mode calibrate requires --mocap-calibration <path to save/load>.")
    calib_args = (model, data, spec, reference, viewer_cfg, joint_qpos_adr, root_qpos_adr, source)
    calib_kwargs = dict(
      out_path=mocap_calibration, height_target=calibrate_height_target, seed=calibration
    )
    have_gui = True
    try:  # probe both the import and a usable display before committing
      import tkinter as _tk

      _probe = _tk.Tk()
      _probe.withdraw()
      _probe.destroy()
    except Exception as exc:  # noqa: BLE001 - ImportError or TclError (no $DISPLAY)
      have_gui = False
      print(f"[Calibrate] slider GUI unavailable ({exc}); using keyboard/terminal mode.")
    if have_gui:
      _run_gui_calibration(*calib_args, **calib_kwargs)
    else:
      _run_interactive_calibration(*calib_args, **calib_kwargs)
    return

  # ── Kinematic replay: write the mocap pose straight to qpos (no policy) ─────
  if mode == "replay":
    _run_kinematic_replay(
      model, data, spec, reference, viewer_cfg, joint_qpos_adr, root_qpos_adr, source
    )
    return

  # ── Policy teleoperation: bring up the ONNX session ────────────────────────
  session, providers = _create_onnx_session(onnx_path, ort_provider)
  input_name = session.get_inputs()[0].name
  output_name = session.get_outputs()[0].name

  # ── Initialise the simulated robot at the first live pose ─────────────────
  initial_frame = reference.sample(0.0)
  _initialize_state(
    data,
    initial_frame,
    spec,
    joint_qpos_adr,
    joint_qvel_adr,
    root_qpos_adr,
    root_qvel_adr,
    init_default_joints=False,
  )
  mujoco.mj_forward(model, data)

  n = len(spec.joint_names)
  default_joint_pos = np.asarray(spec.default_joint_pos, dtype=np.float64)

  # ── Target smoother (breaks the sim2sim high-freq action limit cycle) ──────
  smoother = TargetSmoother(
    init=default_joint_pos, ema=target_ema, max_rate=target_max_rate, control_dt=spec.control_dt
  )
  if smoother.enabled:
    print(
      f"[INFO] target smoothing ON: ema={target_ema:.2f} max_rate={target_max_rate:.2f} rad/s "
      f"({target_max_rate * spec.control_dt * 180.0 / np.pi:.1f}°/step)"
    )
  else:
    print("[INFO] target smoothing OFF (raw policy target sent to actuators).")

  # ── Sim2sim data recorder (optional; --record) ─────────────────────────────
  # Same CSV/meta format as run_yahmp_onnx_real.py --record, so the identical
  # `run_yahmp_onnx_recorder analyze` / `replay_yahmp_onnx_sim` tools apply and a
  # sim run can be compared column-for-column against a real one (e.g. to see why
  # MuJoCo shakes while the robot does not).
  recorder = (
    Sim2RealRecorder(
      record,
      spec.joint_names,
      action_target_names=spec.action_target_names,
      meta=dict(
        sim2sim=True,
        onnx_path=str(onnx_path),
        task_id=task_id,
        source=source_kind,
        mode=mode,
        vel_smoothing=vel_smoothing,
        mocap_calibration=mocap_calibration or None,
        calibrated=bool(calibration is not None),
        target_ema=target_ema,
        target_max_rate=target_max_rate,
        control_dt=spec.control_dt,
        action_semantics=spec.action_semantics,
      ),
    )
    if record
    else None
  )

  previous_action = np.zeros(spec.action_dim, dtype=np.float32)
  terms = _term_values(
    model, data, spec, reference, 0.0, initial_frame,
    joint_qpos_adr, joint_qvel_adr, previous_action,
  )
  history = _initialize_history(spec, terms)

  # Shared loop context (static handles + mutable state) so the per-step logic
  # lives in exactly one place (`_teleop_step`), used by both the headless loop
  # and the tuning GUI.
  ctx = dict(
    model=model, data=data, spec=spec, reference=reference,
    session=session, input_name=input_name, output_name=output_name,
    joint_qpos_adr=joint_qpos_adr, joint_qvel_adr=joint_qvel_adr, root_qpos_adr=root_qpos_adr,
    action_actuator_ids=action_actuator_ids,
    action_target_joint_indices=action_target_joint_indices,
    steps_per_control=int(round(spec.control_dt / spec.physics_dt)),
    default_joint_pos=default_joint_pos, n=n,
    smoother=smoother, recorder=recorder, record_steps=record_steps,
    previous_action=previous_action, history=history, prev_loop_start=None,
  )

  print(f"[INFO] ONNX providers: {providers}")
  print(f"[INFO] Teleop source: {source_kind}  control_dt={spec.control_dt:.4f}s")

  import mujoco.viewer as mujoco_viewer

  # ── Live tuning GUI (policy in the loop): root R/P/Y + height + smoothing ───
  if tune:
    if calibration is None:
      raise SystemExit(
        "--tune adjusts an existing calibration; pass --mocap-calibration <file> "
        "(build one first with --mode calibrate)."
      )
    have_gui = True
    try:  # probe tkinter + a usable display before committing
      import tkinter as _tk

      _probe = _tk.Tk()
      _probe.withdraw()
      _probe.destroy()
    except Exception as exc:  # noqa: BLE001 - ImportError or TclError (no $DISPLAY)
      have_gui = False
      print(f"[Tune] GUI unavailable ({exc}); falling back to headless teleop.")
    if have_gui:
      try:
        _run_gui_teleop(
          ctx, viewer_cfg, source,
          calibration=calibration, out_path=mocap_calibration,
          target_ema=target_ema, target_max_rate=target_max_rate,
        )
      finally:
        source.stop()
      return

  with mujoco_viewer.launch_passive(model, data) as viewer:
    _configure_camera(viewer, model, spec.root_body_name, viewer_cfg)
    try:
      while viewer.is_running():
        _, wall_t0 = _teleop_step(ctx)
        viewer.sync()
        time.sleep(max(0.0, spec.control_dt - (time.perf_counter() - wall_t0)))
    finally:
      source.stop()
      if ctx["recorder"] is not None and ctx["recorder"].count > 0:
        ctx["recorder"].close()


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Real-time mocap teleoperation of a YAHMP ONNX policy (sim2sim).")
  p.add_argument("--onnx-path", type=Path, required=True)
  p.add_argument("--task-id", type=str, required=True)
  p.add_argument("--source", choices=("npz", "chingmu", "redis"), default="npz")
  p.add_argument(
    "--mode",
    choices=("teleop", "replay", "calibrate"),
    default="teleop",
    help="teleop = policy in the loop; replay = kinematic full-body remap to qpos (no policy); "
    "calibrate = interactive kinematic tool to build a --mocap-calibration JSON.",
  )
  p.add_argument("--ort-provider", choices=("auto", "cpu", "cuda"), default="auto")
  p.add_argument("--vel-smoothing", type=float, default=0.0, help="EMA factor [0,1) for finite-diff velocities.")
  p.add_argument("--mocap-calibration", type=str, default="", help="Neutral-pose calibration JSON to apply (teleop/replay) or build/save (--mode calibrate).")
  p.add_argument("--calibrate-height-target", type=float, default=0.793, help="Nominal pelvis height (m) neutral maps to in --mode calibrate. G1 default ~0.793.")
  p.add_argument("--track-legs-only", action="store_true", help="Hold waist + arms at the robot default and track only the legs from mocap (isolates an off-default upper-body command as a balance destabilizer).")
  p.add_argument("--track-gain", type=float, default=1.0, help="Blend the command toward the robot default: ref = default + gain*(mocap-default), gain in [0,1]. 1=full tracking, 0=hold default. Lower it to trade imitation for stability.")
  # NPZ mock
  p.add_argument("--npz-clip", type=Path, default=None)
  # ChingMu
  p.add_argument(
    "--chingmu-dll",
    type=str,
    default=None,
    help="Path to libCMVrpn.so (default: vendored third_party/chingmu/libCMVrpn.so).",
  )
  p.add_argument("--chingmu-host", type=str, default=None, help="e.g. MCAvatar@192.168.123.112")
  p.add_argument("--sensor-root", type=int, default=0)
  p.add_argument("--sensor-joint-first", type=int, default=1)
  p.add_argument("--num-joints", type=int, default=None, help="Streamed joint count (defaults to len(joint-order)).")
  p.add_argument("--pos-scale", type=float, default=0.001)
  p.add_argument(
    "--joint-order",
    type=Path,
    default=None,
    help="JSON list of ChingMu joint names in stream order; mapped to the policy's joint_names.",
  )
  # Redis (GMR-retargeted robot pose from retarget_server_gmr.py)
  p.add_argument("--redis-host", type=str, default="localhost")
  p.add_argument("--redis-port", type=int, default=6379)
  p.add_argument("--redis-db", type=int, default=0)
  p.add_argument("--redis-key", type=str, default="yahmp:mocap:pose")
  p.add_argument("--redis-poll-hz", type=float, default=200.0)
  # Recording (teleop only) — same format as run_yahmp_onnx_real.py --record
  p.add_argument(
    "--record",
    type=str,
    default="",
    help="Sim2sim CSV recording prefix, empty = off. e.g. --record ./sim2real_data/sim. "
    "Analyze with `run_yahmp_onnx_recorder analyze` or replay with replay_yahmp_onnx_sim.",
  )
  p.add_argument("--record-steps", type=int, default=0, help="Max control steps to record (0 = unlimited).")
  # Target smoothing (teleop) — same knobs as run_yahmp_onnx_real.py; breaks the
  # high-frequency action limit cycle that makes the sim2sim robot shake.
  p.add_argument("--target-ema", type=float, default=1.0, help="EMA on the commanded target: weight of the new sample each step, (0,1]. 1.0=off. Try 0.3-0.5 to suppress action chatter.")
  p.add_argument("--target-max-rate", type=float, default=0.0, help="Per-joint slew-rate cap on the target in rad/s. 0=off. e.g. 6.0 -> ~6.9 deg per 20ms step.")
  p.add_argument("--tune", action="store_true", help="Teleop only: open a slider GUI to dial root roll/pitch/yaw + height + target smoothing LIVE while the policy runs, and Save to --mocap-calibration. Needs a display + tkinter.")
  return p


def main() -> None:
  args = _build_argparser().parse_args()

  joint_order = None
  if args.joint_order is not None:
    joint_order = json.loads(Path(args.joint_order).read_text())
    if not isinstance(joint_order, list):
      raise ValueError("--joint-order JSON must be a list of joint names.")

  chingmu_kwargs = {}
  if args.source == "chingmu":
    if not args.chingmu_host:
      raise SystemExit("--source chingmu requires --chingmu-host.")
    dll_path = (
      Path(args.chingmu_dll).expanduser() if args.chingmu_dll else _VENDORED_CHINGMU_DLL
    )
    if not dll_path.is_file():
      raise SystemExit(
        f"ChingMu DLL not found: {dll_path}. Pass --chingmu-dll or place it in "
        "third_party/chingmu/libCMVrpn.so."
      )
    num_joints = args.num_joints or (len(joint_order) if joint_order else None)
    if num_joints is None:
      raise SystemExit("Provide --num-joints or --joint-order for chingmu source.")
    chingmu_kwargs = dict(
      dll_path=str(dll_path),
      host=args.chingmu_host,
      sensor_root=args.sensor_root,
      sensor_joint_first=args.sensor_joint_first,
      num_joints=num_joints,
      pos_scale=args.pos_scale,
    )

  redis_kwargs = {}
  if args.source == "redis":
    if joint_order is not None:
      raise SystemExit("--source redis resolves joints by name; do not pass --joint-order.")
    redis_kwargs = dict(
      host=args.redis_host,
      port=args.redis_port,
      db=args.redis_db,
      key=args.redis_key,
      poll_hz=args.redis_poll_hz,
    )

  run(
    onnx_path=args.onnx_path.expanduser().resolve(),
    task_id=str(args.task_id),
    mode=str(args.mode),
    source_kind=str(args.source),
    ort_provider=str(args.ort_provider),
    npz_clip=args.npz_clip.expanduser().resolve() if args.npz_clip else None,
    chingmu_kwargs=chingmu_kwargs,
    joint_order=joint_order,
    vel_smoothing=float(args.vel_smoothing),
    mocap_calibration=str(args.mocap_calibration),
    calibrate_height_target=float(args.calibrate_height_target),
    track_legs_only=bool(args.track_legs_only),
    track_gain=float(args.track_gain),
    redis_kwargs=redis_kwargs,
    record=str(args.record),
    record_steps=int(args.record_steps),
    target_ema=float(args.target_ema),
    target_max_rate=float(args.target_max_rate),
    tune=bool(args.tune),
  )


if __name__ == "__main__":
  main()
