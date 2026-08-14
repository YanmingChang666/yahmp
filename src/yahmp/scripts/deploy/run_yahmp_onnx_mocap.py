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
import threading
import time
from dataclasses import dataclass
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
  _apply_action,
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
  _root_addresses,
  _term_values,
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
    * root tilt    calibrated_quat = conj(root_quat_neutral) ⊗ raw_quat
                   → neutral pelvis orientation becomes identity, so the command's
                   root_roll / root_pitch ≈ 0 when the operator stands straight
                   (also re-aligns neutral facing to robot +X).
    * root height  calibrated_z = raw_z − root_height_neutral + root_height_target

  Assumes a **Z-up** mocap world (ChingMu default): the neutral inverse alone
  aligns the operator to the robot frame, with no extra axis swap.
  """

  joint_offset: np.ndarray        # (N,) policy joint order, radians
  root_quat_neutral: np.ndarray   # (4,) wxyz
  root_height_neutral: float
  root_height_target: float

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
    )

  def apply(
    self, joint_pos: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jp = joint_pos - self.joint_offset
    rp = root_pos.copy()
    rp[2] = rp[2] - self.root_height_neutral + self.root_height_target
    rq = _quat_normalize(_quat_mul(_quat_conj(self.root_quat_neutral), _quat_normalize(root_quat)))
    return jp, rp, rq


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
  ) -> None:
    self._state = state
    self._spec = spec
    self._alpha = float(np.clip(vel_smoothing, 0.0, 0.99))
    self._calib = calibration

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
) -> None:
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
  else:
    raise ValueError(f"Unknown --source {source_kind!r}.")
  source.start()

  calibration = MocapCalibration.load(mocap_calibration, spec) if mocap_calibration else None
  if calibration is not None:
    print(f"[INFO] mocap calibration loaded from {mocap_calibration}")
  reference = LiveReference(
    state, spec, joint_order, vel_smoothing=vel_smoothing, calibration=calibration
  )
  reference.wait_for_detection(timeout_s=15.0)

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

  previous_action = np.zeros(spec.action_dim, dtype=np.float32)
  terms = _term_values(
    model, data, spec, reference, 0.0, initial_frame,
    joint_qpos_adr, joint_qvel_adr, previous_action,
  )
  history = _initialize_history(spec, terms)
  steps_per_control = int(round(spec.control_dt / spec.physics_dt))

  print(f"[INFO] ONNX providers: {providers}")
  print(f"[INFO] Teleop source: {source_kind}  control_dt={spec.control_dt:.4f}s")

  import mujoco.viewer as mujoco_viewer

  with mujoco_viewer.launch_passive(model, data) as viewer:
    _configure_camera(viewer, model, spec.root_body_name, viewer_cfg)
    try:
      while viewer.is_running():
        wall_t0 = time.perf_counter()

        # OBSERVATION DIMS (g1_yahmp, num_joints N=29) — total obs = 1727:
        #   command 64 (=2N+6) + base_ang_vel 3 + projected_gravity 3 +
        #   joint_pos N + joint_vel N + actions N(=action_dim) + history 1570
        #   history 1570 = 10 frames × 157-dim block
        #     (block = command 64 + base_ang_vel 3 + proj_gravity 3 + joint_pos N + joint_vel N + actions N)
        #   command 64 = joint_pos_ref N | joint_vel_ref N | anchor_lin_vel_xy 2 |
        #                anchor_ang_vel_yaw 1 | root_height 1 | root_roll,pitch 2
        frame = reference.sample(0.0)  # latest mocap pose -> MotionFrame (see LiveReference.sample)
        terms = _term_values(  # dict of the 6 current terms above (robot state from sim + command from mocap)
          model, data, spec, reference, 0.0, frame,
          joint_qpos_adr, joint_qvel_adr, previous_action,
        )
        obs = _build_observation(spec, terms, history)  # (1727,) — concatenated in spec.observation_terms order
        raw_action = (
          session.run([output_name], {input_name: obs[None, :].astype(np.float32)})[0]
          .reshape(-1)
          .astype(np.float32)  # (29,) = action_dim
        )

        _apply_action(
          data, spec, raw_action, frame, action_actuator_ids, action_target_joint_indices
        )
        for _ in range(steps_per_control):
          mujoco.mj_step(model, data)

        previous_action = raw_action
        next_frame = reference.sample(0.0)
        next_terms = _term_values(
          model, data, spec, reference, 0.0, next_frame,
          joint_qpos_adr, joint_qvel_adr, previous_action,
        )
        _append_history(spec, history, next_terms)

        viewer.sync()
        time.sleep(max(0.0, spec.control_dt - (time.perf_counter() - wall_t0)))
    finally:
      source.stop()


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Real-time mocap teleoperation of a YAHMP ONNX policy (sim2sim).")
  p.add_argument("--onnx-path", type=Path, required=True)
  p.add_argument("--task-id", type=str, required=True)
  p.add_argument("--source", choices=("npz", "chingmu"), default="npz")
  p.add_argument(
    "--mode",
    choices=("teleop", "replay"),
    default="teleop",
    help="teleop = policy in the loop; replay = kinematic full-body remap to qpos (no policy).",
  )
  p.add_argument("--ort-provider", choices=("auto", "cpu", "cuda"), default="auto")
  p.add_argument("--vel-smoothing", type=float, default=0.0, help="EMA factor [0,1) for finite-diff velocities.")
  p.add_argument("--mocap-calibration", type=str, default="", help="Path to a neutral-pose calibration JSON (captured on the real robot via --calibrate) to apply to the live command.")
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
  )


if __name__ == "__main__":
  main()
