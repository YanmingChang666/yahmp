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

Live ChingMu teleoperation
    uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
      --task-id Mjlab-YAHMP-Unitree-G1 \
      --onnx-path assets/models/g1_yahmp.onnx \
      --source chingmu \
      --chingmu-dll /path/to/libCMVrpn.so \
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
  ) -> None:
    self._state = state
    self._spec = spec
    self._alpha = float(np.clip(vel_smoothing, 0.0, 0.99))

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
    return MotionFrame(
      joint_pos=joint_pos.astype(np.float64),
      joint_vel=np.asarray(joint_vel, dtype=np.float64),
      root_pos_w=root_pos.astype(np.float64),
      root_quat_w=root_quat.astype(np.float64),
      root_lin_vel_w=np.asarray(root_lin_vel, dtype=np.float64),
      root_ang_vel_w=np.asarray(root_ang_vel, dtype=np.float64),
    )

  # NOTE: the Future variant (`Mjlab-YAHMP-Future-Unitree-G1`) asks for frames at
  # `time + offset*control_dt`, which do not exist in a live stream. `_command_value`
  # only calls `.sample()`, so future steps here collapse to the current pose — an
  # approximation. Prefer the base `Mjlab-YAHMP-Unitree-G1` policy for teleop.


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
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
def run(
  *,
  onnx_path: Path,
  task_id: str,
  source_kind: str,
  ort_provider: str,
  npz_clip: Optional[Path],
  chingmu_kwargs: dict,
  joint_order: Optional[list[str]],
  vel_smoothing: float,
) -> None:
  spec = PolicySpec.from_onnx(onnx_path)
  spec.validate()
  session, providers = _create_onnx_session(onnx_path, ort_provider)
  input_name = session.get_inputs()[0].name
  output_name = session.get_outputs()[0].name

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

  reference = LiveReference(state, spec, joint_order, vel_smoothing=vel_smoothing)
  reference.wait_for_detection(timeout_s=15.0)

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

        frame = reference.sample(0.0)  # latest mocap pose
        terms = _term_values(
          model, data, spec, reference, 0.0, frame,
          joint_qpos_adr, joint_qvel_adr, previous_action,
        )
        obs = _build_observation(spec, terms, history)
        raw_action = (
          session.run([output_name], {input_name: obs[None, :].astype(np.float32)})[0]
          .reshape(-1)
          .astype(np.float32)
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
  p.add_argument("--ort-provider", choices=("auto", "cpu", "cuda"), default="auto")
  p.add_argument("--vel-smoothing", type=float, default=0.0, help="EMA factor [0,1) for finite-diff velocities.")
  # NPZ mock
  p.add_argument("--npz-clip", type=Path, default=None)
  # ChingMu
  p.add_argument("--chingmu-dll", type=str, default=None, help="Path to libCMVrpn.so.")
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
    if not args.chingmu_dll or not args.chingmu_host:
      raise SystemExit("--source chingmu requires --chingmu-dll and --chingmu-host.")
    num_joints = args.num_joints or (len(joint_order) if joint_order else None)
    if num_joints is None:
      raise SystemExit("Provide --num-joints or --joint-order for chingmu source.")
    chingmu_kwargs = dict(
      dll_path=args.chingmu_dll,
      host=args.chingmu_host,
      sensor_root=args.sensor_root,
      sensor_joint_first=args.sensor_joint_first,
      num_joints=num_joints,
      pos_scale=args.pos_scale,
    )

  run(
    onnx_path=args.onnx_path.expanduser().resolve(),
    task_id=str(args.task_id),
    source_kind=str(args.source),
    ort_provider=str(args.ort_provider),
    npz_clip=args.npz_clip.expanduser().resolve() if args.npz_clip else None,
    chingmu_kwargs=chingmu_kwargs,
    joint_order=joint_order,
    vel_smoothing=float(args.vel_smoothing),
  )


if __name__ == "__main__":
  main()
