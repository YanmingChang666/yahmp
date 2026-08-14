"""Sim2real deployment of a YAHMP ONNX policy on a physical Unitree G1, driven
by a live ChingMu mocap command.

┌────────────────────────────────────────────────────────────────────────────┐
│  ⚠  PHYSICAL ROBOT. READ DEPLOYMENT.md §6 (SAFETY) FIRST.                     │
│  First runs on a HOIST, feet off the ground, with a tested E-STOP.           │
│  Start with SMALL gains (--kp-scale 0.25 --kd-scale 0.5) and --dry-run.      │
│  This script has NOT been hardware-tested by its author — validate on-robot. │
└────────────────────────────────────────────────────────────────────────────┘

Design: this mirrors the structure of the proven Beyondmimic
`deploy_real/deploy_real4bydmimic.py` (Unitree `unitree_sdk2py` DDS, hg LowCmd/
LowState, CRC, zero-torque → move-to-default → run → damping), but:

  * The observation and action are built with YAHMP's OWN pipeline (imported
    from `run_yahmp_onnx_mujoco`), guaranteeing sim/real parity.
  * PD gains, default pose, joint order and action scaling come from the ONNX
    metadata — the exact values the policy trained with.
  * The motion **command** is streamed live from ChingMu (reused from
    `run_yahmp_onnx_mocap`), so the robot tracks the human in real time.
  * `--kp-scale` / `--kd-scale` let you start with small gains to prevent
    excessive motion, then ramp up as you gain confidence.

The G1 motor index order equals YAHMP's `joint_names` order (legs 0-11, waist
12-14, arms 15-28), so motor i ↔ joint_names[i] with no remap. Override with
`--joint2motor` if your robot differs.

Prerequisite (not a YAHMP dependency — install separately):
    uv pip install unitree_sdk2py     # or: pip install unitree_sdk2py

Example (small-gain dry run first, then live):
    uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_real \
      --onnx-path assets/models/g1_yahmp.onnx --net enp4s0 \
      --source chingmu --chingmu-host MCAvatar@192.168.123.112:3884 \
      --sensor-root 301 --sensor-joint-first 302 \
      --joint-order config/chingmu_joint_order.json \
      --kp-scale 0.25 --kd-scale 0.5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

from yahmp.scripts.deploy.run_yahmp_onnx_mocap import (
  ChingMuMocapSource,
  LiveReference,
  MocapCalibration,
  MocapState,
  NpzMockSource,
  capture_calibration,
)
from yahmp.scripts.deploy.run_yahmp_onnx_recorder import Sim2RealRecorder
from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import (
  PolicySpec,
  _append_history,
  _build_observation,
  _command_value,
  _create_onnx_session,
  _csv,
  _initialize_history,
  _load_onnx_metadata,
  _quat_rotate_inverse,
)

_GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Vendored Unitree remote-controller parser (from Beyondmimic common/)
# ══════════════════════════════════════════════════════════════════════════════
class KeyMap:
  R1, L1, start, select, R2, L2, F1, F2 = 0, 1, 2, 3, 4, 5, 6, 7
  A, B, X, Y, up, right, down, left = 8, 9, 10, 11, 12, 13, 14, 15


class RemoteController:
  def __init__(self) -> None:
    self.button = [0] * 16

  def set(self, data) -> None:
    keys = struct.unpack("H", data[2:4])[0]
    for i in range(16):
      self.button[i] = (keys & (1 << i)) >> i


def _parse_gains(onnx_path: Path, num_joints: int) -> tuple[np.ndarray, np.ndarray]:
  """Read the trained PD gains (Kp/Kd) from the ONNX metadata."""
  md = _load_onnx_metadata(onnx_path)
  if "joint_stiffness" not in md or "joint_damping" not in md:
    raise KeyError(
      "ONNX is missing joint_stiffness/joint_damping. Re-export with the current "
      "exporter (DEPLOYMENT.md §1) — older files predate these keys."
    )
  kp = np.asarray(_csv(md["joint_stiffness"], float), dtype=np.float64)
  kd = np.asarray(_csv(md["joint_damping"], float), dtype=np.float64)
  if kp.shape != (num_joints,) or kd.shape != (num_joints,):
    raise ValueError(
      f"Gain vectors have wrong length: kp={kp.shape}, kd={kd.shape}, "
      f"expected ({num_joints},)."
    )
  return kp, kd


# ══════════════════════════════════════════════════════════════════════════════
# Unitree G1 controller (hg LowCmd/LowState over DDS)
# ══════════════════════════════════════════════════════════════════════════════
class G1Controller:
  def __init__(
    self,
    *,
    net_iface: str,
    control_dt: float,
    num_joints: int,
    dof2motor: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    lowcmd_topic: str,
    lowstate_topic: str,
    dry_run: bool,
  ) -> None:
    # Lazy import so the module loads without the SDK (e.g. for syntax checks).
    from unitree_sdk2py.core.channel import (
      ChannelFactoryInitialize,
      ChannelPublisher,
      ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import (
      unitree_hg_msg_dds__LowCmd_,
      unitree_hg_msg_dds__LowState_,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
    from unitree_sdk2py.utils.crc import CRC

    self._CRC = CRC()
    self.control_dt = float(control_dt)
    self.n = int(num_joints)
    self.dof2motor = dof2motor
    self.kp = kp
    self.kd = kd
    self.dry_run = bool(dry_run)
    self.remote = RemoteController()

    ChannelFactoryInitialize(0, net_iface)
    self.low_cmd = unitree_hg_msg_dds__LowCmd_()
    self.low_state = unitree_hg_msg_dds__LowState_()
    self.mode_machine = 0

    self._pub = ChannelPublisher(lowcmd_topic, LowCmdHG)
    self._pub.Init()
    self._sub = ChannelSubscriber(lowstate_topic, LowStateHG)
    self._sub.Init(self._on_low_state, 10)

    self._wait_for_low_state()
    self._init_cmd_hg()

  # ── DDS callbacks / IO ─────────────────────────────────────────────────────
  def _on_low_state(self, msg) -> None:
    self.low_state = msg
    self.mode_machine = msg.mode_machine
    self.remote.set(msg.wireless_remote)

  def _wait_for_low_state(self) -> None:
    while self.low_state.tick == 0:
      time.sleep(self.control_dt)
    print("[G1] connected to robot (LowState received).")

  def _init_cmd_hg(self) -> None:
    self.low_cmd.mode_machine = self.mode_machine
    self.low_cmd.mode_pr = 0  # PR series control
    for m in self.low_cmd.motor_cmd:
      m.mode, m.q, m.qd, m.kp, m.kd, m.tau = 1, 0.0, 0.0, 0.0, 0.0, 0.0

  def send(self) -> None:
    self.low_cmd.crc = self._CRC.Crc(self.low_cmd)
    self._pub.Write(self.low_cmd)

  # ── State read (in joint_names order) ──────────────────────────────────────
  def read_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
    qj = np.empty(self.n, dtype=np.float64)
    dqj = np.empty(self.n, dtype=np.float64)
    for i in range(self.n):
      ms = self.low_state.motor_state[int(self.dof2motor[i])]
      qj[i], dqj[i] = ms.q, ms.dq
    return qj, dqj

  def read_imu(self) -> tuple[np.ndarray, np.ndarray]:
    quat = np.asarray(self.low_state.imu_state.quaternion, dtype=np.float64)  # wxyz
    gyro = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float64)
    return quat, gyro

  # ── Safe startup / shutdown ────────────────────────────────────────────────
  def zero_torque_wait_start(self) -> None:
    print("[G1] zero-torque. Hoist the robot, then press START on the remote…")
    while self.remote.button[KeyMap.start] != 1:
      for m in self.low_cmd.motor_cmd:
        m.q = m.qd = m.kp = m.kd = m.tau = 0.0
      self.send()
      time.sleep(self.control_dt)

  def move_to_default(self, default_joint_pos: np.ndarray, kp_scale: float, kd_scale: float) -> None:
    print("[G1] ramping to default pose (2 s)…")
    steps = max(int(2.0 / self.control_dt), 1)
    q0, _ = self.read_joint_state()
    for s in range(steps):
      a = (s + 1) / steps
      for i in range(self.n):
        m = self.low_cmd.motor_cmd[int(self.dof2motor[i])]
        m.q = float(q0[i] * (1 - a) + default_joint_pos[i] * a)
        m.qd, m.tau = 0.0, 0.0
        m.kp = float(self.kp[i] * kp_scale)
        m.kd = float(self.kd[i] * kd_scale)
      self.send()
      time.sleep(self.control_dt)

  def hold_default_wait_a(self, default_joint_pos: np.ndarray, kp_scale: float, kd_scale: float) -> None:
    print("[G1] holding default. Press A to start policy tracking…")
    while self.remote.button[KeyMap.A] != 1:
      for i in range(self.n):
        m = self.low_cmd.motor_cmd[int(self.dof2motor[i])]
        m.q = float(default_joint_pos[i])
        m.qd, m.tau = 0.0, 0.0
        m.kp = float(self.kp[i] * kp_scale)
        m.kd = float(self.kd[i] * kd_scale)
      self.send()
      time.sleep(self.control_dt)

  def send_targets(self, target: np.ndarray, kp_scale: float, kd_scale: float) -> None:
    for i in range(self.n):
      m = self.low_cmd.motor_cmd[int(self.dof2motor[i])]
      if self.dry_run:  # limp: compute everything but do not energize
        m.q = m.qd = m.kp = m.kd = m.tau = 0.0
      else:
        m.q = float(target[i])
        m.qd, m.tau = 0.0, 0.0
        m.kp = float(self.kp[i] * kp_scale)
        m.kd = float(self.kd[i] * kd_scale)
    self.send()

  def damping_stop(self) -> None:
    for m in self.low_cmd.motor_cmd:
      m.q = m.qd = m.kp = m.tau = 0.0
      m.kd = 8.0
    self.send()
    print("[G1] damping state — exit.")


# ══════════════════════════════════════════════════════════════════════════════
# Observation (real state + live command) — reuses YAHMP's assembly verbatim
# ══════════════════════════════════════════════════════════════════════════════
def _real_terms(spec, reference, frame, qj, dqj, quat, gyro, default_joint_pos, prev_action):
  """Build the per-step observation terms from real robot state + live command.

  OBSERVATION LAYOUT — dims below are for the shipped `g1_yahmp.onnx`
  (num_joints N = 29). The code reads the true sizes from the ONNX metadata, so
  it adapts automatically if you re-export a different policy.

    term                dim    source
    ─────────────────── ────── ───────────────────────────────────────────────
    command               64    motion reference = 2N + 6  (breakdown below)
    base_ang_vel           3    IMU gyroscope, body frame
    projected_gravity      3    gravity rotated into body frame (from IMU quat)
    joint_pos             29    encoder q − default_joint_pos            (= N)
    joint_vel             29    encoder dq                               (= N)
    actions               29    previous policy output      (= action_dim = N)
    history             1570    last 10 frames × 157-dim block (10 × 157)
    ─────────────────── ────── ───────────────────────────────────────────────
    TOTAL               1727    == spec.observation_dim

    command (64) = joint_pos_ref 29 | joint_vel_ref 29 | anchor_lin_vel_xy 2 |
                   anchor_ang_vel_yaw 1 | root_height 1 | root_roll,pitch 2
    history block (157) = command 64 + base_ang_vel 3 + projected_gravity 3 +
                   joint_pos 29 + joint_vel 29 + actions 29   (buffered 10 deep)

  The `history` term is assembled from these six terms inside
  `_build_observation()`; this function only produces the six current terms.
  """
  if any(t.name == "base_lin_vel" for t in spec.observation_terms):
    raise NotImplementedError(
      "This policy's observation includes `base_lin_vel` (privileged/teacher). "
      "It is not deployable on hardware without a base-velocity estimator."
    )
  return {
    "command": _command_value(spec, reference, 0.0, frame),                          # (64,)  = 2N + 6
    "base_ang_vel": gyro.astype(np.float32),                                         # (3,)
    "projected_gravity": _quat_rotate_inverse(quat, _GRAVITY_W).astype(np.float32),  # (3,)
    "joint_pos": (qj - default_joint_pos).astype(np.float32),                        # (29,) = N
    "joint_vel": dqj.astype(np.float32),                                             # (29,) = N
    "actions": prev_action.astype(np.float32),                                       # (29,) = action_dim
  }


def _decode_targets(spec, raw_action, frame, default_joint_pos, action_target_joint_indices):
  """Policy action → per-joint position targets (in joint_names / motor order).

  Dims (g1_yahmp): raw_action (29,) = action_dim; processed (29,); target (29,) = N.
  Residual semantics add the mocap reference joint angles; the result is the `q`
  setpoint sent to each motor (with Kp/Kd) in the control loop.
  """
  processed = raw_action.astype(np.float64) * spec.action_scale + spec.action_offset  # (29,)
  target = default_joint_pos.copy()                                                    # (29,) = N
  if spec.action_semantics == "residual_joint_position":
    target[action_target_joint_indices] = frame.joint_pos[action_target_joint_indices] + processed
  else:
    target[action_target_joint_indices] = processed
  return target


class TargetSmoother:
  """Low-pass + slew-rate limit on the per-joint position target.

  A YAHMP policy can fall into a high-frequency limit cycle on hardware (an
  ~8-9 Hz chatter in the raw action) that never showed up in sim, because the
  real actuation/sensing latency is not modelled in training. The motors cannot
  follow such a target and the base shakes instead of standing. This filter
  breaks that limit cycle mechanically by smoothing the *commanded* target only
  — the observation fed back to the policy is left untouched, so sim/real
  observation parity is preserved.

  Two independent, composable stages (either can be disabled):

    * EMA         y = ema · target + (1 - ema) · y_prev
                  `ema` is the weight of the new sample: 1.0 = off (passthrough),
                  smaller = heavier smoothing (and more lag). 0.3-0.5 is a good
                  starting range for suppressing chatter.
    * rate limit  |y - y_prev| <= max_rate · control_dt   (applied per joint)
                  A hard cap on how fast any target may slew, in rad/s. 0 = off.

  State is seeded with `init` (the default/held pose) so the first commanded
  target starts from the current stance without a step.
  """

  def __init__(
    self,
    *,
    init: np.ndarray,
    ema: float,
    max_rate: float,
    control_dt: float,
  ) -> None:
    if not 0.0 < ema <= 1.0:
      raise ValueError(f"--target-ema must be in (0, 1], got {ema}.")
    if max_rate < 0.0:
      raise ValueError(f"--target-max-rate must be >= 0, got {max_rate}.")
    self._prev = np.asarray(init, dtype=np.float64).copy()
    self._ema = float(ema)
    self._max_step = float(max_rate) * float(control_dt) if max_rate > 0.0 else 0.0

  @property
  def enabled(self) -> bool:
    return self._ema < 1.0 or self._max_step > 0.0

  def step(self, target: np.ndarray) -> np.ndarray:
    y = self._ema * np.asarray(target, dtype=np.float64) + (1.0 - self._ema) * self._prev
    if self._max_step > 0.0:
      y = self._prev + np.clip(y - self._prev, -self._max_step, self._max_step)
    self._prev = y
    return y


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def run(
  *,
  onnx_path: Path,
  net_iface: str,
  source_kind: str,
  ort_provider: str,
  npz_clip: Optional[Path],
  chingmu_kwargs: dict,
  joint_order: Optional[list[str]],
  vel_smoothing: float,
  kp_scale: float,
  kd_scale: float,
  target_ema: float = 1.0,
  target_max_rate: float = 0.0,
  mocap_calibration: str = "",
  calibrate: str = "",
  calibrate_seconds: float = 3.0,
  calibrate_height_target: float = 0.793,
  dof2motor: Optional[np.ndarray] = None,
  lowcmd_topic: str,
  lowstate_topic: str,
  dry_run: bool,
  record: str = "",
  record_steps: int = 0,
) -> None:
  spec = PolicySpec.from_onnx(onnx_path)
  spec.validate()
  n = len(spec.joint_names)
  default_joint_pos = np.asarray(spec.default_joint_pos, dtype=np.float64)
  action_target_joint_indices = np.asarray(
    [spec.joint_names.index(name) for name in spec.action_target_names], dtype=np.int32
  )
  kp, kd = _parse_gains(onnx_path, n)
  if dof2motor is None:
    dof2motor = np.arange(n, dtype=np.int32)  # motor order == joint_names order on G1

  session, providers = _create_onnx_session(onnx_path, ort_provider)
  input_name = session.get_inputs()[0].name
  output_name = session.get_outputs()[0].name

  # ── Live command source (mocap) ────────────────────────────────────────────
  state = MocapState(num_joints=n if joint_order is None else len(joint_order))
  if source_kind == "npz":
    if npz_clip is None:
      raise ValueError("--source npz requires --npz-clip.")
    source = NpzMockSource(state, spec, npz_clip)
  elif source_kind == "chingmu":
    source = ChingMuMocapSource(state, **chingmu_kwargs)
  else:
    raise ValueError(f"Unknown --source {source_kind!r}.")
  source.start()

  # ── Calibration capture (operator stands neutral; write JSON and exit) ───────
  if calibrate:
    raw_reference = LiveReference(state, spec, joint_order, vel_smoothing=0.0)
    try:
      capture_calibration(
        raw_reference,
        spec,
        seconds=calibrate_seconds,
        height_target=calibrate_height_target,
        out_path=calibrate,
      )
    finally:
      source.stop()
    print("[Calibrate] done — the robot was NOT energized. Re-run with "
          f"--mocap-calibration {calibrate} to teleoperate.")
    return

  calibration = MocapCalibration.load(mocap_calibration, spec) if mocap_calibration else None
  if calibration is not None:
    print(f"[INFO] mocap calibration loaded from {mocap_calibration}")
  reference = LiveReference(
    state, spec, joint_order, vel_smoothing=vel_smoothing, calibration=calibration
  )
  reference.wait_for_detection(timeout_s=15.0)

  # ── Robot bring-up ─────────────────────────────────────────────────────────
  print(f"[INFO] ONNX providers: {providers}")
  print(
    f"[INFO] gains: kp*{kp_scale:.2f} kd*{kd_scale:.2f}  "
    f"{'DRY-RUN (motors limp)' if dry_run else 'LIVE'}  control_dt={spec.control_dt:.3f}s"
  )
  robot = G1Controller(
    net_iface=net_iface,
    control_dt=spec.control_dt,
    num_joints=n,
    dof2motor=dof2motor,
    kp=kp,
    kd=kd,
    lowcmd_topic=lowcmd_topic,
    lowstate_topic=lowstate_topic,
    dry_run=dry_run,
  )
  robot.zero_torque_wait_start()
  robot.move_to_default(default_joint_pos, kp_scale, kd_scale)
  robot.hold_default_wait_a(default_joint_pos, kp_scale, kd_scale)

  # ── Initialise the observation history at the current state ─────────────────
  qj, dqj = robot.read_joint_state()
  quat, gyro = robot.read_imu()
  prev_action = np.zeros(spec.action_dim, dtype=np.float32)
  frame = reference.sample(0.0)
  terms = _real_terms(spec, reference, frame, qj, dqj, quat, gyro, default_joint_pos, prev_action)
  history = _initialize_history(spec, terms)

  # ── Sim2real data recorder (optional; --record) ─────────────────────────────
  recorder = (
    Sim2RealRecorder(
      record,
      spec.joint_names,
      action_target_names=spec.action_target_names,
      meta=dict(
        onnx_path=str(onnx_path),
        source=source_kind,
        kp_scale=kp_scale,
        kd_scale=kd_scale,
        target_ema=target_ema,
        target_max_rate=target_max_rate,
        vel_smoothing=vel_smoothing,
        mocap_calibration=mocap_calibration or None,
        calibrated=bool(calibration is not None),
        dry_run=dry_run,
        control_dt=spec.control_dt,
        action_semantics=spec.action_semantics,
      ),
    )
    if record
    else None
  )

  # ── Target smoother (breaks the sim2real high-freq limit cycle) ─────────────
  target_smoother = TargetSmoother(
    init=default_joint_pos,
    ema=target_ema,
    max_rate=target_max_rate,
    control_dt=spec.control_dt,
  )
  if target_smoother.enabled:
    print(
      f"[INFO] target smoothing ON: ema={target_ema:.2f} "
      f"max_rate={target_max_rate:.2f} rad/s "
      f"({target_max_rate * spec.control_dt * 180.0 / np.pi:.1f}°/step)"
    )
  else:
    print("[INFO] target smoothing OFF (raw policy target sent to motors).")

  # ── Control loop ───────────────────────────────────────────────────────────
  print("[G1] policy running. Press SELECT to stop (→ damping).")
  prev_loop_start: Optional[float] = None
  try:
    while robot.remote.button[KeyMap.select] != 1:
      t0 = time.perf_counter()

      qj, dqj = robot.read_joint_state()   # each (29,) = N, in joint_names/motor order
      quat, gyro = robot.read_imu()        # quat (4,) wxyz, gyro (3,)
      frame = reference.sample(0.0)        # MotionFrame: joint_pos/vel (29,), root pose/vel
      terms = _real_terms(spec, reference, frame, qj, dqj, quat, gyro, default_joint_pos, prev_action)
      obs = _build_observation(spec, terms, history)  # (1727,) for g1_yahmp = 6 terms + history(1570)

      t_infer0 = time.perf_counter()
      raw_action = (
        session.run([output_name], {input_name: obs[None, :].astype(np.float32)})[0]
        .reshape(-1)
        .astype(np.float32)  # (29,) = action_dim
      )
      infer_ms = (time.perf_counter() - t_infer0) * 1e3
      target = _decode_targets(spec, raw_action, frame, default_joint_pos, action_target_joint_indices)
      target = target_smoother.step(target)  # smooth the commanded target only (obs left untouched)
      robot.send_targets(target, kp_scale, kd_scale)

      prev_action = raw_action
      _append_history(spec, history, terms)

      dt = time.perf_counter() - t0
      if dt > 1.5 * spec.control_dt:
        print(f"[warn] control loop overrun: {dt * 1e3:.1f} ms > {spec.control_dt * 1e3:.1f} ms")

      if recorder is not None:
        recorder.step(
          wall_time=t0,
          ref_joint_pos=frame.joint_pos,
          target_joint_pos=target,
          actual_joint_pos=qj,
          actual_joint_vel=dqj,
          quat=quat,
          ang_vel=gyro,
          proj_grav=terms["projected_gravity"],
          raw_action=raw_action,
          root_cmd=terms["command"][2 * n : 2 * n + 6],  # [vx,vy,wyaw,height,roll,pitch]
          timing=dict(
            infer_ms=infer_ms,
            step_ms=dt * 1e3,
            period_ms=(t0 - prev_loop_start) * 1e3 if prev_loop_start is not None else float("nan"),
            overrun=dt > spec.control_dt,
          ),
        )
        if record_steps > 0 and recorder.count >= record_steps:
          print("[Recorder] reached --record-steps limit, stopping recording.")
          recorder.close()
          recorder = None
      prev_loop_start = t0

      time.sleep(max(0.0, spec.control_dt - dt))
  except KeyboardInterrupt:
    pass
  finally:
    robot.damping_stop()
    source.stop()
    if recorder is not None and recorder.count > 0:
      recorder.close()


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Deploy a YAHMP ONNX policy on a real Unitree G1 (mocap teleop).")
  p.add_argument("--onnx-path", type=Path, required=True)
  p.add_argument("--net", dest="net_iface", type=str, required=True, help="DDS network interface, e.g. enp4s0.")
  p.add_argument("--source", choices=("npz", "chingmu"), default="chingmu")
  p.add_argument("--ort-provider", choices=("auto", "cpu", "cuda"), default="cpu")
  p.add_argument("--vel-smoothing", type=float, default=0.7)
  p.add_argument("--kp-scale", type=float, default=1.0, help="Scale trained Kp (start small, e.g. 0.25).")
  p.add_argument("--kd-scale", type=float, default=1.0, help="Scale trained Kd (start small, e.g. 0.5).")
  p.add_argument("--target-ema", type=float, default=1.0, help="EMA on the commanded target: weight of the new sample each step, (0,1]. 1.0=off. Try 0.3-0.5 to suppress high-frequency action chatter.")
  p.add_argument("--target-max-rate", type=float, default=0.0, help="Per-joint slew-rate cap on the target in rad/s. 0=off. e.g. 6.0 -> ~6.9 deg per 20ms step.")
  # Mocap neutral-pose calibration
  p.add_argument("--mocap-calibration", type=str, default="", help="Path to a neutral-pose calibration JSON (from --calibrate) to apply to the live command.")
  p.add_argument("--calibrate", type=str, default="", help="CAPTURE mode: operator stands neutral; write a calibration JSON to this path and exit (robot NOT energized).")
  p.add_argument("--calibrate-seconds", type=float, default=3.0, help="Seconds to average while capturing the neutral pose (--calibrate).")
  p.add_argument("--calibrate-height-target", type=float, default=0.793, help="Nominal pelvis height (m) the neutral pose maps to (--calibrate). G1 default ~0.793.")
  p.add_argument("--dry-run", action="store_true", help="Run the full loop but keep motors limp (kp=kd=0).")
  p.add_argument("--record", type=str, default="", help="Sim2real CSV recording prefix, empty = off. e.g. --record ./sim2real_data/run")
  p.add_argument("--record-steps", type=int, default=0, help="Max steps to record (0 = unlimited).")
  p.add_argument("--lowcmd-topic", type=str, default="rt/lowcmd")
  p.add_argument("--lowstate-topic", type=str, default="rt/lowstate")
  p.add_argument("--joint2motor", type=Path, default=None, help="JSON list mapping joint index -> motor index (default: identity).")
  # NPZ mock
  p.add_argument("--npz-clip", type=Path, default=None)
  # ChingMu
  p.add_argument("--chingmu-dll", type=str, default=None)
  p.add_argument("--chingmu-host", type=str, default=None)
  p.add_argument("--sensor-root", type=int, default=301)
  p.add_argument("--sensor-joint-first", type=int, default=302)
  p.add_argument("--num-joints", type=int, default=None)
  p.add_argument("--pos-scale", type=float, default=0.001)
  p.add_argument("--joint-order", type=Path, default=None)
  return p


def main() -> None:
  from yahmp.scripts.deploy.run_yahmp_onnx_mocap import _VENDORED_CHINGMU_DLL

  args = _build_argparser().parse_args()

  joint_order = None
  if args.joint_order is not None:
    joint_order = json.loads(Path(args.joint_order).read_text())
    if not isinstance(joint_order, list):
      raise SystemExit("--joint-order JSON must be a list of joint names.")

  dof2motor = None
  if args.joint2motor is not None:
    dof2motor = np.asarray(json.loads(Path(args.joint2motor).read_text()), dtype=np.int32)

  chingmu_kwargs = {}
  if args.source == "chingmu":
    if not args.chingmu_host:
      raise SystemExit("--source chingmu requires --chingmu-host.")
    dll_path = Path(args.chingmu_dll).expanduser() if args.chingmu_dll else _VENDORED_CHINGMU_DLL
    if not dll_path.is_file():
      raise SystemExit(f"ChingMu DLL not found: {dll_path}. Pass --chingmu-dll.")
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
    net_iface=str(args.net_iface),
    source_kind=str(args.source),
    ort_provider=str(args.ort_provider),
    npz_clip=args.npz_clip.expanduser().resolve() if args.npz_clip else None,
    chingmu_kwargs=chingmu_kwargs,
    joint_order=joint_order,
    vel_smoothing=float(args.vel_smoothing),
    kp_scale=float(args.kp_scale),
    kd_scale=float(args.kd_scale),
    target_ema=float(args.target_ema),
    target_max_rate=float(args.target_max_rate),
    mocap_calibration=str(args.mocap_calibration),
    calibrate=str(args.calibrate),
    calibrate_seconds=float(args.calibrate_seconds),
    calibrate_height_target=float(args.calibrate_height_target),
    dof2motor=dof2motor,
    lowcmd_topic=str(args.lowcmd_topic),
    lowstate_topic=str(args.lowstate_topic),
    dry_run=bool(args.dry_run),
    record=str(args.record),
    record_steps=int(args.record_steps),
  )


if __name__ == "__main__":
  main()
