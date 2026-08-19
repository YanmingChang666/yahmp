"""Deploy the original TWIST2 ONNX policy on a real Unitree G1 from LIVE mocap.

Sim2real counterpart of `run_twist2_onnx_mocap.py`. It reuses:
  * the Unitree G1 DDS controller + bring-up/shutdown from `run_yahmp_onnx_real.py`
    (`G1Controller`: zero-torque → move-to-default → run → damping),
  * the mocap source + `LiveReference` from `run_yahmp_onnx_mocap.py`,
  * the TWIST2 observation / action-decode from `twist2_mocap_common.py`.

The obs/control are byte-identical to the sim2sim runner, so a policy verified in
`run_twist2_onnx_mocap.py` transfers here unchanged. Gains come from the TWIST2
control profile (not ONNX metadata, which a TWIST2 net lacks); the on-board PD is
`tau = kp·(target − q) − kd·dq`, matching the MuJoCo position actuators.

┌────────────────────────────────────────────────────────────────────────────┐
│  SAFETY: hoist the robot. Start with --dry-run, then small gains             │
│  (--kp-scale 0.25 --kd-scale 0.5), and a stability-biased --track-gain.      │
└────────────────────────────────────────────────────────────────────────────┘

    # 0) capture neutral-pose calibration (robot NOT energized):
    uv run python -m yahmp.scripts.deploy.run_twist2_onnx_real \
        --net enp4s0 --source chingmu --chingmu-host MCAvatar@192.168.123.112 \
        --sensor-root 301 --sensor-joint-first 302 \
        --joint-order config/chingmu_joint_order.json \
        --calibrate config/chingmu_calibration.json

    # 1) dry-run the full loop (motors limp), then go live with small gains:
    uv run python -m yahmp.scripts.deploy.run_twist2_onnx_real \
        --net enp4s0 --source chingmu --chingmu-host MCAvatar@192.168.123.112 \
        --sensor-root 301 --sensor-joint-first 302 \
        --joint-order config/chingmu_joint_order.json \
        --mocap-calibration config/chingmu_calibration.json \
        --kp-scale 0.25 --kd-scale 0.5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from yahmp.scripts.deploy.mocap_calibration_gui import run_gui_joint_calibration
from yahmp.scripts.deploy.run_twist2_onnx_mujoco import (
  CONTROL_PROFILES,
  DEFAULT_MJLAB_TASK_ID,
  TWIST2_DEFAULT_DOF_POS,
  TWIST2_G1_JOINT_NAMES,
  _build_task_scene,
  _create_onnx_session,
  _init_history,
  _joint_addresses,
  _onnx_input_dim,
  _root_addresses,
)
from yahmp.scripts.deploy.run_yahmp_onnx_mocap import (
  ChingMuMocapSource,
  LiveReference,
  MocapCalibration,
  MocapState,
  NpzMockSource,
  TargetSmoother,
  _VENDORED_CHINGMU_DLL,
  capture_calibration,
)
from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import _quat_rotate_inverse
from yahmp.scripts.deploy.run_yahmp_onnx_real import G1Controller, KeyMap
from yahmp.scripts.deploy.run_yahmp_onnx_recorder import Sim2RealRecorder
from yahmp.scripts.deploy.twist2_mocap_common import (
  Twist2Spec,
  build_twist2_obs,
  decode_pd_target,
  recorder_root_cmd,
  resolve_future_block,
)

_GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def run(
  *,
  onnx_path: Path,
  net_iface: str,
  ort_provider: str,
  source_kind: str,
  npz_clip: Optional[Path],
  chingmu_kwargs: dict,
  joint_order: Optional[list[str]],
  control_profile_name: str,
  clip_actions: float,
  zero_ankle_vel: bool,
  history_init: str,
  control_dt: float,
  vel_smoothing: float,
  kp_scale: float,
  kd_scale: float,
  target_ema: float,
  target_max_rate: float,
  mocap_calibration: str,
  calibrate: str,
  calibrate_seconds: float,
  calibrate_height_target: float,
  track_legs_only: bool,
  track_gain: float,
  dof2motor: Optional[np.ndarray],
  lowcmd_topic: str,
  lowstate_topic: str,
  dry_run: bool,
  record: str,
  record_steps: int,
) -> None:
  session, providers = _create_onnx_session(onnx_path, ort_provider)
  input_name = session.get_inputs()[0].name
  output_name = session.get_outputs()[0].name
  input_dim = _onnx_input_dim(session)
  include_future_block = resolve_future_block(input_dim)

  profile = CONTROL_PROFILES[control_profile_name]()
  kp = profile.stiffness.copy()
  kd = profile.damping.copy()
  default_joint_pos = TWIST2_DEFAULT_DOF_POS.copy()
  n = len(TWIST2_G1_JOINT_NAMES)
  spec = Twist2Spec(control_dt=float(control_dt))
  if dof2motor is None:
    dof2motor = np.arange(n, dtype=np.int32)  # motor order == joint order on G1

  # ── Live command source (mocap) ─────────────────────────────────────────────
  state = MocapState(num_joints=len(joint_order) if joint_order else n)
  if source_kind == "npz":
    if npz_clip is None:
      raise ValueError("--source npz requires --npz-clip.")
    source = NpzMockSource(state, spec, npz_clip)
  elif source_kind == "chingmu":
    source = ChingMuMocapSource(state, **chingmu_kwargs)
  else:
    raise ValueError(f"Unknown --source {source_kind!r}.")
  source.start()

  # ── Calibration (operator stands neutral; robot NOT energized) ───────────────
  # Interactive MuJoCo GUI: per-joint offset sliders + root R/P/Y + height, with
  # the automatic neutral-pose capture kept as a button. Falls back to headless
  # auto-capture when no display/tkinter is available.
  if calibrate:
    raw_reference = LiveReference(state, spec, joint_order, vel_smoothing=0.0)
    seed = MocapCalibration.load(calibrate, spec) if Path(calibrate).is_file() else None
    try:
      raw_reference.wait_for_detection(timeout_s=15.0)
      model, viewer_cfg, _physics_dt, _control_dt = _build_task_scene(DEFAULT_MJLAB_TASK_ID)
      data = mujoco.MjData(model)
      joint_qpos_adr, _jqv = _joint_addresses(model, TWIST2_G1_JOINT_NAMES)
      root_qpos_adr, _rqv = _root_addresses(model, "pelvis")
      ran = run_gui_joint_calibration(
        model=model, data=data, spec=spec, reference=raw_reference, viewer_cfg=viewer_cfg,
        joint_qpos_adr=joint_qpos_adr, root_qpos_adr=root_qpos_adr, out_path=calibrate,
        height_target=calibrate_height_target, seed=seed, root_body_name="pelvis",
        title="TWIST2 mocap calibration — per-joint offsets",
      )
      if not ran:
        capture_calibration(
          raw_reference, spec, seconds=calibrate_seconds,
          height_target=calibrate_height_target, out_path=calibrate,
        )
    finally:
      source.stop()
    print(f"[Calibrate] done (robot NOT energized). Re-run with "
          f"--mocap-calibration {calibrate} to teleoperate.")
    return

  calibration = MocapCalibration.load(mocap_calibration, spec) if mocap_calibration else None
  if calibration is not None:
    print(f"[INFO] mocap calibration loaded from {mocap_calibration}")
  reference = LiveReference(
    state, spec, joint_order, vel_smoothing=vel_smoothing, calibration=calibration,
    track_legs_only=track_legs_only, track_gain=track_gain,
  )
  if track_legs_only:
    print("[INFO] track-legs-only: waist + arms held at default; tracking legs only.")
  if track_gain != 1.0:
    print(f"[INFO] track-gain={track_gain:.2f}: command blended toward default.")
  reference.wait_for_detection(timeout_s=15.0)

  # ── Robot bring-up ──────────────────────────────────────────────────────────
  print(f"[INFO] TWIST2 sim2real | providers={providers} obs={input_dim} "
        f"future={include_future_block} profile={profile.name}")
  print(f"[INFO] gains: kp*{kp_scale:.2f} kd*{kd_scale:.2f}  "
        f"{'DRY-RUN (motors limp)' if dry_run else 'LIVE'}  control_dt={control_dt:.3f}s")
  robot = G1Controller(
    net_iface=net_iface, control_dt=control_dt, num_joints=n, dof2motor=dof2motor,
    kp=kp, kd=kd, lowcmd_topic=lowcmd_topic, lowstate_topic=lowstate_topic, dry_run=dry_run,
  )
  robot.zero_torque_wait_start()
  robot.move_to_default(default_joint_pos, kp_scale, kd_scale)
  robot.hold_default_wait_a(default_joint_pos, kp_scale, kd_scale)

  # ── Seed observation history at the current state ────────────────────────────
  # An empty history is fine here: build_twist2_obs only flattens it, and we keep
  # just `current` (the 127-dim mimic+proprio block) to seed the real history.
  qj, dqj = robot.read_joint_state()
  quat, gyro = robot.read_imu()
  last_action = np.zeros(n, dtype=np.float32)
  frame = reference.sample(0.0)
  _, current = build_twist2_obs(
    frame=frame, dof_pos=qj, dof_vel=dqj, quat=quat, ang_vel=gyro,
    last_action=last_action, history=deque(),
    include_future_block=include_future_block, zero_ankle_vel=zero_ankle_vel,
  )
  history = _init_history(history_init, current)

  smoother = TargetSmoother(
    init=default_joint_pos.copy(), ema=target_ema, max_rate=target_max_rate,
    control_dt=control_dt,
  )
  if smoother.enabled:
    print(f"[INFO] target smoothing ON: ema={target_ema:.2f} max_rate={target_max_rate:.2f} rad/s")

  recorder = (
    Sim2RealRecorder(
      record, TWIST2_G1_JOINT_NAMES, action_target_names=TWIST2_G1_JOINT_NAMES,
      meta=dict(
        policy="twist2", onnx_path=str(onnx_path), source=source_kind,
        control_profile=profile.name, kp_scale=kp_scale, kd_scale=kd_scale,
        target_ema=target_ema, vel_smoothing=vel_smoothing,
        mocap_calibration=mocap_calibration or None,
        calibrated=bool(calibration is not None), dry_run=dry_run,
        control_dt=control_dt, track_gain=track_gain,
      ),
    )
    if record
    else None
  )

  # ── Control loop ────────────────────────────────────────────────────────────
  print("[G1] policy running. Press SELECT to stop (→ damping).")
  prev_loop_start: Optional[float] = None
  try:
    while robot.remote.button[KeyMap.select] != 1:
      t0 = time.perf_counter()

      qj, dqj = robot.read_joint_state()   # (29,) in joint/motor order
      quat, gyro = robot.read_imu()        # quat (4,) wxyz, gyro (3,) body-frame
      frame = reference.sample(0.0)

      obs, current = build_twist2_obs(
        frame=frame, dof_pos=qj, dof_vel=dqj, quat=quat, ang_vel=gyro,
        last_action=last_action, history=history,
        include_future_block=include_future_block, zero_ankle_vel=zero_ankle_vel,
      )
      t_infer0 = time.perf_counter()
      raw_action = (
        session.run([output_name], {input_name: obs[None, :].astype(np.float32)})[0]
        .reshape(-1)
        .astype(np.float32)
      )
      infer_ms = (time.perf_counter() - t_infer0) * 1e3

      pd_target = decode_pd_target(raw_action, profile, clip_actions)
      pd_target = smoother.step(pd_target)
      robot.send_targets(pd_target, kp_scale, kd_scale)

      last_action = raw_action.copy()
      history.append(current)

      dt = time.perf_counter() - t0
      if dt > 1.5 * control_dt:
        print(f"[warn] control loop overrun: {dt * 1e3:.1f} ms > {control_dt * 1e3:.1f} ms")

      if recorder is not None:
        recorder.step(
          wall_time=t0, ref_joint_pos=frame.joint_pos, target_joint_pos=pd_target,
          actual_joint_pos=qj, actual_joint_vel=dqj, quat=quat, ang_vel=gyro,
          proj_grav=_quat_rotate_inverse(quat, _GRAVITY_W).astype(np.float32),
          raw_action=raw_action, root_cmd=recorder_root_cmd(frame),
          timing=dict(
            infer_ms=infer_ms, step_ms=dt * 1e3,
            period_ms=(t0 - prev_loop_start) * 1e3 if prev_loop_start is not None else float("nan"),
            overrun=dt > control_dt,
          ),
        )
        if record_steps > 0 and recorder.count >= record_steps:
          print("[Recorder] reached --record-steps limit, stopping recording.")
          recorder.close()
          recorder = None
      prev_loop_start = t0

      time.sleep(max(0.0, control_dt - dt))
  except KeyboardInterrupt:
    pass
  finally:
    robot.damping_stop()
    source.stop()
    if recorder is not None and recorder.count > 0:
      recorder.close()


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--onnx-path", type=Path, default=Path("assets/models/twist2_1017_25k.onnx"))
  p.add_argument("--net", dest="net_iface", type=str, required=True, help="DDS network interface, e.g. enp4s0.")
  p.add_argument("--ort-provider", choices=("auto", "cpu", "cuda"), default="cpu")
  p.add_argument("--source", choices=("npz", "chingmu"), default="chingmu")
  # TWIST2 control
  p.add_argument("--control-profile", choices=tuple(CONTROL_PROFILES.keys()), default="twist2-real-yaml")
  p.add_argument("--clip-actions", type=float, default=10.0)
  p.add_argument("--zero-ankle-vel", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--history-init", choices=("zeros", "current"), default="zeros")
  p.add_argument("--control-dt", type=float, default=0.02, help="Policy control period (s). 50 Hz = 0.02.")
  # Safety / gains
  p.add_argument("--kp-scale", type=float, default=0.25, help="Scale profile Kp (start small).")
  p.add_argument("--kd-scale", type=float, default=0.5, help="Scale profile Kd (start small).")
  p.add_argument("--dry-run", action="store_true", help="Run the full loop but keep motors limp.")
  # Reference shaping
  p.add_argument("--vel-smoothing", type=float, default=0.7, help="EMA [0,1) on finite-diff reference velocities.")
  p.add_argument("--track-legs-only", action="store_true")
  p.add_argument("--track-gain", type=float, default=1.0)
  p.add_argument("--target-ema", type=float, default=1.0, help="EMA on the commanded PD target, (0,1]. 1=off.")
  p.add_argument("--target-max-rate", type=float, default=0.0)
  # Calibration
  p.add_argument("--mocap-calibration", type=str, default="")
  p.add_argument("--calibrate", type=str, default="", help="CAPTURE mode: write neutral-pose JSON here and exit.")
  p.add_argument("--calibrate-seconds", type=float, default=3.0)
  p.add_argument("--calibrate-height-target", type=float, default=0.793)
  # DDS
  p.add_argument("--lowcmd-topic", type=str, default="rt/lowcmd")
  p.add_argument("--lowstate-topic", type=str, default="rt/lowstate")
  p.add_argument("--joint2motor", type=Path, default=None, help="JSON list mapping joint index -> motor index.")
  # Recording
  p.add_argument("--record", type=str, default="")
  p.add_argument("--record-steps", type=int, default=0)
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
  args = _build_argparser().parse_args()

  joint_order = None
  if args.joint_order is not None:
    joint_order = json.loads(Path(args.joint_order).read_text())
    if not isinstance(joint_order, list):
      raise SystemExit("--joint-order JSON must be a list of joint names.")

  dof2motor = None
  if args.joint2motor is not None:
    dof2motor = np.asarray(json.loads(Path(args.joint2motor).read_text()), dtype=np.int32)

  chingmu_kwargs: dict = {}
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
      dll_path=str(dll_path), host=args.chingmu_host, sensor_root=args.sensor_root,
      sensor_joint_first=args.sensor_joint_first, num_joints=num_joints,
      pos_scale=args.pos_scale,
    )

  run(
    onnx_path=args.onnx_path.expanduser().resolve(),
    net_iface=str(args.net_iface),
    ort_provider=str(args.ort_provider),
    source_kind=str(args.source),
    npz_clip=args.npz_clip.expanduser().resolve() if args.npz_clip else None,
    chingmu_kwargs=chingmu_kwargs,
    joint_order=joint_order,
    control_profile_name=str(args.control_profile),
    clip_actions=float(args.clip_actions),
    zero_ankle_vel=bool(args.zero_ankle_vel),
    history_init=str(args.history_init),
    control_dt=float(args.control_dt),
    vel_smoothing=float(args.vel_smoothing),
    kp_scale=float(args.kp_scale),
    kd_scale=float(args.kd_scale),
    target_ema=float(args.target_ema),
    target_max_rate=float(args.target_max_rate),
    mocap_calibration=str(args.mocap_calibration),
    calibrate=str(args.calibrate),
    calibrate_seconds=float(args.calibrate_seconds),
    calibrate_height_target=float(args.calibrate_height_target),
    track_legs_only=bool(args.track_legs_only),
    track_gain=float(args.track_gain),
    dof2motor=dof2motor,
    lowcmd_topic=str(args.lowcmd_topic),
    lowstate_topic=str(args.lowstate_topic),
    dry_run=bool(args.dry_run),
    record=str(args.record),
    record_steps=int(args.record_steps),
  )


if __name__ == "__main__":
  main()
