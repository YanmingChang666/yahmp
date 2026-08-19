"""Drive the original TWIST2 ONNX policy from LIVE mocap in MuJoCo (sim2sim).

This is the mocap/teleop counterpart of `run_twist2_onnx_mujoco.py`: instead of
tracking a pre-recorded clip, the reference comes from the live ChingMu stream
(or an npz replayed as synthetic mocap), exactly like `run_yahmp_onnx_mocap.py`
does for a YAHMP policy — but the observation, action decode and PD control are
TWIST2's.

    Reference:   ChingMu / npz ─► MocapState ─► LiveReference ─► MotionFrame
    Observation: TWIST2 mimic(frame) + proprio(sim state) + history (+ future)
    Control:     pd_target = TWIST2_DEFAULT + clip(action)·scale  ─► data.ctrl

Examples
--------
    # 1) sanity check against an npz clip (should match run_twist2_onnx_mujoco):
    uv run python -m yahmp.scripts.deploy.run_twist2_onnx_mocap \
        --onnx-path assets/models/twist2_1017_25k.onnx \
        --source npz --npz-clip assets/motions/wbt/yuanditabu.npz

    # 2) live ChingMu teleop:
    uv run python -m yahmp.scripts.deploy.run_twist2_onnx_mocap \
        --onnx-path assets/models/twist2_1017_25k.onnx \
        --source chingmu --chingmu-host MCAvatar@192.168.123.112 \
        --sensor-root 301 --sensor-joint-first 302 \
        --joint-order config/chingmu_joint_order.json \
        --mocap-calibration config/chingmu_calibration.json \
        --record ./sim2real_data/twist2
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np

from yahmp.scripts.deploy.run_twist2_onnx_mujoco import (
  CONTROL_PROFILES,
  DEFAULT_MJLAB_TASK_ID,
  HISTORY_LEN,
  N_OBS_SINGLE,
  TWIST2_DEFAULT_DOF_POS,
  TWIST2_G1_JOINT_NAMES,
  _actuator_ids_for_joint_names,
  _build_task_scene,
  _configure_camera,
  _create_onnx_session,
  _init_history,
  _joint_addresses,
  _onnx_input_dim,
  _override_position_actuator_gains,
  _root_addresses,
  _set_reference_state,
  _set_twist2_default_state,
)
from yahmp.scripts.deploy.run_yahmp_onnx_mocap import (
  ChingMuMocapSource,
  LiveReference,
  MocapCalibration,
  MocapState,
  NpzMockSource,
  TargetSmoother,
  _VENDORED_CHINGMU_DLL,
)
from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import _quat_rotate_inverse
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
  ort_provider: str,
  source_kind: str,
  npz_clip: Optional[Path],
  chingmu_kwargs: dict,
  joint_order: Optional[list[str]],
  control_profile_name: str,
  clip_actions: float,
  zero_ankle_vel: bool,
  history_init: str,
  init_pose: str,
  vel_smoothing: float,
  mocap_calibration: str,
  track_legs_only: bool,
  track_gain: float,
  target_ema: float,
  target_max_rate: float,
  root_body_name: str,
  headless: bool,
  real_time: bool,
  max_time_s: Optional[float],
  record: str,
  record_steps: int,
) -> None:
  # ── Policy ──────────────────────────────────────────────────────────────────
  session, providers = _create_onnx_session(onnx_path, ort_provider)
  input_name = session.get_inputs()[0].name
  output_name = session.get_outputs()[0].name
  input_dim = _onnx_input_dim(session)
  include_future_block = resolve_future_block(input_dim)
  profile = CONTROL_PROFILES[control_profile_name]()

  # ── MuJoCo scene ────────────────────────────────────────────────────────────
  task_id = DEFAULT_MJLAB_TASK_ID
  model, viewer_cfg, physics_dt, control_dt = _build_task_scene(task_id)
  data = mujoco.MjData(model)
  if model.nu != 29:
    raise ValueError(f"Expected 29 MuJoCo actuators for TWIST2 G1, got {model.nu}.")

  joint_qpos_adr, joint_qvel_adr = _joint_addresses(model, TWIST2_G1_JOINT_NAMES)
  root_qpos_adr, root_qvel_adr = _root_addresses(model, root_body_name)
  actuator_ids = _actuator_ids_for_joint_names(model, TWIST2_G1_JOINT_NAMES)
  _override_position_actuator_gains(model, actuator_ids, profile)

  control_decimation = int(round(control_dt / physics_dt))
  if control_decimation <= 0:
    raise ValueError("Invalid control/physics dt combination.")

  spec = Twist2Spec(control_dt=float(control_dt))

  # ── Live reference source (ChingMu / npz) ───────────────────────────────────
  state = MocapState(
    num_joints=len(joint_order) if joint_order else len(TWIST2_G1_JOINT_NAMES)
  )
  if source_kind == "npz":
    if npz_clip is None:
      raise ValueError("--source npz requires --npz-clip.")
    source = NpzMockSource(state, spec, npz_clip)
  elif source_kind == "chingmu":
    source = ChingMuMocapSource(state, **chingmu_kwargs)
  else:
    raise ValueError(f"Unknown --source {source_kind!r}.")
  source.start()

  seed_exists = bool(mocap_calibration) and Path(mocap_calibration).is_file()
  calibration = MocapCalibration.load(mocap_calibration, spec) if seed_exists else None
  if mocap_calibration and calibration is None:
    raise SystemExit(f"--mocap-calibration file not found: {mocap_calibration}")
  if calibration is not None:
    print(f"[INFO] mocap calibration loaded from {mocap_calibration}")

  reference = LiveReference(
    state,
    spec,
    joint_order,
    vel_smoothing=vel_smoothing,
    calibration=calibration,
    track_legs_only=track_legs_only,
    track_gain=track_gain,
  )
  if track_legs_only:
    print("[INFO] track-legs-only: waist + arms held at default; tracking legs only.")
  if track_gain != 1.0:
    print(f"[INFO] track-gain={track_gain:.2f}: command blended toward default.")
  reference.wait_for_detection(timeout_s=15.0)

  # ── Initialise the simulated robot ──────────────────────────────────────────
  def reset_rollout() -> tuple[np.ndarray, deque]:
    frame0 = reference.sample(0.0)
    if init_pose == "reference":
      _set_reference_state(
        model, data, frame0, joint_qpos_adr, joint_qvel_adr, root_qpos_adr, root_qvel_adr
      )
    else:
      _set_twist2_default_state(
        model, data, joint_qpos_adr, joint_qvel_adr, root_qpos_adr, root_qvel_adr
      )
    last_action = np.zeros(29, dtype=np.float32)
    _, current = build_twist2_obs(
      frame=frame0,
      dof_pos=np.asarray(data.qpos[joint_qpos_adr], dtype=np.float64),
      dof_vel=np.asarray(data.qvel[joint_qvel_adr], dtype=np.float64),
      quat=np.asarray(data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7], dtype=np.float64),
      ang_vel=np.asarray(data.qvel[root_qvel_adr + 3 : root_qvel_adr + 6], dtype=np.float64),
      last_action=last_action,
      history=deque([np.zeros(N_OBS_SINGLE, dtype=np.float32)] * HISTORY_LEN),
      include_future_block=include_future_block,
      zero_ankle_vel=zero_ankle_vel,
    )
    return last_action, _init_history(history_init, current)

  last_action, history = reset_rollout()
  smoother = TargetSmoother(
    init=TWIST2_DEFAULT_DOF_POS.copy(), ema=target_ema, max_rate=target_max_rate,
    control_dt=control_dt,
  )

  print(f"[INFO] TWIST2 mocap sim2sim | providers={providers} obs={input_dim} "
        f"future={include_future_block} profile={profile.name} source={source_kind}")
  print(f"[INFO] dt: physics={physics_dt:.4f}s control={control_dt:.4f}s "
        f"decimation={control_decimation}  smoothing={'ON' if smoother.enabled else 'OFF'}")

  recorder = (
    Sim2RealRecorder(
      record,
      TWIST2_G1_JOINT_NAMES,
      action_target_names=TWIST2_G1_JOINT_NAMES,
      meta=dict(
        sim2sim=True, policy="twist2", onnx_path=str(onnx_path), task_id=task_id,
        source=source_kind, control_profile=profile.name, control_dt=control_dt,
        mocap_calibration=mocap_calibration or None,
        calibrated=bool(calibration is not None), target_ema=target_ema,
        track_gain=track_gain, track_legs_only=track_legs_only,
      ),
    )
    if record
    else None
  )

  # ── Control loop ────────────────────────────────────────────────────────────
  prev_loop_start: Optional[float] = None

  def policy_and_control() -> None:
    nonlocal last_action, history, prev_loop_start
    wall_t0 = time.perf_counter()

    frame = reference.sample(0.0)
    dof_pos = np.asarray(data.qpos[joint_qpos_adr], dtype=np.float64).copy()
    dof_vel = np.asarray(data.qvel[joint_qvel_adr], dtype=np.float64).copy()
    quat = np.asarray(data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7], dtype=np.float64).copy()
    ang_vel = np.asarray(data.qvel[root_qvel_adr + 3 : root_qvel_adr + 6], dtype=np.float64).copy()

    obs, current = build_twist2_obs(
      frame=frame, dof_pos=dof_pos, dof_vel=dof_vel, quat=quat, ang_vel=ang_vel,
      last_action=last_action, history=history,
      include_future_block=include_future_block, zero_ankle_vel=zero_ankle_vel,
    )
    if obs.shape != (input_dim,):
      raise ValueError(f"Built obs {obs.shape}, expected ({input_dim},).")

    t_infer0 = time.perf_counter()
    raw_action = (
      session.run([output_name], {input_name: obs[None, :].astype(np.float32)})[0]
      .reshape(-1)
      .astype(np.float32)
    )
    infer_ms = (time.perf_counter() - t_infer0) * 1e3

    last_action = raw_action.copy()
    pd_target = decode_pd_target(raw_action, profile, clip_actions)
    pd_target = smoother.step(pd_target)
    history.append(current)

    data.ctrl[actuator_ids] = pd_target
    for _ in range(control_decimation):
      mujoco.mj_step(model, data)

    if recorder is not None:
      step_ms = (time.perf_counter() - wall_t0) * 1e3
      recorder.step(
        wall_time=wall_t0,
        ref_joint_pos=frame.joint_pos,
        target_joint_pos=pd_target,
        actual_joint_pos=dof_pos,
        actual_joint_vel=dof_vel,
        quat=quat,
        ang_vel=ang_vel,
        proj_grav=_quat_rotate_inverse(quat, _GRAVITY_W).astype(np.float32),
        raw_action=raw_action,
        root_cmd=recorder_root_cmd(frame),
        timing=dict(
          infer_ms=infer_ms,
          step_ms=step_ms,
          period_ms=(wall_t0 - prev_loop_start) * 1e3 if prev_loop_start is not None else float("nan"),
          overrun=step_ms > control_dt * 1e3,
        ),
      )
    prev_loop_start = wall_t0

    if real_time:
      time.sleep(max(0.0, control_dt - (time.perf_counter() - wall_t0)))

  def run_loop(viewer: Any | None) -> None:
    t_start = time.perf_counter()
    while viewer is None or viewer.is_running():
      if max_time_s is not None and time.perf_counter() - t_start >= max_time_s:
        break
      policy_and_control()
      if viewer is not None:
        viewer.sync()
      if recorder is not None and record_steps > 0 and recorder.count >= record_steps:
        print("[Recorder] reached --record-steps limit, stopping recording.")
        break

  try:
    if headless:
      run_loop(None)
    else:
      import mujoco.viewer as mujoco_viewer

      with mujoco_viewer.launch_passive(model, data) as viewer:
        _configure_camera(viewer, model, root_body_name, viewer_cfg)
        run_loop(viewer)
  except KeyboardInterrupt:
    pass
  finally:
    source.stop()
    if recorder is not None and recorder.count > 0:
      recorder.close()


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--onnx-path", type=Path, default=Path("assets/models/twist2_1017_25k.onnx"))
  p.add_argument("--ort-provider", choices=("auto", "cpu", "cuda"), default="auto")
  p.add_argument("--source", choices=("npz", "chingmu"), default="chingmu")
  # TWIST2 control
  p.add_argument("--control-profile", choices=tuple(CONTROL_PROFILES.keys()), default="twist2-sim2sim")
  p.add_argument("--clip-actions", type=float, default=10.0)
  p.add_argument("--zero-ankle-vel", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--history-init", choices=("zeros", "current"), default="zeros")
  p.add_argument("--init-pose", choices=("reference", "twist2-default"), default="reference")
  p.add_argument("--root-body-name", type=str, default="pelvis")
  # Reference shaping (shared with the YAHMP mocap runner)
  p.add_argument("--vel-smoothing", type=float, default=0.0, help="EMA [0,1) on finite-diff reference velocities.")
  p.add_argument("--mocap-calibration", type=str, default="", help="Neutral-pose calibration JSON to apply.")
  p.add_argument("--track-legs-only", action="store_true")
  p.add_argument("--track-gain", type=float, default=1.0, help="Blend command toward default; 1=full track, 0=hold default.")
  p.add_argument("--target-ema", type=float, default=1.0, help="EMA on the commanded PD target, (0,1]. 1=off.")
  p.add_argument("--target-max-rate", type=float, default=0.0, help="Per-joint target slew cap (rad/s). 0=off.")
  # Loop
  p.add_argument("--headless", action="store_true")
  p.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--max-time-s", type=float, default=None)
  p.add_argument("--record", type=str, default="", help="CSV recording prefix, empty = off.")
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
    ort_provider=str(args.ort_provider),
    source_kind=str(args.source),
    npz_clip=args.npz_clip.expanduser().resolve() if args.npz_clip else None,
    chingmu_kwargs=chingmu_kwargs,
    joint_order=joint_order,
    control_profile_name=str(args.control_profile),
    clip_actions=float(args.clip_actions),
    zero_ankle_vel=bool(args.zero_ankle_vel),
    history_init=str(args.history_init),
    init_pose=str(args.init_pose),
    vel_smoothing=float(args.vel_smoothing),
    mocap_calibration=str(args.mocap_calibration),
    track_legs_only=bool(args.track_legs_only),
    track_gain=float(args.track_gain),
    target_ema=float(args.target_ema),
    target_max_rate=float(args.target_max_rate),
    root_body_name=str(args.root_body_name),
    headless=bool(args.headless),
    real_time=bool(args.real_time),
    max_time_s=args.max_time_s if args.max_time_s is None else float(args.max_time_s),
    record=str(args.record),
    record_steps=int(args.record_steps),
  )


if __name__ == "__main__":
  main()
