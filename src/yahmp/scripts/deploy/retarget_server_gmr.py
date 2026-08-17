#!/usr/bin/env python3
"""GMR retarget server — ChingMu human skeleton → robot qpos → redis.

This is the *retargeting half* of the TWIST2-style two-process teleop, and the
faithful analog of TWIST2's `xrobot_teleop_to_robot_w_hand.py`:

    ChingMu human skeleton (VRPN) ─► GMR.retarget ─► robot qpos ─► redis
                                                                     │
    yahmp deploy (--source redis) ◄─────────────────────────────────┘

Run it in the **`gmr` conda env** (python 3.10 + `general_motion_retargeting`),
NOT the yahmp env — hence this file imports no `yahmp` code and is launched as a
plain script:

    conda activate gmr
    python retarget_server_gmr.py \
        --chingmu-host MCAvatar@192.168.123.112 \
        --body-map config/chingmu_body_map.json \
        --robot unitree_g1 --src-human xrobot \
        --actual-human-height 1.6 --target-fps 100 --smooth 0.5

Then, in the yahmp env, the deploy reads the retargeted command straight from
redis (sim2sim first, hardware second):

    uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
        --task-id Mjlab-YAHMP-Unitree-G1 --onnx-path assets/models/g1_yahmp.onnx \
        --source redis

Why this fixes the VRPN instability (see the sim2real analysis):
  * GMR solves IK against the robot's own kinematics → feasible, convention-correct
    joint targets (no raw sign/offset/frame mismatch).
  * `offset_to_ground=True` → a physically grounded pelvis height (no floating).
  * The server publishes clean joint *velocities* (finite-diff + EMA at the 100 Hz
    retarget rate), so the policy's noisy `joint_vel_ref` channel goes away.
  * EMA smoothing + engage ramp mirror TWIST2's `--smooth_body` + interpolation.

┌── INTEGRATION POINTS you must fill for YOUR setup ────────────────────────────┐
│ 1. --body-map: JSON {chingmu_sensor_id: gmr_body_name}. The body names must be │
│    the ones GMR's chosen --src-human expects (print `retarget.ik_match_table1` │
│    keys). Sensor ids are your ChingMu human-skeleton segment ids.              │
│ 2. --src-human: GMR source preset for your input. TWIST2 uses "xrobot" (PICO). │
│    If GMR ships no ChingMu preset, add one, or map ChingMu → the xrobot names. │
└──────────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
import time
from typing import Optional

import numpy as np

_DEFAULT_KEY = "yahmp:mocap:pose"


# ══════════════════════════════════════════════════════════════════════════════
# Small pure helpers (unit-testable without GMR / ChingMu / mujoco)
# ══════════════════════════════════════════════════════════════════════════════
class EMA:
  """Per-element exponential moving average. alpha in [0,1): 0 = off."""

  def __init__(self, alpha: float) -> None:
    self.alpha = float(np.clip(alpha, 0.0, 0.99))
    self._v: Optional[np.ndarray] = None

  def __call__(self, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if self._v is None or self.alpha == 0.0:
      self._v = x.copy()
    else:
      self._v = self.alpha * self._v + (1.0 - self.alpha) * x
    return self._v

  def reset(self) -> None:
    self._v = None


def build_pose_message(
  seq: int,
  t: float,
  root_pos: np.ndarray,
  root_quat_wxyz: np.ndarray,
  joint_names: list[str],
  joint_pos: np.ndarray,
  joint_vel: Optional[np.ndarray],
  detected: bool = True,
) -> dict:
  """Assemble the redis payload consumed by RedisMocapSource (name-keyed joints)."""
  msg = {
    "seq": int(seq),
    "t": float(t),
    "root_pos": [float(v) for v in root_pos[:3]],
    "root_quat": [float(v) for v in root_quat_wxyz[:4]],  # wxyz (MuJoCo)
    "joint_pos": {n: float(v) for n, v in zip(joint_names, joint_pos)},
    "joint_vel": (
      None if joint_vel is None else {n: float(v) for n, v in zip(joint_names, joint_vel)}
    ),
    "detected": bool(detected),
  }
  return msg


def ramp_alpha(step: int, ramp_steps: int) -> float:
  """0→1 engage ramp (blend default→live) over `ramp_steps`. TWIST2 interpolation."""
  if ramp_steps <= 0:
    return 1.0
  return float(np.clip(step / ramp_steps, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# ChingMu human-skeleton reader (VRPN) — full pose per body segment
# ══════════════════════════════════════════════════════════════════════════════
class _Timeval(ctypes.Structure):
  _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class _VrpnTracker(ctypes.Structure):
  _fields_ = [
    ("msg_time", _Timeval),
    ("sensor", ctypes.c_int),
    ("frameCounter", ctypes.c_int),
    ("pos", ctypes.c_double * 3),
    ("quat", ctypes.c_double * 4),  # qx, qy, qz, qw  (xyzw — GMR/scipy convention)
  ]


class ChingMuSkeletonReader:
  """Reads ChingMu human-skeleton segments over the CMTrackerExternTC POLLING
  API (indexed by body_id, as TTRL's WaistBodyPollerLive does) into
  `{body_name: (pos_m[3], quat_xyzw[4])}` per `body_map`.

  The polling API (not the tracker-data callback) is where the human skeleton
  lives — the callback only carries rigid bodies. `body_map` therefore maps
  ChingMu **body_id** (from --probe-poll) → GMR smplx body name.

  Unlike the yahmp `ChingMuMocapSource` (which reads pre-retargeted *robot* joint
  angles), this reads the raw *human* body poses that GMR needs as its input.
  """

  def __init__(
    self,
    dll_path: str,
    host: str,
    body_map: dict[int, str],
    *,
    pos_scale: float = 0.001,
    poll_hz: float = 200.0,
    encoding: str = "gbk",
  ) -> None:
    self._dll = ctypes.CDLL(dll_path)
    self._host = bytes(host, encoding)
    # Keys prefixed with "_" are documentation (e.g. "_comment"), not body_ids.
    self._body_map = {
      int(k): str(v) for k, v in body_map.items() if not str(k).startswith("_")
    }
    self._scale = float(pos_scale)
    self._poll_dt = 1.0 / max(poll_hz, 1.0)
    self._lock = threading.Lock()
    self._bodies: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    self._running = False
    self._thread: Optional[threading.Thread] = None

  def start(self) -> None:
    self._dll.CMVrpnStartExtern()
    try:
      self._dll.CMVrpnEnableLog(False)
    except Exception:  # noqa: BLE001
      pass
    self._running = True
    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()
    print(f"[ChingMu] polling {len(self._body_map)} skeleton body_ids from "
          f"{self._host.decode(errors='replace')}")

  def _loop(self) -> None:
    body_pos = (ctypes.c_double * 3)()
    body_rot = (ctypes.c_double * 4)()  # xyzw
    timecode = (ctypes.c_int * 1)()
    tv = _Timeval()
    s = self._scale
    while self._running:
      for bid, name in self._body_map.items():
        det = self._dll.CMTrackerExternTC(
          self._host, bid, timecode, body_pos, body_rot, ctypes.byref(tv)
        )
        if det:
          pos = np.array([body_pos[0] * s, body_pos[1] * s, body_pos[2] * s])
          quat_xyzw = np.array([body_rot[0], body_rot[1], body_rot[2], body_rot[3]])
          with self._lock:
            self._bodies[name] = (pos, quat_xyzw)
      time.sleep(self._poll_dt)

  def get_human_data(self) -> Optional[dict]:
    """Return the GMR `human_data` dict, or None until all segments are seen."""
    with self._lock:
      if len(self._bodies) < len(self._body_map):
        return None
      return {k: (p.copy(), q.copy()) for k, (p, q) in self._bodies.items()}

  def pending(self) -> tuple[int, int, list[int]]:
    """(n_seen, n_total, sorted mapped body_ids not yet received)."""
    with self._lock:
      have = set(self._bodies)
    missing = sorted(sid for sid, name in self._body_map.items() if name not in have)
    return len(have), len(self._body_map), missing

  def stop(self) -> None:
    self._running = False
    if self._thread:
      self._thread.join(timeout=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Robot joint layout from the GMR target MJCF
# ══════════════════════════════════════════════════════════════════════════════
def robot_joint_layout(xml_path: str):
  """(joint_names, qpos_adr) for every 1-DOF joint, in qpos order. Needs mujoco."""
  import mujoco

  model = mujoco.MjModel.from_xml_path(str(xml_path))
  names, adr = [], []
  for j in range(model.njnt):
    jtype = int(model.jnt_type[j])
    if jtype in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)):
      continue  # skip the floating base / ball joints — those are the root
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    names.append(name)
    adr.append(int(model.jnt_qposadr[j]))
  return names, np.asarray(adr, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# Main retarget loop
# ══════════════════════════════════════════════════════════════════════════════
def _probe_sensors(dll_path: str, host: str, seconds: float, pos_scale: float,
                   encoding: str = "gbk") -> None:
  """Print every ChingMu VRPN sensor id seen (with position) so you can map ids
  to body segments — wiggle a limb and watch which id's position changes."""
  dll = ctypes.CDLL(dll_path)
  hostb = bytes(host, encoding)
  seen: dict[int, tuple] = {}
  lock = threading.Lock()
  cb_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.POINTER(_VrpnTracker))

  def _cb(_ptr, b):
    d = b.contents
    with lock:
      seen[int(d.sensor)] = (d.pos[0] * pos_scale, d.pos[1] * pos_scale, d.pos[2] * pos_scale)

  cb = cb_type(_cb)
  dll.CMVrpnStartExtern()
  try:
    dll.CMVrpnEnableLog(False)
  except Exception:  # noqa: BLE001
    pass
  dll.CMPluginConnectServer(hostb)
  dll.CMPluginRegisterTrackerData(hostb, None, cb)
  print(f"[Probe] listening {seconds:.0f}s on {host} — move each limb to identify ids…")
  t0 = time.time()
  while time.time() - t0 < seconds:
    dll.CMPluginRegisterTrackerData(hostb, None, cb)
    time.sleep(0.05)
    with lock:
      ids = sorted(seen)
    print(f"\r[Probe] {len(ids)} sensor ids: {ids}   ", end="", flush=True)
  print("\n[Probe] final snapshot (sensor_id -> position m):")
  with lock:
    for sid in sorted(seen):
      x, y, z = seen[sid]
      print(f"   {sid:4d}  ({x:+.3f}, {y:+.3f}, {z:+.3f})")
  print("[Probe] NOTE: this callback API returns RIGID BODIES. If you see only a few "
        "(not ~14 skeleton joints), your human skeleton is on the polling API — "
        "run --probe-poll instead.")
  os._exit(0)  # skip interpreter teardown; the VRPN C thread segfaults on shutdown


def _probe_poll(dll_path: str, host: str, max_body: int, seconds: float,
                pos_scale: float, encoding: str = "gbk") -> None:
  """Enumerate ChingMu body poses via the CMTrackerExternTC POLLING API (the one
  TTRL's WaistBodyPollerLive uses). Human-skeleton segments usually live here,
  indexed by body_id, when the tracker-data callback only shows rigid bodies."""
  dll = ctypes.CDLL(dll_path)
  hostb = bytes(host, encoding)
  dll.CMVrpnStartExtern()
  try:
    dll.CMVrpnEnableLog(False)
  except Exception:  # noqa: BLE001
    pass
  body_pos = (ctypes.c_double * 3)()
  body_rot = (ctypes.c_double * 4)()  # xyzw
  timecode = (ctypes.c_int * 1)()
  tv = _Timeval()
  print(f"[ProbePoll] polling body_id 0..{max_body} for {seconds:.0f}s — stand in view…")
  seen: dict[int, tuple] = {}
  t0 = time.time()
  while time.time() - t0 < seconds:
    for bid in range(max_body + 1):
      det = dll.CMTrackerExternTC(hostb, bid, timecode, body_pos, body_rot, ctypes.byref(tv))
      if det:
        seen[bid] = (body_pos[0] * pos_scale, body_pos[1] * pos_scale, body_pos[2] * pos_scale)
    time.sleep(0.05)
    print(f"\r[ProbePoll] detected body_ids: {sorted(seen)}   ", end="", flush=True)
  print(f"\n[ProbePoll] {len(seen)} body_ids detected (id -> position m):")
  for bid in sorted(seen):
    x, y, z = seen[bid]
    print(f"   body_id {bid:3d}  ({x:+.3f}, {y:+.3f}, {z:+.3f})")
  print("[ProbePoll] if ~14+ ids appear, that's your skeleton — map them in --body-map.")
  os._exit(0)


def _resolve_gmr_class(pkg):
  """The retargeter class, tolerant of fork naming (GeneralMotionRetargeting/GMR)."""
  for name in ("GeneralMotionRetargeting", "GMR"):
    cls = getattr(pkg, name, None)
    if cls is not None:
      return cls
  raise SystemExit(
    "general_motion_retargeting exposes no GeneralMotionRetargeting/GMR class. "
    f"Available top-level names: {[n for n in dir(pkg) if not n.startswith('_')]}. "
    "Tell me these and I'll adapt the import."
  )


def _resolve_robot_xml(pkg, robot: str, override: Optional[str]) -> str:
  """Robot MJCF path: --robot-xml wins; else the fork's ROBOT_XML_DICT."""
  if override:
    return override
  table = getattr(pkg, "ROBOT_XML_DICT", None)
  if table is None:
    raise SystemExit(
      "This GMR build has no ROBOT_XML_DICT — pass --robot-xml <path to robot MJCF>."
    )
  if robot not in table:
    raise SystemExit(
      f"--robot {robot!r} not in ROBOT_XML_DICT (keys: {list(table)}). Use --robot-xml."
    )
  return table[robot]


def run(args: argparse.Namespace) -> None:
  # Sensor discovery needs neither GMR nor a body-map — do it first and exit.
  if args.probe:
    _probe_sensors(args.chingmu_dll, args.chingmu_host, args.probe_seconds, args.pos_scale)
    return
  if args.probe_poll:
    _probe_poll(args.chingmu_dll, args.chingmu_host, args.probe_max_body,
                args.probe_seconds, args.pos_scale)
    return
  if not args.body_map:
    raise SystemExit("--body-map is required (or use --probe to discover sensor ids).")

  import redis
  import general_motion_retargeting as gmr_pkg  # gmr env only

  GMR = _resolve_gmr_class(gmr_pkg)
  robot_xml = _resolve_robot_xml(gmr_pkg, args.robot, args.robot_xml)

  ik_srcs = getattr(gmr_pkg, "IK_CONFIG_DICT", {})
  if ik_srcs and args.src_human not in ik_srcs:
    raise SystemExit(
      f"--src-human {args.src_human!r} not available in this GMR build. "
      f"Options: {list(ik_srcs)}. For live ChingMu streaming use 'smplx'."
    )

  body_map = json.loads(open(args.body_map, encoding="utf-8").read())
  reader = ChingMuSkeletonReader(
    dll_path=args.chingmu_dll,
    host=args.chingmu_host,
    body_map=body_map,
    pos_scale=args.pos_scale,
  )
  # NOTE: TWIST2's constructor is GMR(src_human=, tgt_robot=, actual_human_height=).
  # If your droid_gmr fork differs, the TypeError names the expected args — send it over.
  retarget = GMR(
    src_human=args.src_human,
    tgt_robot=args.robot,
    actual_human_height=args.actual_human_height,
  )
  joint_names, joint_adr = robot_joint_layout(robot_xml)
  print(f"[GMR] {args.robot}: {len(joint_names)} joints; retarget src={args.src_human}")

  r = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db)
  r.ping()
  print(f"[Redis] publishing to {args.redis_key!r} @ {args.target_fps:.0f} Hz")

  pos_ema = EMA(args.smooth)
  vel_ema = EMA(max(args.smooth, 0.5))  # velocities are noisier; smooth them harder
  reader.start()

  seq = 0
  dt_nominal = 1.0 / max(args.target_fps, 1.0)
  prev_q: Optional[np.ndarray] = None
  prev_t: Optional[float] = None
  default_q: Optional[np.ndarray] = None
  ramp_step = 0
  last_warn = 0.0
  try:
    while True:
      t_loop = time.perf_counter()
      human = reader.get_human_data()
      if human is None:
        if t_loop - last_warn > 2.0:
          n_have, n_total, missing = reader.pending()
          print(
            f"\r[GMR] waiting for ChingMu: have {n_have}/{n_total} segments; "
            f"missing sensor ids {missing} — fix these in --body-map    ",
            end="", flush=True,
          )
          last_warn = t_loop
        time.sleep(dt_nominal)
        continue

      qpos = np.asarray(retarget.retarget(human, offset_to_ground=True), dtype=np.float64)
      root_pos = qpos[0:3]
      root_quat = qpos[3:7]  # MuJoCo wxyz
      q_joints = qpos[joint_adr]

      # Engage ramp: blend default → live over --ramp-seconds on first detection.
      if default_q is None:
        default_q = q_joints.copy()
      a = ramp_alpha(ramp_step, int(args.ramp_seconds * args.target_fps))
      q_joints = (1.0 - a) * default_q + a * q_joints
      ramp_step += 1

      q_joints = pos_ema(q_joints)

      now = time.time()
      if prev_q is not None and prev_t is not None and now > prev_t:
        j_vel = vel_ema((q_joints - prev_q) / (now - prev_t))
      else:
        j_vel = np.zeros_like(q_joints)
      prev_q, prev_t = q_joints.copy(), now

      msg = build_pose_message(
        seq, now, root_pos, root_quat, joint_names, q_joints,
        None if args.no_joint_vel else j_vel,
      )
      r.set(args.redis_key, json.dumps(msg))
      seq += 1

      if args.measure_fps and seq % 100 == 0:
        print(f"[GMR] seq={seq} root_z={root_pos[2]:.3f} ramp={a:.2f}")

      sleep = dt_nominal - (time.perf_counter() - t_loop)
      if sleep > 0:
        time.sleep(sleep)
  except KeyboardInterrupt:
    pass
  finally:
    reader.stop()
    print("[GMR] retarget server stopped.")


def _build_argparser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="GMR retarget server: ChingMu human skeleton -> robot qpos -> redis.")
  p.add_argument("--chingmu-host", type=str, required=True, help="e.g. MCAvatar@192.168.123.112")
  p.add_argument("--chingmu-dll", type=str, required=True, help="Path to libCMVrpn.so.")
  p.add_argument("--body-map", type=str, default=None, help="JSON {chingmu_sensor_id: gmr_body_name}. Required unless --probe.")
  p.add_argument("--probe", action="store_true", help="Print live ChingMu rigid-body sensor ids (tracker-data callback) and exit.")
  p.add_argument("--probe-poll", action="store_true", help="Enumerate body_ids via the CMTrackerExternTC polling API (where the skeleton usually is) and exit.")
  p.add_argument("--probe-max-body", type=int, default=40, help="Highest body_id to poll in --probe-poll.")
  p.add_argument("--probe-seconds", type=float, default=15.0)
  p.add_argument("--pos-scale", type=float, default=0.001, help="ChingMu position units -> metres (mm=0.001).")
  p.add_argument("--robot", type=str, default="unitree_g1", help="GMR tgt_robot key (ROBOT_XML_DICT).")
  p.add_argument("--robot-xml", type=str, default=None, help="Robot MJCF path; overrides ROBOT_XML_DICT if the fork lacks it.")
  p.add_argument("--src-human", type=str, default="smplx", help="GMR src_human preset. 'smplx' for live ChingMu; fork has no 'xrobot'.")
  p.add_argument("--actual-human-height", type=float, default=1.6, help="Operator height (m); GMR scales to it.")
  p.add_argument("--target-fps", type=float, default=100.0)
  p.add_argument("--smooth", type=float, default=0.5, help="EMA alpha on retargeted joints [0,1). 0=off.")
  p.add_argument("--ramp-seconds", type=float, default=1.5, help="Engage ramp default->live on first detection.")
  p.add_argument("--no-joint-vel", action="store_true", help="Publish positions only; let the deploy finite-diff.")
  p.add_argument("--measure-fps", action="store_true")
  p.add_argument("--redis-host", type=str, default="localhost")
  p.add_argument("--redis-port", type=int, default=6379)
  p.add_argument("--redis-db", type=int, default=0)
  p.add_argument("--redis-key", type=str, default=_DEFAULT_KEY)
  return p


if __name__ == "__main__":
  run(_build_argparser().parse_args())
