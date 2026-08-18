"""Dump the ChingMu human skeleton and reconstruct GLOBAL joint positions via FK.

Why: every ChingMu *human* API (CMHumanExtern / CMHumanGlobalTLocalRTC /
CMRetargetHumanExternTC) returns the ROOT global pose + each segment's LOCAL
offset & LOCAL rotation — NOT global joint positions. The GMR retarget server
mistakenly feeds those local offsets to GMR as world positions, so GMR fits the
robot to a folded ~30 cm point cloud → the robot lies down (90° hips). GMR needs
GLOBAL joint positions, which only exist after forward kinematics.

This tool captures root + local offsets (humanT) + local rotations (humanLocalR)
+ the parent hierarchy, runs FK, and prints both the raw per-segment data and the
reconstructed world positions. If FK is right the head sits ~1.7 m up and the
feet near 0; if not, paste the raw table and the convention gets fixed offline.

Needs only the vendored ChingMu DLL + ctypes + numpy (no gmr / mujoco / yahmp):

    python src/yahmp/scripts/deploy/dump_chingmu_human.py \
      --host MCAvatar@192.168.123.112 \
      --dll third_party/chingmu/libCMVrpn.so --seconds 5
"""

from __future__ import annotations

import argparse
import time
from ctypes import (
  CDLL,
  CFUNCTYPE,
  Structure,
  byref,
  c_char,
  c_char_p,
  c_double,
  c_int,
  c_long,
)

import numpy as np

MAX_SEGMENT_NUM = 150


class _Timeval(Structure):
  _fields_ = [("tv_sec", c_long), ("tv_usec", c_long)]


class _VrpnHierarchy(Structure):
  _fields_ = [
    ("msg_time", _Timeval),
    ("sensor", c_int),
    ("parent", c_int),
    ("name", c_char * 127),
  ]


# ── quaternion helpers (xyzw, matching scipy / ChingMu order) ────────────────
def _qmul(a, b):
  ax, ay, az, aw = a
  bx, by, bz, bw = b
  return np.array([
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ])


def _qrot(q, v):
  x, y, z, w = q
  t = 2.0 * np.cross([x, y, z], v)
  return np.asarray(v) + w * t + np.cross([x, y, z], t)


def _forward_kinematics(seg, pos_scale):
  """seg: {idx: dict(parent, offset(3,mm), quat(4,xyzw))} -> {idx: world_pos_m}."""
  gpos: dict[int, np.ndarray] = {}
  grot: dict[int, np.ndarray] = {}
  pending = set(seg)
  # roots first: parent not among segments
  progressed = True
  while pending and progressed:
    progressed = False
    for i in list(pending):
      p = seg[i]["parent"]
      if p not in seg:  # root
        grot[i] = seg[i]["quat"]
        gpos[i] = np.asarray(seg[i]["offset"], float) * pos_scale
        pending.discard(i)
        progressed = True
      elif p in gpos:  # parent already resolved
        grot[i] = _qmul(grot[p], seg[i]["quat"])
        gpos[i] = gpos[p] + _qrot(grot[p], np.asarray(seg[i]["offset"], float) * pos_scale)
        pending.discard(i)
        progressed = True
  return gpos


def main() -> None:
  ap = argparse.ArgumentParser(description="Dump ChingMu human skeleton + FK to global positions.")
  ap.add_argument("--host", required=True, help="e.g. MCAvatar@192.168.123.112")
  ap.add_argument("--dll", required=True, help="Path to libCMVrpn.so.")
  ap.add_argument("--human-id", type=int, default=0)
  ap.add_argument("--seconds", type=float, default=5.0)
  ap.add_argument("--pos-scale", type=float, default=0.001, help="mm -> m.")
  args = ap.parse_args()

  dll = CDLL(args.dll)
  host = bytes(args.host, "gbk")
  dll.CMVrpnStartExtern()
  try:
    dll.CMVrpnEnableLog(False)
  except Exception:  # noqa: BLE001
    pass
  dll.CMPluginConnectServer(host)

  # ── Hierarchy: {segment_id: (name, parent_id)} ─────────────────────────────
  hier: dict[int, tuple[str, int]] = {}
  userdata = _VrpnHierarchy(_Timeval(0, 0), 0, 0, b"0" * 127)

  def _on_hier(_ptr, h) -> None:
    try:
      hier[int(h.sensor)] = (h.name.decode(errors="replace").strip("\x00"), int(h.parent))
    except Exception:  # noqa: BLE001
      pass

  cb = CFUNCTYPE(None, c_char_p, _VrpnHierarchy)(_on_hier)
  for _ in range(80):
    if dll.CMPluginRegisterUpdateHierarchy(host, byref(userdata), cb) == 1:
      break
    time.sleep(0.1)

  # ── Poll one detected frame of global-T / local-R ──────────────────────────
  human_t = (c_double * (MAX_SEGMENT_NUM * 3))()
  human_r = (c_double * (MAX_SEGMENT_NUM * 4))()
  det = (c_int * MAX_SEGMENT_NUM)()
  tc = (c_int * 1)()
  print(f"[dump] polling {args.seconds:.0f}s (human {args.human_id}) — stand still, arms slightly out…")
  last = None
  t0 = time.time()
  while time.time() - t0 < args.seconds:
    if dll.CMHumanGlobalTLocalRTC(host, args.human_id, tc, human_t, human_r, det):
      last = (
        [float(human_t[k]) for k in range(MAX_SEGMENT_NUM * 3)],
        [float(human_r[k]) for k in range(MAX_SEGMENT_NUM * 4)],
        [int(det[k]) for k in range(MAX_SEGMENT_NUM)],
      )
    time.sleep(0.05)
  try:
    dll.CMVrpnQuitExtern()
  except Exception:  # noqa: BLE001
    pass
  if last is None:
    print("[dump] human not detected — check --host / --human-id / actor in view.")
    return
  tvals, rvals, detected = last

  # index i (humanT) <-> hierarchy id: infer the offset (min hier key, usually 300)
  base = min(hier) if hier else 0
  print(f"[dump] hierarchy: {len(hier)} segments; id = idx + {base}\n")
  seg: dict[int, dict] = {}
  print(f"  {'idx':>3} {'id':>4} {'name':>16} {'par':>4} "
        f"{'off_x':>7} {'off_y':>7} {'off_z':>7}  {'qx':>6} {'qy':>6} {'qz':>6} {'qw':>6}")
  print("  " + "-" * 84)
  for i in range(MAX_SEGMENT_NUM):
    if not detected[i]:
      continue
    off = np.array(tvals[i * 3:i * 3 + 3], float)
    quat = np.array(rvals[i * 4:i * 4 + 4], float)
    name, parent_id = hier.get(i + base, ("?", -1))
    parent_idx = parent_id - base if parent_id >= base else -1
    seg[i] = dict(parent=parent_idx, offset=off, quat=quat, name=name, id=i + base)
    print(f"  {i:>3} {i + base:>4} {name:>16} {parent_id:>4} "
          f"{off[0]:>7.1f} {off[1]:>7.1f} {off[2]:>7.1f}  "
          f"{quat[0]:>6.3f} {quat[1]:>6.3f} {quat[2]:>6.3f} {quat[3]:>6.3f}")

  # ── FK → world positions, and a standing sanity check ──────────────────────
  gpos = _forward_kinematics(seg, args.pos_scale)
  print(f"\n  {'idx':>3} {'name':>16} {'world_X':>8} {'world_Y':>8} {'world_Z':>8}")
  print("  " + "-" * 48)
  zs = []
  for i in sorted(gpos):
    p = gpos[i]
    zs.append((i, p))
    print(f"  {i:>3} {seg[i]['name']:>16} {p[0]:>8.3f} {p[1]:>8.3f} {p[2]:>8.3f}")
  if zs:
    allp = np.array([p for _, p in zs])
    spread = allp.max(0) - allp.min(0)
    up = "XYZ"[int(np.argmax(spread))]
    top = zs[int(np.argmax(allp[:, int(np.argmax(spread))]))]
    bot = zs[int(np.argmin(allp[:, int(np.argmax(spread))]))]
    print(f"\n[FK] world spread  X={spread[0]:.2f} Y={spread[1]:.2f} Z={spread[2]:.2f} m  → up≈{up}")
    print(f"[FK] top segment idx {top[0]} ({seg[top[0]]['name']})  bottom idx {bot[0]} ({seg[bot[0]]['name']})")
    print("[FK] GOOD if spread ~1.6-1.8 m, top=head, bottom=foot. If tiny/scrambled, paste the raw table above.")


if __name__ == "__main__":
  main()
