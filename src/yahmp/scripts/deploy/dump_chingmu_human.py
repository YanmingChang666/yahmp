"""Dump the ChingMu GLOBAL human skeleton (CMHumanGlobalTLocalRTC) with names.

Why: the GMR retarget server currently reads the *tracker-data callback*
(callback_data.py), which streams LOCAL hierarchical bone offsets. Feeding those
to GMR as world positions collapses the robot (90° hips, lying). The correct API
for retargeting is `CMHumanGlobalTLocalRTC`, whose `humanT` is the GLOBAL
translation of every segment — exactly what GMR's position IK needs. This tool
dumps it (plus the segment names from the hierarchy stream) so we can build a
name→GMR-body map and verify the skeleton actually stands (spine above pelvis,
feet near the floor).

Needs only the vendored ChingMu DLL + ctypes (no gmr / mujoco / yahmp):

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


def main() -> None:
  ap = argparse.ArgumentParser(description="Dump ChingMu global human skeleton (CMHumanGlobalTLocalRTC).")
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

  # ── Hierarchy: collect {segment_id: (name, parent)} (best-effort) ───────────
  hier: dict[int, tuple[str, int]] = {}
  userdata = _VrpnHierarchy(_Timeval(0, 0), 0, 0, b"0" * 127)

  def _on_hier(_ptr, h) -> None:
    try:
      hier[int(h.sensor)] = (h.name.decode(errors="replace"), int(h.parent))
    except Exception:  # noqa: BLE001
      pass

  cb = CFUNCTYPE(None, c_char_p, _VrpnHierarchy)(_on_hier)
  dll.CMPluginConnectServer(host)
  for _ in range(50):
    if dll.CMPluginRegisterUpdateHierarchy(host, byref(userdata), cb) == 1:
      break
    time.sleep(0.1)

  # ── Poll global-T / local-R until we get one detected frame ────────────────
  human_t = (c_double * (MAX_SEGMENT_NUM * 3))()
  human_r = (c_double * (MAX_SEGMENT_NUM * 4))()
  detected = (c_int * MAX_SEGMENT_NUM)()
  timecode = (c_int * 1)()
  print(f"[dump] polling CMHumanGlobalTLocalRTC (human {args.human_id}) for {args.seconds:.0f}s — stand still…")
  last: tuple[list, list] | None = None
  t0 = time.time()
  while time.time() - t0 < args.seconds:
    ok = dll.CMHumanGlobalTLocalRTC(host, args.human_id, timecode, human_t, human_r, detected)
    if ok:
      last = (
        [float(human_t[k]) for k in range(MAX_SEGMENT_NUM * 3)],
        [int(detected[k]) for k in range(MAX_SEGMENT_NUM)],
      )
    time.sleep(0.05)
  try:
    dll.CMVrpnQuitExtern()
  except Exception:  # noqa: BLE001
    pass

  if last is None:
    print("[dump] human not detected — check --host / --human-id and that the actor is in view.")
    return

  tvals, det = last
  print(f"[dump] hierarchy segments received: {len(hier)}")
  print(f"\n  {'idx':>3} {'name':>18} {'parent':>6} {'X(m)':>8} {'Y(m)':>8} {'Z(m)':>8}")
  print("  " + "-" * 56)
  pts: list[tuple[float, float, float]] = []
  for i in range(MAX_SEGMENT_NUM):
    if not det[i]:
      continue
    x = tvals[i * 3] * args.pos_scale
    y = tvals[i * 3 + 1] * args.pos_scale
    z = tvals[i * 3 + 2] * args.pos_scale
    name, parent = hier.get(i, ("?", -1))
    print(f"  {i:>3} {name:>18} {parent:>6} {x:>8.3f} {y:>8.3f} {z:>8.3f}")
    pts.append((x, y, z))

  if pts:
    cols = list(zip(*pts))
    spreads = [max(c) - min(c) for c in cols]
    up = "XYZ"[spreads.index(max(spreads))]
    print(
      f"\n[dump] axis spreads  X={spreads[0]:.2f}  Y={spreads[1]:.2f}  Z={spreads[2]:.2f} m  → "
      f"'up' is likely {up} (largest spread when the actor stands upright)."
    )
    print("[dump] sanity: the top segment (head) should have the largest 'up' value, feet the smallest.")


if __name__ == "__main__":
  main()
