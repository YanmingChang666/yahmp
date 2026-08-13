"""Dump a ChingMu skeleton's segment hierarchy (name / parent / sensor id).

Use this to discover the joint stream order when building the `--joint-order`
map for `run_yahmp_onnx_mocap.py` (see DEPLOYMENT.md §5.2). Self-contained: it
loads the vendored `third_party/chingmu/libCMVrpn.so` by default.

    uv run python -m yahmp.scripts.deploy.chingmu_hierarchy \
      --host MCAvatar@192.168.123.112

Each streamed segment prints once as:

    segment name:<name>  parent id:<n>  sensor id:<n>

List the segments in ascending `sensor id` order, keep the ones that map to G1
joints, and write their policy joint names into `config/chingmu_joint_order.json`.
"""

from __future__ import annotations

import argparse
import time
from ctypes import (
  CDLL,
  CFUNCTYPE,
  Structure,
  byref,
  c_char_p,
  c_int,
  c_long,
)
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_VENDORED_CHINGMU_DLL = _REPO_ROOT / "third_party" / "chingmu" / "libCMVrpn.so"


class _Timeval(Structure):
  _fields_ = [("tv_sec", c_long), ("tv_usec", c_long)]


class _VrpnHierarchy(Structure):
  _fields_ = [
    ("msg_time", _Timeval),
    ("sensor", c_int),
    ("parent", c_int),
    ("name", c_char * 127),
  ]


def run(dll_path: Path, host: str, seconds: float, encoding: str) -> None:
  if not dll_path.is_file():
    raise SystemExit(
      f"ChingMu DLL not found: {dll_path}. Pass --dll or vendor it in "
      "third_party/chingmu/libCMVrpn.so."
    )
  dll = CDLL(str(dll_path))
  host_b = bytes(host, encoding)

  seen: set[int] = set()

  def _on_hierarchy(_ptr, h) -> None:
    if h.sensor not in seen:
      seen.add(h.sensor)
      print(f"segment name:{h.name.decode(errors='replace')}  "
            f"parent id:{h.parent}  sensor id:{h.sensor}")

  cb = CFUNCTYPE(None, c_char_p, _VrpnHierarchy)(_on_hierarchy)
  user = _VrpnHierarchy(_Timeval(c_long(0), c_long(0)), c_int(0), c_int(0), b"0" * 127)

  dll.CMVrpnStartExtern()
  try:
    dll.CMVrpnEnableLog(False)
  except Exception:
    pass

  ret = dll.CMPluginRegisterUpdateHierarchy(host_b, byref(user), cb)
  t0 = time.perf_counter()
  while ret != 1 and time.perf_counter() - t0 < 5.0:
    time.sleep(0.2)
    ret = dll.CMPluginRegisterUpdateHierarchy(host_b, byref(user), cb)
  print(f"[chingmu] RegisterUpdateHierarchy -> {ret}  host={host}")

  print(f"[chingmu] listening {seconds:.0f}s (Ctrl-C to stop)…")
  try:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
      time.sleep(0.2)
  except KeyboardInterrupt:
    pass
  finally:
    try:
      dll.CMPluginUnRegisterUpdateHierarchy(host_b, byref(user), cb)
      dll.CMVrpnQuitExtern()
    except Exception:
      pass
  print(f"[chingmu] done — {len(seen)} unique segments seen.")


def main() -> None:
  p = argparse.ArgumentParser(description="Dump a ChingMu skeleton hierarchy.")
  p.add_argument("--host", required=True, help="e.g. MCAvatar@192.168.123.112")
  p.add_argument(
    "--dll",
    type=Path,
    default=_VENDORED_CHINGMU_DLL,
    help="Path to libCMVrpn.so (default: vendored third_party/chingmu/libCMVrpn.so).",
  )
  p.add_argument("--seconds", type=float, default=10.0)
  p.add_argument("--encoding", type=str, default="gbk")
  args = p.parse_args()
  run(Path(args.dll).expanduser(), str(args.host), float(args.seconds), str(args.encoding))


if __name__ == "__main__":
  main()
