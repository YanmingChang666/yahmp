# Vendored ChingMu VRPN library

Third-party shared library for the ChingMu (青瞳) motion-capture VRPN interface,
used by the real-time teleoperation path
([`run_yahmp_onnx_mocap.py`](../../src/yahmp/scripts/deploy/run_yahmp_onnx_mocap.py),
DEPLOYMENT.md §5).

| File | Platform | Notes |
|---|---|---|
| `libCMVrpn.so` | Linux x86-64 | Loaded by default via `ctypes`. Links only against system libs (`libpthread`, `libz`, `libstdc++`, `libm`, `libc`). |
| `CMVrpn.dll` | Windows x64 | Counterpart for Windows deployment. |

These are proprietary ChingMu SDK binaries, redistributed here only for
convenience within this project. They are **not** covered by this repository's
Apache-2.0 license — their use is governed by the ChingMu SDK license. Obtain
updates from the official ChingMu SDK distribution.

The default lookup path is resolved relative to the repo root; override with
`--chingmu-dll` / `--dll` if you keep the library elsewhere.
