# YAHMP Deployment Guide

End-to-end guide for taking a trained YAHMP policy from a checkpoint to
simulation validation, and then to a **real Unitree G1** robot.

> **Scope note.** This repository ships **simulation deployment**
> (`export → ONNX → MuJoCo`) plus a **real-time mocap teleoperation** path
> (§5), but **no turnkey real-robot script**. The ONNX export bakes in every
> constant a hardware controller needs, and
> [`run_yahmp_onnx_mujoco.py`](src/yahmp/scripts/deploy/run_yahmp_onnx_mujoco.py)
> is the **reference inference loop** you port to the robot. Section 4 explains
> exactly how.

> **Already trained.** You don't need to train first — a ready-to-use policy
> ships in [`assets/models/g1_yahmp.onnx`](assets/models/g1_yahmp.onnx). See
> [`assets/models/README.md`](assets/models/README.md) and skip to §2.

---

## 0. Environment setup

The project is driven by [`uv`](https://docs.astral.sh/uv/).

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"          # current shell
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc   # make permanent
uv --version                                  # verify
```

> Prefer the installer over `snap install astral-uv` (the snap needs
> `--classic` confinement). The installer needs no `sudo`.

### Pin Python 3.12 and sync

`onnxruntime-gpu` only ships wheels for **Python 3.11 / 3.12 / 3.13** — a
system Python 3.10 will fail with *"doesn't have a source distribution or wheel
for the current platform"*. Let uv manage the interpreter:

```bash
uv python pin 3.12     # writes .python-version; uv downloads CPython 3.12
uv sync                # builds .venv from uv.lock (torch, mjlab, onnxruntime-gpu, ...)
```

First sync pulls the full stack (a few minutes, a couple GB). The GPU wheels
bundle their own CUDA libraries, so no system CUDA install is required to
import them.

### Smoke test

```bash
uv run play Mjlab-YAHMP-Unitree-G1 --agent zero
```

---

## 1. Export a checkpoint to ONNX

The ONNX file is the **single portable artifact** for deployment: it contains
the policy network **and** all deployment metadata (gains, joint names, action
scaling, observation layout, control rate).

From a local checkpoint:

```bash
uv run python -m yahmp.scripts.deploy.export_checkpoint_to_onnx \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --output-path assets/models/g1_yahmp.onnx
```

From a Weights & Biases run:

```bash
uv run python -m yahmp.scripts.deploy.export_checkpoint_to_onnx \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --wandb-run-path entity/project/run_id
```

Useful flags: `--wandb-checkpoint-name model_7000.pt`, `--device cpu|cuda`,
`--num-envs 1`.

---

## 2. Validate in simulation (MuJoCo)

**Always** confirm the exported ONNX tracks correctly in sim before touching
hardware:

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mujoco \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --motion-file assets/motions/g1_omomo_amass_clean/<motion-name>.npz
```

Flags:

| Flag | Purpose |
|---|---|
| `--ort-provider auto\|cpu\|cuda` | ONNX Runtime execution provider |
| `--init-default-joints` | Start from the default pose instead of the first reference frame |
| `--motion-file` / `--motion-npz` | Reference clip (YAHMP `.npz` or TWIST2 `.pkl`) |

> To drive the policy from a **live human** instead of a recorded clip, see the
> real-time mocap teleoperation path in §5.

This runner is also your **specification** for the hardware loop — read it
alongside Section 4.

---

## 3. The ONNX metadata contract

`attach_onnx_metadata` in
[`src/yahmp/rl/exporter.py`](src/yahmp/rl/exporter.py) embeds these keys into
the ONNX file. A hardware controller reads them (see `PolicySpec.from_onnx` in
the MuJoCo runner) instead of hard-coding anything:

| Metadata key | Meaning | Hardware use |
|---|---|---|
| `physics_dt`, `control_dt` | Sim step / policy tick period | Policy runs every `control_dt` seconds |
| `joint_names` | Ordered joints the policy observes | Defines observation/action ordering |
| `joint_stiffness` | Per-joint **Kp** | PD gain sent to motors |
| `joint_damping` | Per-joint **Kd** | PD gain sent to motors |
| `default_joint_pos` | Default/home pose | Observation offset + move-to-default target |
| `action_semantics` | `residual_joint_position` or `joint_position` | How to turn actions into targets |
| `action_scale`, `action_offset` | Action post-processing | `target = raw*scale + offset` |
| `action_target_names` | Joints the action controls | Map action → motors |
| `observation_terms_layout` | Ordered obs terms + history lengths | Build the observation vector |
| `motion_command_class`, `motion_command_dim`, `motion_command_step_offsets` | Reference command spec | Build the `command` term |
| `root_body_name`, `body_names` | Anchor/tracked bodies | Reference frame construction |

> Gains come **from training** — deploy with the same Kp/Kd the policy was
> trained with, or the plant no longer matches and the robot will likely fall.

---

## 4. Deploy to a real Unitree G1

> **A reference implementation now ships:**
> [`run_yahmp_onnx_real.py`](src/yahmp/scripts/deploy/run_yahmp_onnx_real.py)
> (§5.4) already does everything below — Unitree SDK I/O, gains from metadata,
> safe bring-up, mocap-driven command. This section explains **how it works** so
> you can adapt it; jump to §5.4 for the ready-to-run command.

You build a hardware runtime by mirroring
[`run_yahmp_onnx_mujoco.py`](src/yahmp/scripts/deploy/run_yahmp_onnx_mujoco.py)
and swapping MuJoCo I/O for the Unitree SDK
([`unitree_sdk2_python`](https://github.com/unitreerobotics/unitree_sdk2_python),
DDS `LowState`/`LowCmd`). Reuse `PolicySpec`, the observation builder, and the
action decoder **verbatim** — only the read/write endpoints change.

### 4.1 Control architecture

```
  Reference motion (.npz)          Robot sensors (IMU + encoders)
            │                                │
            ▼                                ▼
     build `command`  ┌──────────────► build observation
                      │                      │
                      │                      ▼
                      │              ONNX policy inference   (every control_dt, e.g. 50 Hz)
                      │                      │
                      │                      ▼
                      │        action → joint position targets
                      │                      │
                      └──────────────────────┼─► LowCmd{ q=target, Kp, Kd, tau=0 }
                                             ▼
                              Motor-side PD loop  (~1 kHz, on the robot)
```

Your policy emits **position targets** at `control_dt`; the robot's motor
controllers run the fast PD loop using `joint_stiffness` / `joint_damping`.

### 4.2 Building the observation

The deployable observation block is
[`_current_observation_block`](src/yahmp/scripts/deploy/run_yahmp_onnx_mujoco.py)
(plus its history buffer). Map each term to a real sensor:

| Obs term | Source on real G1 |
|---|---|
| `command` | Sampled from the retargeted `.npz` clip (see 4.4). **From the clip, not the robot.** |
| `base_ang_vel` | IMU gyroscope, expressed in the base/body frame |
| `projected_gravity` | Gravity direction rotated into the body frame via IMU orientation |
| `joint_pos` | Motor encoder positions **minus** `default_joint_pos` |
| `joint_vel` | Motor encoder velocities |
| `actions` | The previous policy output |
| `history` | Ring buffer of the block above (`history_length` from metadata) |

> ⚠️ **Check `observation_terms_layout` in your exported ONNX.** The MuJoCo
> runner also computes `base_lin_vel`, but the deployable (base/student)
> policies **do not** include it — base linear velocity is privileged (teacher
> only) and hard to estimate on hardware. If `base_lin_vel` appears in your
> layout, that checkpoint is **not** directly deployable without a state
> estimator.

### 4.3 Decoding the action

From [`_apply_action`](src/yahmp/scripts/deploy/run_yahmp_onnx_mujoco.py):

```python
processed = raw_action * action_scale + action_offset
if action_semantics == "residual_joint_position":
    target = reference_joint_pos[action_target_indices] + processed
else:  # "joint_position"
    target = processed
# send to motors:
LowCmd[j] = { q = target[j], dq = 0, Kp = joint_stiffness[j],
              Kd = joint_damping[j], tau = 0 }
```

### 4.4 Building the `command` (reference) term

From [`_single_command_step_value`](src/yahmp/scripts/deploy/run_yahmp_onnx_mujoco.py),
per reference frame (per-step dim = `2 * num_joints + 6`):

```
[ joint_pos_ref,                 # num_joints
  joint_vel_ref,                 # num_joints
  anchor_lin_vel_body_xy,        # 2   (planar velocity in body frame)
  anchor_ang_vel_body_yaw,       # 1   (yaw rate)
  root_height_z,                 # 1
  root_roll, root_pitch ]        # 2
```

For the **Future** variant (`Mjlab-YAHMP-Future-Unitree-G1`), concatenate one
block per offset in `motion_command_step_offsets`, sampling the clip at
`time + offset * control_dt`.

### 4.5 Hardware loop skeleton

```python
spec = PolicySpec.from_onnx(onnx_path)          # reuse from the MuJoCo runner
clip = MotionClip(motion_file, spec.root_body_name)
session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

move_to_default(spec.default_joint_pos)         # slow, safe ramp — YOU implement
history = init_history(spec)
prev_action = zeros(spec.action_dim)
t = 0.0

while running:
    tic = time.perf_counter()

    low_state   = robot.read_low_state()        # Unitree SDK: encoders + IMU
    frame       = clip.sample(t)                # reference at time t
    obs         = build_observation(spec, low_state, frame, prev_action, history)

    raw_action  = session.run(None, {input_name: obs[None]})[0].reshape(-1)
    target      = decode_action(spec, raw_action, frame)   # 4.3

    robot.send_low_cmd(q=target,                # Unitree SDK: LowCmd
                       Kp=spec.joint_stiffness,
                       Kd=spec.joint_damping, tau=0)

    prev_action = raw_action
    t += spec.control_dt
    append_history(spec, history, next_terms)
    sleep_until(tic + spec.control_dt)          # hold control_dt cadence
```

### 4.6 Joint-order mapping (critical)

The order in `spec.joint_names` is the **mjlab** order, which is **not** the
Unitree SDK motor-index order. Build an explicit bidirectional index map and
unit-test it. **A wrong mapping is the most dangerous bug** — it silently
commands the wrong motors.

---

## 5. Real-time mocap teleoperation (sim2sim)

Beyond replaying a recorded `.npz` clip (§2), you can drive the policy's motion
**command** live from a motion-capture stream. The script
[`run_yahmp_onnx_mocap.py`](src/yahmp/scripts/deploy/run_yahmp_onnx_mocap.py)
puts the **policy in the loop**: a simulated G1 tracks the human motion the
mocap system retargets in real time (unlike a pure kinematic replay, which just
copies poses onto the model).

```
  robot state (MuJoCo sim) ─┐
                            ├─► observation ─► ONNX ─► action ─► sim
  command (mocap, LIVE) ────┘
```

It **reuses** the observation/action pipeline from `run_yahmp_onnx_mujoco.py`
unchanged. The only additions are a live pose source and a `LiveReference` that
turns the latest streamed pose into the `MotionFrame` the command builder
expects, finite-differencing joint/root velocities.

### 5.1 Device-free test (no hardware)

Validate the whole live-command path by streaming a clip *as if* it were a
device — this needs no mocap gear:

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --source npz \
  --npz-clip assets/motions/g1_omomo_amass_clean/<motion>.npz
```

If the simulated G1 tracks here, the plumbing is correct and only device wiring
remains.

### 5.2 Live ChingMu teleoperation

The retargeted **robot skeleton** is streamed on a **dedicated VRPN port** (e.g.
`:3884`), typically with the **root at sensor 301** and **joints from sensor
302** — distinct from the ball/marker stream on the default port. The values
below are from a working G1 setup; confirm yours with the hierarchy dump.

**Step 1 — kinematic replay (validate the remap, no policy).** This writes the
mocap pose straight to the G1's `qpos` — the model should mirror the human
limb-for-limb. Fix joint order / frame here *before* closing the policy loop:

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
  --task-id Mjlab-YAHMP-Unitree-G1 --onnx-path assets/models/g1_yahmp.onnx \
  --source chingmu --mode replay \
  --chingmu-host MCAvatar@192.168.123.112:3884 \
  --sensor-root 301 --sensor-joint-first 302 \
  --joint-order config/chingmu_joint_order.json
```

**Step 2 — close the policy loop (teleoperation).** Once replay looks right,
swap to `--mode teleop`: a simulated G1 now *tracks* the human via the policy:

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
  --task-id Mjlab-YAHMP-Unitree-G1 --onnx-path assets/models/g1_yahmp.onnx \
  --source chingmu --mode teleop \
  --chingmu-host MCAvatar@192.168.123.112:3884 \
  --sensor-root 301 --sensor-joint-first 302 \
  --joint-order config/chingmu_joint_order.json \
  --vel-smoothing 0.7
```

The ChingMu VRPN library is **vendored** at
[`third_party/chingmu/libCMVrpn.so`](third_party/chingmu/) and loaded by default
— no absolute path needed. Override with `--chingmu-dll` only if you keep it
elsewhere.

The retargeting (human → G1 skeleton) is done **upstream in the ChingMu
software**; the VRPN stream delivers a retargeted root pose + per-joint angles,
which the source reads (same `[qx,qy,qz,qw]→[qw,qx,qy,qz]` and mm→m conventions
as a standard ChingMu VRPN client). Two configuration steps are **mandatory**:

| Step | Why | How |
|---|---|---|
| **Joint-order map** | Mocap streams joints in its skeleton order; the policy expects `spec.joint_names` order. A wrong map silently drives the wrong joints. | Dump the policy order + your ChingMu segment order (both below), write a JSON list in stream order, pass `--joint-order`. |
| **Frame calibration** | The command uses root height, roll/pitch and body-frame velocities; a mocap/world frame mismatch makes the policy track garbage. | Apply an offset/rotation to `root_pos` / `root_quat` in `ChingMuMocapSource._on_tracker`. |

Print the policy's joint order:

```bash
uv run python -c "import onnx; m=onnx.load('assets/models/g1_yahmp.onnx'); print({e.key:e.value for e in m.metadata_props}['joint_names'])"
```

Dump your ChingMu skeleton's segment/sensor order (self-contained helper, uses
the vendored library):

```bash
uv run python -m yahmp.scripts.deploy.chingmu_hierarchy --host MCAvatar@192.168.123.112
```

A starter [`config/chingmu_joint_order.json`](config/chingmu_joint_order.json)
ships in identity order — reorder it to match the dump (see
[`config/README.md`](config/README.md)).

### 5.3 Caveats

- **Use the base policy.** `Mjlab-YAHMP-Future-Unitree-G1` needs *future*
  reference frames, which don't exist in a live stream (the script approximates
  them as the current pose). Prefer `Mjlab-YAHMP-Unitree-G1` for teleop.
- **Velocity noise.** Finite-differenced mocap velocities are jittery — tune
  `--vel-smoothing 0.5–0.8`, or feed a real velocity stream into
  `MocapState.update(joint_vel=...)`.
- **From sim2sim to sim2real.** The mocap/command half is identical on hardware
  — `run_yahmp_onnx_real.py` (§5.4) does exactly this. Complete the §6 safety
  checklist first.

### 5.4 Sim2real — teleoperate the physical G1

[`run_yahmp_onnx_real.py`](src/yahmp/scripts/deploy/run_yahmp_onnx_real.py)
closes the loop on real hardware: it reads the G1's joint encoders + pelvis IMU
over the Unitree SDK, builds the **same** YAHMP observation as the sim, runs the
ONNX policy, and sends joint-position targets with the trained Kp/Kd. It mirrors
the proven Beyondmimic `deploy_real4bydmimic.py` bring-up sequence and reuses
YAHMP's obs/action code verbatim for sim/real parity.

> ⚠️ **Physical robot.** Do the §6 safety checklist first. This script has **not
> been hardware-tested** — validate on a hoist, with small gains and `--dry-run`.

**Prerequisite** (not a YAHMP dependency — install separately):

```bash
uv pip install unitree_sdk2py
```

**Bring-up sequence** (operator-gated via the wireless remote):

1. Zero-torque — hoist the robot, press **START**.
2. 2 s ramp to `default_joint_pos`.
3. Hold default — press **A** to begin policy tracking.
4. Track the mocap; press **SELECT** to stop (→ damping state).

**Recommended first run — small gains + dry-run.** `--dry-run` runs the whole
loop (state → obs → policy → targets) but keeps the motors **limp** (`kp=kd=0`),
so you validate the pipeline without energizing:

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_real \
  --onnx-path assets/models/g1_yahmp.onnx --net enp4s0 \
  --source chingmu --chingmu-host MCAvatar@192.168.123.112:3884 \
  --sensor-root 301 --sensor-joint-first 302 \
  --joint-order config/chingmu_joint_order.json \
  --kp-scale 0.25 --kd-scale 0.5 --dry-run
```

Then drop `--dry-run` and raise `--kp-scale` / `--kd-scale` toward `1.0` as you
gain confidence. `--net` is your DDS interface (`ip a` to find it).

| Point | Detail |
|---|---|
| **Gains from the policy** | Kp/Kd are read from the ONNX `joint_stiffness` / `joint_damping`; `--kp-scale` / `--kd-scale` scale them for a gentle start (your *small Kp/Kd to prevent excessive motion*). |
| **No joint remap** | G1 motor order **equals** YAHMP's `joint_names` (legs→waist→arms); override with `--joint2motor` only if your robot differs. |
| **IMU = pelvis** | YAHMP's root is the pelvis, so the pelvis IMU is used directly (no torso transform). If your IMU is on the torso, transform it to the pelvis frame first. |
| **Validate in sim first** | Same ONNX with `--source npz` (§5.1) and the kinematic replay (§5.2) before ever energizing. |

---

## 6. Safety checklist (real robot)

- [ ] **Hoist / gantry.** First runs with the robot suspended, feet off the ground.
- [ ] **Kill switch** wired and tested before enabling the policy.
- [ ] **Move-to-default** with a slow ramp to `default_joint_pos` before starting inference — never snap to it.
- [ ] **Gains match training.** Use `joint_stiffness` / `joint_damping` from the ONNX metadata unchanged.
- [ ] **Joint-order map** verified with a unit test.
- [ ] **Clamps** on joint position/velocity limits and torque before sending `LowCmd`.
- [ ] **Loop timing** holds `control_dt`; detect and handle missed deadlines / stale `LowState`.
- [ ] **Sim first.** Same ONNX + same clip validated in MuJoCo (Section 2).
- [ ] Start with **short, calm clips**; escalate to dynamic motions only after they're stable.

> **Sim-to-real is partly trained-in.** The config uses `DelayedActuatorCfg`
> (0–4 step position delay) plus mass / CoM / friction / push randomization,
> which is what makes transfer feasible. It is **not** a guarantee — expect
> tuning and never skip the hoist.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Command 'uv' not found` | Install uv (Section 0); `export PATH="$HOME/.local/bin:$PATH"`. |
| `source $HOME/.local/bin/env: No such file` | That file isn't created by newer uv; just add `~/.local/bin` to `PATH` directly. |
| `onnxruntime-gpu ... only has wheels for cp311/cp312/cp313` | System Python is 3.10. Run `uv python pin 3.12 && uv sync`. |
| `uv sync` slow / huge | Expected on first run (torch + CUDA libs). Cached afterwards. |
| CUDA/cuDNN load error at inference | ONNX Runtime CUDA/cuDNN mismatch; try `--ort-provider cpu` to isolate, then align CUDA versions. |
| `Missing ONNX metadata key ...` | Re-export with the current exporter (Section 1). |
| MuJoCo `physics_dt`/`control_dt` mismatch error | The ONNX was exported from a different task config than `--task-id`; re-export or pass the matching task. |
| Teleop: `No mocap frame detected within timeout` | Check `--chingmu-host` / `--sensor-root`; confirm the ChingMu VRPN stream is up (test with your `get_hierarchy.py`). |
| Teleop: `Joints missing from --joint-order` | Your `--joint-order` JSON is missing a name from the policy's `joint_names`; print the policy order (§5.2) and align them. |
| Teleop: robot tracks a mirrored / drifting pose | Frame calibration (§5.2) — fix the root offset/rotation in `ChingMuMocapSource._on_tracker`. |
| Teleop: robot jitters violently | Raise `--vel-smoothing` (e.g. `0.7`), or use the base (not Future) policy. |
| Real: `ModuleNotFoundError: unitree_sdk2py` | Install it into the env: `uv pip install unitree_sdk2py`. |
| Real: hangs at "waiting for LowState" | Wrong `--net` interface or robot not on the DDS network; `ip a` to find the interface, check the cable/`rt/lowstate` topic. |
| Real: robot sags / can't hold pose | `--kp-scale` too low; raise it toward `1.0`. Too-weak Kp can't counter gravity. |
| Real: robot overshoots / oscillates | `--kp-scale` too high or `--kd-scale` too low; lower Kp / raise Kd. Start at `0.25` / `0.5`. |
| Real: limbs move but wrong ones | Joint mapping — verify with the kinematic replay (§5.2); set `--joint2motor` if the motor order differs. |

---

## Command quick reference

```bash
# Setup
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python pin 3.12
uv sync

# Export
uv run python -m yahmp.scripts.deploy.export_checkpoint_to_onnx \
  --task-id Mjlab-YAHMP-Unitree-G1 --checkpoint-file model.pt \
  --output-path assets/models/g1_yahmp.onnx

# Validate in sim
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mujoco \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --motion-file assets/motions/g1_omomo_amass_clean/<motion>.npz

# Mocap teleoperation, device-free test (sim2sim)
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --source npz --npz-clip assets/motions/g1_omomo_amass_clean/<motion>.npz

# Real robot (sim2real): small gains + dry-run first — SEE §5.4 / §6 SAFETY
uv pip install unitree_sdk2py
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_real \
  --onnx-path assets/models/g1_yahmp.onnx --net enp4s0 \
  --source chingmu --chingmu-host MCAvatar@192.168.123.112:3884 \
  --sensor-root 301 --sensor-joint-first 302 \
  --joint-order config/chingmu_joint_order.json \
  --kp-scale 0.25 --kd-scale 0.5 --dry-run
```
