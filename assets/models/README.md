# Pre-trained Models

Ready-to-use ONNX policies. You do **not** need to train to run these — download
one motion clip (see [`../motions/README.md`](../motions/README.md)) and go.

| File | Policy | Use it for |
|---|---|---|
| `g1_yahmp.onnx` | The trained **YAHMP** G1 motion-tracking policy | Sim playback, mocap teleop, and real-robot deployment |
| `twist2_1017_25k.onnx` | The original **TWIST2** reference policy | Reproducibility / comparison experiments only |

Each ONNX carries all deployment metadata (PD gains, joint names, action
scaling, observation layout, control rate) — see
[`../../DEPLOYMENT.md`](../../DEPLOYMENT.md) §3. Inspect it with:

```bash
uv run python -c "import onnx; m=onnx.load('assets/models/g1_yahmp.onnx'); \
print('\n'.join(f'{e.key}: {e.value[:80]}' for e in m.metadata_props))"
```

## Run the pre-trained YAHMP policy

Track a recorded clip in MuJoCo (see [`../../DEPLOYMENT.md`](../../DEPLOYMENT.md) §2):

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mujoco \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --motion-file assets/motions/g1_omomo_amass_clean/<motion>.npz
```

Real-time mocap teleoperation, device-free (see [`../../DEPLOYMENT.md`](../../DEPLOYMENT.md) §5):

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --source npz --npz-clip assets/motions/g1_omomo_amass_clean/<motion>.npz
```

## Add your own

Export a checkpoint here with the exporter ([`../../DEPLOYMENT.md`](../../DEPLOYMENT.md) §1):

```bash
uv run python -m yahmp.scripts.deploy.export_checkpoint_to_onnx \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --output-path assets/models/<your-model>.onnx
```

> Re-export with the current exporter if a script complains about a
> `Missing ONNX metadata key` — older ONNX files may predate some metadata keys.
