# Runtime config

## `chingmu_joint_order.json`

Maps the ChingMu mocap joint stream onto the policy's joint order, for
`run_yahmp_onnx_mocap.py --source chingmu --joint-order config/chingmu_joint_order.json`.

**Semantics.** The file is a JSON list of the policy's 29 joint names. The
**position** in the list = the ChingMu sensor index (`sensor_id -
--sensor-joint-first`); the **value** = which policy joint that sensor carries.
`LiveReference` reorders the incoming stream into `spec.joint_names` order using
this list.

> ⚠️ The shipped file is in **identity (policy) order** — a placeholder. It is
> only correct if your ChingMu retarget skeleton happens to emit joints in the
> exact G1 order below. **Verify and reorder it** to match your setup, or the
> policy will drive the wrong joints.

### How to build the correct order

1. Print the policy's joint order (the names that must all appear here):

   ```bash
   uv run python -c "import onnx; m=onnx.load('assets/models/g1_yahmp.onnx'); \
   print({e.key:e.value for e in m.metadata_props}['joint_names'])"
   ```

2. Dump your ChingMu skeleton's segment names + sensor ids (self-contained
   helper, uses the vendored `third_party/chingmu/libCMVrpn.so`):

   ```bash
   uv run python -m yahmp.scripts.deploy.chingmu_hierarchy --host MCAvatar@192.168.123.112
   ```

3. For each ChingMu sensor index in ascending order, write the matching policy
   joint name at that position in the list. Every name in step 1 must appear
   exactly once; the script errors if any is missing.

The policy order (G1 29-DOF: legs → waist → arms) is:
`left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch,
left_ankle_roll, right_hip_* (×6), waist_yaw, waist_roll, waist_pitch,
left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
left_wrist_roll, left_wrist_pitch, left_wrist_yaw, right_shoulder/elbow/wrist (×7)`
— all suffixed `_joint`.
