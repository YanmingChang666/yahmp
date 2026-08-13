# YAHMP 部署教程（中文）

本教程完整介绍如何把训练好的 YAHMP 策略从 checkpoint 导出、在仿真中验证，
再部署到**真实的 Unitree G1** 机器人上。

> **适用范围说明。** 本仓库提供**仿真部署**（`导出 → ONNX → MuJoCo`）以及
> **实时动捕遥操作**路径（第 5 节），但**没有开箱即用的真机脚本**。ONNX 导出时会
> 把真机控制器需要的所有常量都写入元数据，而
> [`run_yahmp_onnx_mujoco.py`](src/yahmp/scripts/deploy/run_yahmp_onnx_mujoco.py)
> 就是你要移植到真机上的**参考推理循环**。第 4 节会讲清楚具体怎么做。

> **已有预训练策略。** 你无需先训练——仓库自带可直接使用的策略
> [`assets/models/g1_yahmp.onnx`](assets/models/g1_yahmp.onnx)。详见
> [`assets/models/README.md`](assets/models/README.md)，可直接跳到第 2 节。

---

## 0. 环境搭建

本项目基于 [`uv`](https://docs.astral.sh/uv/) 管理。

### 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"          # 仅当前终端生效
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc   # 永久生效
uv --version                                  # 验证
```

> 推荐用官方安装脚本，不要用 `snap install astral-uv`（snap 需要 `--classic`
> 特权模式）。官方脚本无需 `sudo`。

### 指定 Python 3.12 并同步依赖

`onnxruntime-gpu` 只提供 **Python 3.11 / 3.12 / 3.13** 的 wheel，系统自带的
Python 3.10 会报错 *"doesn't have a source distribution or wheel for the
current platform"*。让 uv 自己管理解释器即可：

```bash
uv python pin 3.12     # 写入 .python-version，uv 会自动下载 CPython 3.12
uv sync                # 根据 uv.lock 构建 .venv（torch、mjlab、onnxruntime-gpu 等）
```

首次 `uv sync` 会拉取完整依赖栈（几分钟，约几 GB）。GPU 版 wheel 自带 CUDA
运行库，因此**无需**系统预装 CUDA 就能导入。

### 冒烟测试

```bash
uv run play Mjlab-YAHMP-Unitree-G1 --agent zero
```

---

## 1. 把 checkpoint 导出为 ONNX

ONNX 文件是部署时**唯一的可移植产物**：它同时包含策略网络**和**全部部署元数据
（PD 增益、关节名、动作缩放、观测布局、控制频率）。

从本地 checkpoint 导出：

```bash
uv run python -m yahmp.scripts.deploy.export_checkpoint_to_onnx \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --output-path assets/models/g1_yahmp.onnx
```

从 Weights & Biases run 导出：

```bash
uv run python -m yahmp.scripts.deploy.export_checkpoint_to_onnx \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --wandb-run-path entity/project/run_id
```

常用参数：`--wandb-checkpoint-name model_7000.pt`、`--device cpu|cuda`、
`--num-envs 1`。

---

## 2. 在仿真（MuJoCo）中验证

上真机之前**务必**先确认导出的 ONNX 在仿真里能正确跟踪动作：

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mujoco \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --motion-file assets/motions/g1_omomo_amass_clean/<motion-name>.npz
```

参数说明：

| 参数 | 作用 |
|---|---|
| `--ort-provider auto\|cpu\|cuda` | ONNX Runtime 执行后端 |
| `--init-default-joints` | 从默认姿态而非参考动作首帧初始化 |
| `--motion-file` / `--motion-npz` | 参考动作片段（YAHMP `.npz` 或 TWIST2 `.pkl`） |

这个脚本同时也是真机循环的**规范参考**，请结合第 4 节一起阅读。

> 若想用**真人实时动作**代替录制片段来驱动策略，见第 5 节的实时动捕遥操作。

---

## 3. ONNX 元数据约定

[`src/yahmp/rl/exporter.py`](src/yahmp/rl/exporter.py) 中的
`attach_onnx_metadata` 会把下列键写入 ONNX 文件。真机控制器直接读取这些元数据
（参见 MuJoCo 脚本里的 `PolicySpec.from_onnx`），而不要硬编码任何常量：

| 元数据键 | 含义 | 真机用途 |
|---|---|---|
| `physics_dt`、`control_dt` | 仿真步长 / 策略推理周期 | 策略每隔 `control_dt` 秒运行一次 |
| `joint_names` | 策略观测的关节顺序 | 决定观测/动作的排列顺序 |
| `joint_stiffness` | 每个关节的 **Kp** | 下发给电机的 PD 增益 |
| `joint_damping` | 每个关节的 **Kd** | 下发给电机的 PD 增益 |
| `default_joint_pos` | 默认/初始姿态 | 观测偏置 + 归位目标 |
| `action_semantics` | `residual_joint_position` 或 `joint_position` | 决定如何把动作转成目标位置 |
| `action_scale`、`action_offset` | 动作后处理 | `target = raw*scale + offset` |
| `action_target_names` | 动作控制的关节 | 动作 → 电机映射 |
| `observation_terms_layout` | 观测项顺序 + 历史长度 | 构建观测向量 |
| `motion_command_class`、`motion_command_dim`、`motion_command_step_offsets` | 参考指令规格 | 构建 `command` 观测项 |
| `root_body_name`、`body_names` | 锚点/被跟踪刚体 | 构建参考坐标系 |

> 增益来自**训练**——部署时必须用与训练相同的 Kp/Kd，否则被控对象特性不匹配，
> 机器人极可能摔倒。

---

## 4. 部署到真实 Unitree G1

仓库不提供真机脚本，需要你自己写：以
[`run_yahmp_onnx_mujoco.py`](src/yahmp/scripts/deploy/run_yahmp_onnx_mujoco.py)
为模板，把 MuJoCo 的读写替换为 Unitree SDK
（[`unitree_sdk2_python`](https://github.com/unitreerobotics/unitree_sdk2_python)，
DDS 的 `LowState`/`LowCmd`）。`PolicySpec`、观测构建、动作解码这三部分可以**原样复用**，
只需替换传感器读取和指令下发的接口。

### 4.1 控制架构

```
   参考动作 (.npz)                机器人传感器 (IMU + 编码器)
          │                              │
          ▼                              ▼
    构建 `command`  ┌──────────────► 构建 observation
                    │                    │
                    │                    ▼
                    │            ONNX 策略推理   (每 control_dt，例如 50 Hz)
                    │                    │
                    │                    ▼
                    │      动作 → 关节目标位置
                    │                    │
                    └────────────────────┼─► LowCmd{ q=目标, Kp, Kd, tau=0 }
                                         ▼
                          电机侧 PD 环  (~1 kHz，运行在机器人上)
```

策略以 `control_dt` 输出**位置目标**；机器人电机控制器用 `joint_stiffness` /
`joint_damping` 跑高频 PD 环。

### 4.2 构建观测（observation）

可部署的观测块为
[`_current_observation_block`](src/yahmp/scripts/deploy/run_yahmp_onnx_mujoco.py)
（外加其历史缓冲）。把每一项映射到真实传感器：

| 观测项 | 真机 G1 上的来源 |
|---|---|
| `command` | 从重定向后的 `.npz` 片段采样（见 4.4）。**来自动作片段，而非机器人。** |
| `base_ang_vel` | IMU 陀螺仪，转到 base/机体坐标系 |
| `projected_gravity` | 通过 IMU 姿态把重力方向旋转到机体坐标系 |
| `joint_pos` | 电机编码器位置 **减去** `default_joint_pos` |
| `joint_vel` | 电机编码器速度 |
| `actions` | 上一次策略输出 |
| `history` | 上述观测块的环形缓冲（长度取自元数据 `history_length`） |

> ⚠️ **务必检查导出 ONNX 的 `observation_terms_layout`。** MuJoCo 脚本里还会计算
> `base_lin_vel`，但可部署的（base/student）策略**不包含**它——base 线速度属于
> 特权观测（仅 teacher 使用），真机上难以估计。如果你的布局里出现了
> `base_lin_vel`，说明该 checkpoint 在没有状态估计器的情况下**无法**直接部署。

### 4.3 动作解码

参考 [`_apply_action`](src/yahmp/scripts/deploy/run_yahmp_onnx_mujoco.py)：

```python
processed = raw_action * action_scale + action_offset
if action_semantics == "residual_joint_position":
    target = reference_joint_pos[action_target_indices] + processed
else:  # "joint_position"
    target = processed
# 下发给电机：
LowCmd[j] = { q = target[j], dq = 0, Kp = joint_stiffness[j],
              Kd = joint_damping[j], tau = 0 }
```

### 4.4 构建 `command`（参考指令）

参考
[`_single_command_step_value`](src/yahmp/scripts/deploy/run_yahmp_onnx_mujoco.py)，
每个参考帧（每步维度 = `2 * 关节数 + 6`）：

```
[ joint_pos_ref,                 # 关节数
  joint_vel_ref,                 # 关节数
  anchor_lin_vel_body_xy,        # 2   （机体系平面速度）
  anchor_ang_vel_body_yaw,       # 1   （偏航角速度）
  root_height_z,                 # 1
  root_roll, root_pitch ]        # 2
```

对于 **Future** 变体（`Mjlab-YAHMP-Future-Unitree-G1`），按
`motion_command_step_offsets` 中每个偏移各拼接一个块，采样时刻为
`time + offset * control_dt`。

### 4.5 真机循环骨架

```python
spec = PolicySpec.from_onnx(onnx_path)          # 复用 MuJoCo 脚本里的类
clip = MotionClip(motion_file, spec.root_body_name)
session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

move_to_default(spec.default_joint_pos)         # 缓慢、安全地归位——需你自己实现
history = init_history(spec)
prev_action = zeros(spec.action_dim)
t = 0.0

while running:
    tic = time.perf_counter()

    low_state   = robot.read_low_state()        # Unitree SDK：编码器 + IMU
    frame       = clip.sample(t)                # t 时刻的参考帧
    obs         = build_observation(spec, low_state, frame, prev_action, history)

    raw_action  = session.run(None, {input_name: obs[None]})[0].reshape(-1)
    target      = decode_action(spec, raw_action, frame)   # 见 4.3

    robot.send_low_cmd(q=target,                # Unitree SDK：LowCmd
                       Kp=spec.joint_stiffness,
                       Kd=spec.joint_damping, tau=0)

    prev_action = raw_action
    t += spec.control_dt
    append_history(spec, history, next_terms)
    sleep_until(tic + spec.control_dt)          # 保持 control_dt 节拍
```

### 4.6 关节顺序映射（关键）

`spec.joint_names` 里的顺序是 **mjlab** 的顺序，**不等于** Unitree SDK 的电机
下标顺序。必须建立显式的双向索引映射并写单元测试。**映射错误是最危险的
bug**——它会悄无声息地控制到错误的电机。

---

## 5. 实时动捕遥操作（sim2sim）

除了回放录制好的 `.npz` 片段（第 2 节），你还可以用动捕数据流**实时**驱动策略的
运动指令（command）。脚本
[`run_yahmp_onnx_mocap.py`](src/yahmp/scripts/deploy/run_yahmp_onnx_mocap.py)
把**策略放进闭环**：仿真中的 G1 会实时跟踪动捕系统重定向出来的人体动作
（区别于纯运动学回放——后者只是把姿态直接拷到模型上，没有策略参与）。

```
  机器人状态 (MuJoCo 仿真) ─┐
                            ├─► observation ─► ONNX ─► 动作 ─► 仿真
  指令 (动捕，实时) ────────┘
```

它**原样复用** `run_yahmp_onnx_mujoco.py` 的观测/动作流水线，只新增了两部分：一个
实时姿态数据源，以及一个 `LiveReference`——把最新姿态转成指令构建所需的
`MotionFrame`，并对关节/根速度做有限差分。

### 5.1 免设备测试（无需硬件）

把一个片段**当作**实时设备来推流，即可验证整条实时指令链路，无需任何动捕硬件：

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --source npz \
  --npz-clip assets/motions/g1_omomo_amass_clean/<motion>.npz
```

若仿真 G1 在这里能正常跟踪，说明链路正确，剩下的只是设备接线问题。

### 5.2 ChingMu 实时遥操作

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
  --task-id Mjlab-YAHMP-Unitree-G1 --onnx-path assets/models/g1_yahmp.onnx \
  --source chingmu \
  --chingmu-host MCAvatar@192.168.123.112 \
  --sensor-root 0 --sensor-joint-first 1 \
  --joint-order config/chingmu_joint_order.json \
  --vel-smoothing 0.7
```

ChingMu 的 VRPN 库已**随仓库自带**，位于
[`third_party/chingmu/libCMVrpn.so`](third_party/chingmu/)，默认自动加载——无需
绝对路径。只有当你把库放在别处时才需要用 `--chingmu-dll` 覆盖。

重定向（人体 → G1 骨架）是在 **ChingMu 软件上游**完成的；VRPN 数据流直接给出
重定向后的根位姿 + 各关节角度，数据源负责读取（与标准 ChingMu VRPN 客户端一致的
`[qx,qy,qz,qw]→[qw,qx,qy,qz]` 与 mm→m 约定）。有两个**必做**的配置步骤：

| 步骤 | 为什么 | 怎么做 |
|---|---|---|
| **关节顺序映射** | 动捕按其骨架顺序推送关节；策略要的是 `spec.joint_names` 顺序。映射错会悄无声息地控制到错误的关节。 | 打印策略关节顺序 + 你的 ChingMu 段顺序（下面两条命令），按数据流顺序写成 JSON 列表，用 `--joint-order` 传入。 |
| **坐标系标定** | 指令用到根高度、roll/pitch 以及机体系速度；动捕/世界坐标系不一致会让策略跟踪到错误目标。 | 在 `ChingMuMocapSource._on_tracker` 里对 `root_pos` / `root_quat` 施加偏移/旋转。 |

打印策略关节顺序：

```bash
uv run python -c "import onnx; m=onnx.load('assets/models/g1_yahmp.onnx'); print({e.key:e.value for e in m.metadata_props}['joint_names'])"
```

导出你的 ChingMu 骨架段/传感器顺序（自带的独立小工具，使用仓库自带的库）：

```bash
uv run python -m yahmp.scripts.deploy.chingmu_hierarchy --host MCAvatar@192.168.123.112
```

仓库已附带一个 identity 顺序的
[`config/chingmu_joint_order.json`](config/chingmu_joint_order.json) 模板——按上面
的导出结果重新排序即可（见 [`config/README.md`](config/README.md)）。

### 5.3 注意事项

- **使用基础策略。** `Mjlab-YAHMP-Future-Unitree-G1` 需要**未来**参考帧，而实时流
  里并不存在（脚本会用当前姿态近似替代）。遥操作请优先用 `Mjlab-YAHMP-Unitree-G1`。
- **速度噪声。** 有限差分得到的动捕速度会抖动——调 `--vel-smoothing 0.5–0.8`，
  或把真实速度流喂进 `MocapState.update(joint_vel=...)`。
- **仅 sim2sim。** 若要 **sim2real**，按第 4 节把 MuJoCo 的 `data` 读写替换为
  Unitree SDK 即可——动捕/指令这一半完全不变。先完成第 6 节的安全清单。

---

## 6. 真机安全清单

- [ ] **吊架 / 龙门架**：首次运行务必把机器人吊起、双脚离地。
- [ ] **急停开关**：启用策略前先接好并测试。
- [ ] **先归位**：启动推理前用缓慢斜坡运动到 `default_joint_pos`，切勿瞬间跳到该姿态。
- [ ] **增益与训练一致**：直接使用 ONNX 元数据里的 `joint_stiffness` / `joint_damping`，不要改。
- [ ] **关节顺序映射**：用单元测试验证正确。
- [ ] **限幅保护**：下发 `LowCmd` 前对关节位置/速度限位和力矩做钳制。
- [ ] **循环时序**：保持 `control_dt` 节拍；检测并处理错过的截止时间 / 过期的 `LowState`。
- [ ] **先仿真**：同一 ONNX + 同一片段已在 MuJoCo 验证（第 2 节）。
- [ ] 从**短而平缓**的片段开始；只有在稳定之后再逐步过渡到剧烈动作。

> **Sim-to-real 已部分内建于训练。** 配置里使用了 `DelayedActuatorCfg`
> （0–4 步位置延迟）以及质量 / 质心 / 摩擦 / 推力随机化，这正是迁移得以成功的
> 关键。但这**不是**成功的保证——请预留调参时间，且绝不要省掉吊架。

---

## 7. 常见问题排查

| 现象 | 解决办法 |
|---|---|
| `Command 'uv' not found` | 安装 uv（第 0 节）；`export PATH="$HOME/.local/bin:$PATH"`。 |
| `source $HOME/.local/bin/env: No such file` | 新版 uv 不再生成该文件；直接把 `~/.local/bin` 加入 `PATH` 即可。 |
| `onnxruntime-gpu ... only has wheels for cp311/cp312/cp313` | 系统 Python 是 3.10。执行 `uv python pin 3.12 && uv sync`。 |
| `uv sync` 很慢 / 很大 | 首次运行正常（torch + CUDA 库），之后会走缓存。 |
| 推理时 CUDA/cuDNN 加载错误 | ONNX Runtime 与 CUDA/cuDNN 版本不匹配；先用 `--ort-provider cpu` 排查，再对齐 CUDA 版本。 |
| `Missing ONNX metadata key ...` | 用当前版本的导出脚本重新导出（第 1 节）。 |
| MuJoCo 报 `physics_dt`/`control_dt` 不匹配 | ONNX 是用与 `--task-id` 不同的任务配置导出的；重新导出或传入匹配的任务。 |
| 遥操作：`No mocap frame detected within timeout` | 检查 `--chingmu-host` / `--sensor-root`；确认 ChingMu VRPN 数据流已开启（可用 `get_hierarchy.py` 测试）。 |
| 遥操作：`Joints missing from --joint-order` | `--joint-order` JSON 缺少策略 `joint_names` 中的某个关节；打印策略顺序（5.2 节）对齐。 |
| 遥操作：机器人跟踪出镜像/漂移的姿态 | 坐标系标定问题（5.2 节）——在 `ChingMuMocapSource._on_tracker` 里修正根偏移/旋转。 |
| 遥操作：机器人剧烈抖动 | 提高 `--vel-smoothing`（如 `0.7`），或改用基础（非 Future）策略。 |

---

## 命令速查

```bash
# 环境
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python pin 3.12
uv sync

# 导出 ONNX
uv run python -m yahmp.scripts.deploy.export_checkpoint_to_onnx \
  --task-id Mjlab-YAHMP-Unitree-G1 --checkpoint-file model.pt \
  --output-path assets/models/g1_yahmp.onnx

# 仿真验证
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mujoco \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --motion-file assets/motions/g1_omomo_amass_clean/<motion>.npz

# 动捕遥操作，免设备测试（sim2sim）
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mocap \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --source npz --npz-clip assets/motions/g1_omomo_amass_clean/<motion>.npz

# 真机：把 run_yahmp_onnx_mujoco.py 移植到 unitree_sdk2_python（第 4 节）
```
