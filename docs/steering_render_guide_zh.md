# Stage 3 Steering 评估与视频录制操作指南

入口：`scripts/evaluation/render_steering.py`。实现分布在：

- `protomotions/utils/steering_review.py`：命令协议、速度箭头几何、CPU 快照和 SHA256 校验。
- `protomotions/utils/steering_video.py`：IsaacLab 闭环评估、多视角录像及遥测。
- `protomotions/utils/steering_report.py`：完整视频解码、预览图、曲线及同条件对比。
- `protomotions/utils/policy_video.py`：通过 `kind=steering` 请求接入正常 `inference_agent`，普通策略录像保留原路径。

旧的临时脚本仍保留在 `output/steering_atom2_review_20261015/review-steering.py`；后续使用本指南中的正式入口。

## 1. 视频中的红、蓝箭头

红箭头表示**目标水平速度矢量**，方向来自 steering 移动命令，长度采用世界坐标单位：

```text
红箭头长度（米） = 目标速度（米/秒） × arrow_scale（秒）
```

默认 `--arrow-scale 0.35`，因此：

| 目标速度 | 红箭头长度 |
|---|---|
| 0 m/s | 0，隐藏整支红箭头，包括箭头头部 |
| 1.2 m/s | 0.42 m |
| 2 m/s | 0.70 m |
| 4 m/s | 1.40 m |

箭头没有固定的最小长度或默认长度偏移。短箭头的头部同步缩小。
蓝箭头表示**目标朝向**，固定长度 1 米；移动方向与朝向允许不同，便于观察侧移和后退。

红箭头随目标速度变化，不随实际速度变化。停止命令下即便机器人仍在移动，红箭头也隐藏；HUD 仍显示 `target=0` 与实际速度，用来判断是否停住。
`actual` 是沿目标移动方向的速度；停止效果应看报告中的 `speed_horizontal`（水平速度模长）。
世界坐标中的长度严格成比例；屏幕像素长度还会受到透视和观察方向影响。

## 2. 运行环境与文件要求

在仓库根目录运行。1号机示例：

```bash
cd /data/chenguanting/ProtoMotions
PYTHON=/data/chenguanting/.venvs/protomotions-isaaclab/bin/python
$PYTHON scripts/evaluation/render_steering.py --help
$PYTHON scripts/evaluation/render_steering.py run --help
nvidia-smi
```

渲染要求已有可运行的 IsaacLab/Isaac Sim、机器人资产、CUDA 环境，以及 `imageio`、`imageio_ffmpeg`、Pillow、matplotlib。
`--help`、`prepare`、`report` 和 `compare` 都不启动模拟器；`prepare` 仅用 CPU 读写 checkpoint 和动作。
渲染使用独立子进程，并将其 `WANDB_MODE` 设为 `disabled`，清除继承的 DDP rank 变量。

需要准备：

1. **完整 Stage 3 `last.ckpt`**，包含冻结 reference prior。不能用仅含 adapter 的 `inference_last.ckpt` 替代。
2. 同目录的 `resolved_configs.pt` 和 `resolved_configs_inference.pt`。
3. 冻结配置指向的 Stage 2 prior checkpoint 及其父目录 `resolved_configs.pt`。
4. 一个真实存在的物理动作 shard，例如 `bones_seed_train_astro_p2_0.pt`，不要传含 `slurmrank` 的逻辑占位路径。
5. 初始动作 ID，例如 `0 2 10 100`；ID 是该物理 shard 内部的局部索引。

路径失效会直接报错。只有显式提供 `--prior-checkpoint /正确路径/prior.ckpt` 才会覆盖冻结配置里的 prior 路径。
快照目录应使用新名字；程序拒绝覆盖已有快照或已有渲染结果。

## 3. 在 1号机一条命令完成准备、录制和报告

下面使用已经从 2号机转来的 epoch660 完整 checkpoint：

```bash
cd /data/chenguanting/ProtoMotions
PYTHON=/data/chenguanting/.venvs/protomotions-isaaclab/bin/python

$PYTHON scripts/evaluation/render_steering.py run \
  --checkpoint output/steering_atom2_review_20261015/policy/last.ckpt \
  --motion-file ../datasets/BONES-SEED/p2_motionlib_lze_contact_orin/bones_seed_train_astro_p2_0.pt \
  --motion-ids 0 2 10 100 \
  --output-dir output/steering_epoch660_scaled_arrows \
  --gpu 4 \
  --views turn_left speed_stop \
  --duration 16 \
  --switch-time 6 \
  --steady-start 8 \
  --arrow-scale 0.35
```

`--gpu 4` 是示例物理 GPU 编号，执行前用 `nvidia-smi` 选择有足够空闲显存的卡。
程序只为自己的推理子进程设置 `CUDA_VISIBLE_DEVICES`，不停止、重启或修改训练任务。
不要在训练只剩少量空闲显存时叠加渲染；本工具的实际占用应以启动日志和 `nvidia-smi` 为准。

默认同时输出两段视频：

- `turn_left.mp4`：第 6 秒将移动方向和朝向一起左转 90°。
- `speed_stop.mp4`：前半段跟随高速工况，后半段切到停止工况的机器人。停止工况的指令在第 6 秒已经置零。

这些视角来自**同一次 32 环境 rollout**。第二段在中点切换摄像机跟随的环境，不是让同一机器人在中点从高速切换为停止。
每个视频默认展示初始动作 ID 0；可用 `--video-motion-id 2` 观察同一工况下另一初始动作。

## 4. 2号机准备 checkpoint，1号机渲染

两台机器先使用包含本工具的同一仓库版本；目标机器需具备对应机器人资产和可运行的 IsaacLab 环境。
快照包含策略、prior、冻结配置和选出的少量动作，不打包模拟器或机器人资产。

### 4.1 在 2号机固定最新 checkpoint

账号：`atom8@172.16.9.7`。在 2号机执行：

```bash
cd /data/chenguanting/ProtoMotions
PYTHON=/data/chenguanting/.venvs/protomotions-isaaclab/bin/python

$PYTHON scripts/evaluation/render_steering.py prepare \
  --checkpoint results/p2_gpc_stage3_steering_peft_1014/last.ckpt \
  --motion-file ../datasets/BONES-SEED/p2_motionlib_lze_contact_orin/bones_seed_train_astro_p2_0.pt \
  --motion-ids 0 2 10 100 \
  --output-dir output/steering_transfer_20261015_a
```

程序先复制稳定 checkpoint；若复制期间源文件 inode、大小或修改时间改变，会重试，连续三次改变则退出。
快照不会持续追随训练更新：查看 `manifest.json` 中实际固定的 `epoch`、`step_count` 和源文件时间。
策略和 prior 都会保存到独立文件，不修改训练中的 `last.ckpt`。

### 4.2 在 1号机拉取整包

账号：`chenguanting@172.16.9.8`。当前配置允许由 1号机连接 2号机，可在 1号机执行：

```bash
cd /data/chenguanting/ProtoMotions

rsync -a --info=progress2 \
  -e 'ssh -o BatchMode=yes -o ForwardAgent=no' \
  atom8@172.16.9.7:/data/chenguanting/ProtoMotions/output/steering_transfer_20261015_a/ \
  output/steering_transfer_20261015_a/
```

传输整个目录，包括 `prior/` 和 `selected_motions.pt`。无需复制 SSH 私钥或打开 agent forwarding。
传输后允许改变目录名；渲染时按目标机器上的新位置重建运行配置。

### 4.3 在 1号机录制

```bash
PYTHON=/data/chenguanting/.venvs/protomotions-isaaclab/bin/python

$PYTHON scripts/evaluation/render_steering.py render \
  --prepared-dir output/steering_transfer_20261015_a \
  --gpu 4 \
  --views turn_left speed_stop \
  --arrow-scale 0.35
```

渲染前会核对整个快照的大小和 SHA256。
原始冻结配置不会被改写；重定位的 prior 路径只写入 `render/runtime/`。
策略 checkpoint 在同一文件系统上使用硬链接供推理读取，链接失败时退回复制。

## 5. 可选视角和协议参数

`--views` 可选：

```text
turn_left turn_right straight_low straight_cruise straight_high
stop strafe_left backward speed_stop
```

例如：

```bash
$PYTHON scripts/evaluation/render_steering.py run \
  --checkpoint results/p2_gpc_stage3_steering_peft_1014/last.ckpt \
  --motion-file ../datasets/BONES-SEED/p2_motionlib_lze_contact_orin/bones_seed_train_astro_p2_0.pt \
  --output-dir output/steering_custom_views \
  --gpu 4 --views straight_high stop backward \
  --video-motion-id 10 --arrow-scale 0.5
```

此时 4 m/s 的红箭头长 2 米。增加视角只增加渲染开销，不额外重复物理 rollout。
不论选几个视角，始终评估全部八类工况。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--motion-ids` | `0 2 10 100` | 初始动作集合；环境数为动作数 × 8 |
| `--duration` | 16 | 每个环境模拟时长，秒 |
| `--switch-time` | 6 | 左／右转向及停止命令的切换时刻 |
| `--steady-start` | 8 | 汇总均值的起始时刻，结束为 duration |
| `--low-speed` | 1.2 | 低速直行、侧移、后退的目标速度 |
| `--cruise-speed` | 2.0 | 中速直行及左右转目标速度；停止前也用此速度 |
| `--high-speed` | 4.0 | 高速直行目标速度 |
| `--seed` | 0 | rollout 重置和策略采样的种子 |
| `--fps` | 15 | 编码帧率；不改变模拟 dt |
| `--width` / `--height` | 960 / 720 | 视频大小，使用偶数尺寸 |
| `--arrow-scale` | 0.35 | 每 1 m/s 对应的红箭头米数 |
| `--min-height` | 0.30 | 骨盆世界坐标高度阈值，米 |
| `--min-upright` | 0.50 | 骨盆局部上轴与世界 Z 轴夹角余弦阈值 |
| `--fall-hold` | 0.25 | 异常姿态持续多久才记录一次失稳 |
| `--video-motion-id` | 第一个动作 ID | 视频跟随的初始动作 ID；不改变统计集合 |

必须满足 `0 < switch-time < steady-start < duration`。
短时 smoke 示例：`--duration 8 --switch-time 3 --steady-start 4 --motion-ids 0 2`。
短时 smoke 用于检查工具链，不能替代 16 秒或更长的稳定性评估。

## 6. 输出文件、进度和复核

```text
output/steering_epoch660_scaled_arrows/
  manifest.json                 # 源 checkpoint、epoch、step、文件校验值
  policy/last.ckpt              # 完整 Stage 3 checkpoint
  policy/resolved_configs*.pt   # 未修改的源冻结配置
  prior/prior.ckpt              # 对应冻结 prior
  prior/resolved_configs.pt
  selected_motions.pt           # 仅指定动作的帧，不是完整 6GB shard
  render/
    render.log                 # 模型加载、参考策略恢复与 rollout 进度
    request.json               # 实际协议、视角、速度比例及数据校验标识
    runtime/                   # 重定位的推理配置、只读加载的 checkpoint、guard 日志
    turn_left.mp4
    speed_stop.mp4
    metrics.json               # 各工况速度、朝向误差、持续姿态稳定性
    metrics.csv
    telemetry.npz              # 每个时间步、每个环境的完整遥测
    arrow_samples.json         # 每个录制视角的目标速度、红箭头长度和可见性
    media_verification.json    # 完整解码帧数、时长、文件 SHA256
    speed_tracking.png
    preview_turn_left.jpg
    preview_speed_stop.jpg
```

另一个 SSH 终端查看进度：

```bash
tail -f output/steering_epoch660_scaled_arrows/render/render.log
```

看到 `STEERING_REVIEW_COMPLETE` 后还需等待 CLI 完成报告和完整视频解码校验。
最终终端出现 `Review complete` 才表示整个命令成功。

只重新生成图表／解码复核，不重跑模拟器：

```bash
$PYTHON scripts/evaluation/render_steering.py report \
  --render-dir output/steering_epoch660_scaled_arrows/render
```

`arrow_samples.json` 可用于核对 2 m/s → 0.7 m、4 m/s → 1.4 m、停止时长度 0 且不可见。

## 7. 如何解读和比较结果

运动开始时从选定动作的首帧重置，并以各自初始朝向为命令零方向。每个工况包含所有指定初始动作。
录制期间不会因参考 motion 播放结束而重置，也不会按训练的短 episode horizon 提前结束。
初始动作只负责初始化，这不是 motion tracking 视频，因此不画参考动作的蓝色机器人；蓝色只用于目标朝向箭头。

- `speed_along`：沿目标移动方向速度；可能为负。
- `speed_horizontal`：水平速度模长，判断停车时优先看此项。
- `velocity_error`：实际水平速度向量与目标速度向量的误差模长。
- `facing_error_deg`：机器人 anchor 的实际朝向与目标朝向的角度差。
- `survived`：整段 rollout 没有触发持续姿态阈值的数量，不等于真实接触标注的跌倒识别。
- `first_fall_time`：首次持续阈值失败时刻；`null` 表示没触发。
- `physics_guard_recovered`：实际 guard 恢复次数，非零时需检查遥测中的状态跳变。

默认在平地评估，骨盆高度使用世界 Z 坐标，不能直接当作崎岖地形的离地高度。
速度统计包含失稳后的帧；高速只看均值会被倒地轨迹拉低，应结合 `survived`、首次失稳时间和曲线。

比较由本工具生成的两个 render 目录：

```bash
$PYTHON scripts/evaluation/render_steering.py compare \
  --baseline output/review_old/render \
  --candidate output/review_new/render \
  --output-dir output/review_comparison
```

输出 `comparison.png` 和 `comparison.json`。
程序要求 prior、初始动作数据、命令速度、随机种子、模拟 dt、时长和姿态阈值一致，避免将条件不同的结果标为同条件比较。
FPS、图像大小、视角和箭头比例可以不同，因为这些参数不改变物理命令。
旧临时脚本的结果格式不作为此比较命令的输入；需要使用正式工具重新生成两个版本的报告。

## 8. 常见问题

- `full Stage 3 checkpoint`：传入了缺少 reference 的 adapter-only checkpoint；改用同目录完整 `last.ckpt`。
- `Source changed during ... copies`：训练恰好在覆盖 checkpoint，稍后用新的输出目录重新 `prepare`。
- `Snapshot checksum mismatch`：目录传输不完整或文件发生变化，重新同步，保留源 snapshot 的 `manifest.json`。
- `Initial motion ID ... outside`：ID 是物理 shard 内的局部索引；确认 shard 和 motion IDs 配套。
- `Render output exists`：已有 `render/` 结果，使用新快照目录，或只执行 `report` 复核旧结果。
- CUDA OOM：检查其他任务占用，换空闲 GPU 或减少初始动作数；减少视频尺寸主要影响渲染，不能替代对策略和模拟器显存的检查。
- 录像里没有红箭头：如果 HUD 的目标速度为 0，这是正确行为；实际仍在移动说明停车控制尚未完成。
- 训练奖励上升但高速仍倒：默认训练回合可能短于 16 秒，此工具刻意延长连续运行时间；看失稳时刻而不只看训练 episode length。
- 本工具不会自动修改奖励、恢复训练或上传 W&B。代码／数据／资产版本发生变化后，应重新固定快照并记录版本。
