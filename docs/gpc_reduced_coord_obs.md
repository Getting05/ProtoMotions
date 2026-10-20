# P2 GPC reduced_coord_obs：部署对齐 V3

已在 2号机 `/data/chenguanting/ProtoMotions` 的独立 reduced 实验中生效。
参考为用户提供的 `/home/getting/下载/rl_deploy-develop` 快照；未修改该部署仓库。
原有 GPC/FSQ 实验及原 P2 模型、默认机器人配置保持不变。

## 实际输入定义

| slice | 信号 | 约定 |
|---|---|---|
| 0:29 | motor_q | 模型关节零位下的角度，rad，不减默认站姿；排除 head_joint |
| 29:58 | motor_dq | 对应关节角速度，rad/s，直接使用原始 dq，无滤波 |
| 58:62 | imu_quat | pelvis body-to-world，XYZW，单位化；按部署保留四元数符号 |
| 62:65 | imu_gyro | pelvis 轴，rad/s |
| 65:68 | imu_accel | IMU 测量点的 specific force，pelvis 轴，m/s² |
| 68:71 | projected_gravity | R_pelvis^-1[0,0,-1] |

IMU 测量点从 V1 的 pelvis 原点改为部署模型的 `[0,0,-0.04775] m`。
加速度包含根原点平动加速度、`alpha × r` 与 `omega × (omega × r)`，不去除重力比力。
静止直立仍约 `[0,0,9.81]`。采用 5 ms 差分模拟，属于有限差分近似，不是硬件噪声/时延模型。
投影重力、姿态和角速度继续对应 pelvis；不换成 torso。

训练保留原有 **200 Hz 物理步 × decimation 4，策略 50 Hz**。
每个物理子步为 5 ms，关节速度直接取模拟器原始 dq；腰部和其他关节均不做低通滤波。
q、IMU 姿态、角速度和加速度也未新增滤波。IMU 加速度仍由相邻物理状态差分计算。
部署仓库现有滤波不随此次修改而改变，训练不再复制其 500 Hz 更新或截止频率。
真实硬件打包可调用 `pack_deploy_sensors`；SDK qua 是 WXYZ，必须先转为 XYZW。
SDK 的 q/dq 是换算后的关节空间；并联驱动的 q_raw/dq_raw 不能作为这些输入。

## 动作与控制参数

FSQ actor/decoder 改为 29 维策略动作；prior/PEFT 从这条 tracker 链继承 29 维。
独立环境将动作补为 30 维仿真输入，头部始终使用默认位置 0 rad。
头部使用部署 fixed gains：Kp=40、Kd=2；29 个策略关节采用部署 protomotions_tracker 的 rl_kp/rl_kd。
例如髋/膝 Kp=157.52446966411836、Kd=17.907712860695533。
所有 30 个 joint range 与部署默认 XML 对齐，写入新的独立 MJCF，不修改原资产。
显式刷新 KinematicInfo 和 control_info，保证物理限位检查、PD 和 action scale 使用实际新值。

训练仍使用默认站姿为中心的 `q_target=q_default+scale*tanh(action)`；scale 按当前部署限位重建。
导出时若输出已是 joint_pos_targets，部署不能再次 tanh/缩放/加 default pose。
本次没有把新模型强行改成旧 tracker 的八输入签名；新 GPC 的导出/adapter 需要独立合同。

## 历史与 checkpoint

AMP 的在线历史存储真实 50 Hz 传感器观测；专家参考使用同一 200 Hz 点 IMU 和原始 dq。
专家参考无滤波预热，仅为最早一帧加速度保留一个前置物理采样；按 32 样本分块限制插值峰值内存。
IMU 前一采样和观测历史支持局部 reset 与 checkpoint 保存/恢复，无滤波状态。
物理 guard 路径在每个子步恢复有限状态之后才采样，避免传感器读取提前打断隔离流程。
评估器动作缓冲改用 env.get_action_size()，支持策略29维、机器人30关节；原30维策略行为不变。

新默认实验名为 `p2_<stage>_reduced_coord_obs_deploy_v3`，避免加载 V1/V2 同名冻结配置。
stage2/3 检查传感器合同版本、四元数约定、29维 tracker 和 PD 参数；拒绝 V1/V2 reduced/旧 max-coords checkpoint。
不要将 V1/V2 checkpoint 直接混入 V3 训练链。所有三阶段使用同一条 V3 链。

## 启动

```bash
cd /data/chenguanting/ProtoMotions
bash scripts/train_p2_gpc_reduced_coord_obs.sh fsq

TRACKER_CHECKPOINT=results/p2_fsq_reduced_coord_obs_deploy_v3/score_based.ckpt \
  bash scripts/train_p2_gpc_reduced_coord_obs.sh prior

PRIOR_CHECKPOINT=results/p2_prior_reduced_coord_obs_deploy_v3/inference_last.ckpt \
  bash scripts/train_p2_gpc_reduced_coord_obs.sh target_prior_peft

TRACKER_CHECKPOINT=results/p2_fsq_reduced_coord_obs_deploy_v3/score_based.ckpt \
PRIOR_CHECKPOINT=results/p2_prior_reduced_coord_obs_deploy_v3/inference_last.ckpt \
  bash scripts/train_p2_gpc_reduced_coord_obs.sh sft_target_prior_peft
```

支持全部 11 份 `_reduced_coord_obs.py` 实验。默认 8 GPU、每卡4096 env、batch16384；
`NGPU/NUM_ENVS/BATCH_SIZE/MOTION_FILE` 可覆盖，`DRY_RUN=1` 打印命令。
W&B 按需附加 `--use-wandb --wandb-project p2-gpc --wandb-video`。
当前独立传感器环境仅支持 IsaacLab。

## 修改位置

相对远程仓库根目录：

- `protomotions/envs/obs/p2_deploy_contract.json`：部署参数与五个参考源文件哈希。
- `protomotions/envs/obs/p2_deploy_sensors.py`：IMU 安装点、200 Hz 采样、原始 dq、历史和 AMP 专家流水线。
- `protomotions/envs/base_env/p2_deploy.py`：独立环境、29→30动作、子步采样、局部重置和状态恢复。
- `protomotions/envs/context_views.py`：新增可选传感器 view，原环境构造不变。
- `protomotions/envs/component_factories.py`：仅调整本次新增的 reduced factories。
- `protomotions/envs/obs/p2_reduced_coord.py`：旧计算保留，packer 增加可选四元数符号处理。
- `protomotions/data/assets/astro_p2/mjcf/astro_p2_deploy_reduced.xml`：独立限位资产。
- `examples/experiments/gpc/p2_deploy_setup.py`：新实验共用的部署 PD、限位和采样频率设置。
- 全部11份新 reduced 实验：绑定独立环境/参数，FSQ输出29维，AMP参考绑定新流水线。
- `examples/experiments/gpc/reduced_coord_context.py`：V3跨阶段兼容检查。
- `protomotions/agents/evaluators/mimic_evaluator.py`：动作记录尺寸改为策略action_size，一行修复。
- `scripts/train_p2_gpc_reduced_coord_obs.sh`：新默认实验名。
- `protomotions/tests/test_p2_deploy_alignment.py`、原观测测试：数值/配置/恢复/评估回归。

## 使用边界

测试实验目录和运行记录已按要求清理。
当前未启动正式长训；未验证8卡DDP、实机、W&B视频或新GPC ONNX接入。
部署侧并联闭链/TN动力学、5ms仿真延迟尚未移植到IsaacLab；保留此前对齐的观测/动作定义及PD/限位参数。
实机 SDK 零点、IMU 单位/轴向和安装标定仍需现场证据，未在代码中猜测标定值。
