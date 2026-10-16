# P2 Pelvis State Estimator

本轮在 `4090主机 /home/getting/ProtoMotions` 的 `codex/p2-pelvis-state-estimator` 分支实现。遵循用户最新要求：**只实现代码，暂不复制 Stage 1 权重/reference，也不进行真实 tracker 数据训练**。现有 GPC Stage 1/2/3 代码和 checkpoint 均未修改。

## 本轮交付与验证边界

- 独立采集器：冻结 Stage 1 tracker 执行 reference，记录真实物理 rollout，而非把 reference 状态当监督。
- 独立模块：URDF FK/Jacobian、IMU/接触标签、episode 数据集、contact GRU、纯 GRU baseline、Oracle-contact 融合、预测-contact 融合、训练、评估、实时接口、GPC 重建桥接。
- 合成数据上的三种模型均已完成短训练、反向传播、保存、加载、评估、曲线输出；这些权重只用于软件测试。
- 不加载模型的 Isaac 集成测试：2 environments × 120 control steps，50 Hz，中途 reset；实际接触点/地面过滤/IMU接口工作正常。
- 真实 GPC Stage 1 采集入口尚未做 checkpoint 驱动的端到端运行。动作覆盖、接触阈值、真实 held-out 精度、抗扰/滑移能力均待验证。
- 首版仅平地；采集器遇非平地拒绝运行。URDF足底几何点未必等于柔性脚底的有效支撑点，保留学习残差，需用真实数据消融。

## 坐标和数据契约

物理 floating base 为 `pelvis`，GPC heading anchor 为 `waist_roll_link`。四维输出：

`[pelvis_origin_z - ground_z_under_pelvis, pelvis_origin_velocity_in_waist_heading (3)]`

这与 `max_coords_obs` 的全部根特征**不是直接相同的四个数**：当前 max_coords 的高度是 waist anchor 高度，Isaac 刚体线速度使用 COM 速度。先估计 pelvis 原点状态，再用 `bridge.reconstruct_bodies` 调用原有 `deployment.urdf_kinematics.URDFKinematics`，转换成 waist 高度及各刚体 COM 速度，最后交给原有 max_coords 组装。平地上 pelvis 与 waist 下方地面高度相同；坡地需要额外地形信息。

所有关节使用 frozen robot config 的 **common DOF order**，不是 Isaac 后端关节顺序。采集器通过 `sim.get_dof_state()/get_dof_forces()` 完成顺序转换；URDF SHA256、DOF/body names、IMU、root、anchor、速度参考点写入 dataset/checkpoint contract。部署时检查 URDF checksum。

129 维输入切片：

| 切片 | 内容 | 单位/坐标 |
| --- | --- | --- |
| 0:30 | q | rad，common order |
| 30:60 | dq | rad/s |
| 60:90 | 估计 motor torque | Nm；仿真 applied torque 加测量误差 |
| 90:93 | projected gravity | 单位向量，IMU frame，静止正立为 [0,0,-1] |
| 93:96 | gyro | rad/s，IMU frame |
| 96:99 | accelerometer | m/s²，比力 `R_WI^T(a_IMU_world - g_world)` |
| 99:129 | previous PD target | rad，刚完成的控制区间实际 processed PD target |

不包含 steering 指令、base h/v、真实接触力或接触标签。采集读 `s_t` 的时刻，对应前一区间的 `u_(t-1)`；不是原始 FSQ action。只支持 PD 控制模式。

初版 IMU 为 URDF 的 **`imu_2_link`**，固定在 pelvis，安装平移约 `[0,0,-0.04775] m`、旋转 identity。`imu_1_link` 在腰部上方的运动连杆上，本版拒绝将其当 pelvis 固定 IMU。硬件是否实际提供 IMU2 及其外参仍需确认。

采集器用 IMU 安装点的世界速度做 50 Hz 因果差分，再变换到当前 IMU frame；因此包含杆臂的旋转加速度。这是区间平均加速度传感器模型，不是瞬时/200 Hz 原生 IMU。首个 post-reset 差分无效，该帧和包含它的训练窗口被排除。投影重力依赖部署端姿态解算，不能直接用包含运动加速度的 accelerometer 向量替代。

## 足部物理与接触监督

`physics.FootKinematics` 复用已有 URDF 树解析，Torch 批量计算 FK 和解析点 Jacobian。左右足为 `left/right_ankle_roll_link`；选择点从 URDF cylinder/sphere 足底碰撞几何的最低支撑面求得，允许脚跟/脚尖切换，不使用 ankle 原点。最低高度相差不超过 1 μm 的支撑点取平均。

Jacobian 针对**当前选定的刚体材料点**，不对“最小高度点选择函数”求时间导数。支撑候选为：

`h = -(R_WP r_PF)_z`

`v_P^P = -omega_P^P × r_PF - J_F(q) dq`

先用 projected gravity 构造任意 yaw 的 pelvis 姿态，再由 FK 得到 waist heading；最终 waist-heading 速度与全局 yaw gauge 无关。

`isaac_adapter` 单独继承 simulator，只给两只脚打开 raw contact tracking，过滤对象固定为 `/World/ground/terrain/mesh`。不改生产 simulator。它读取当前安装版本的 PhysX 原始 contact points/count/start buffers；API 改动或 buffer 饱和时明确报错，不退化为脚中心速度。

- 法向力：ground-filtered force matrix 的世界 +Z 分量。
- 滑速：每个实际接触点 `v_link_origin + omega × (point-origin)` 的水平速度，取 manifold 中最大值，避免均值掩盖转动滑移。
- 标签：`force > 30 N AND max_slip < 0.1 m/s AND 有实际接触点`。
- 阈值仅初始值。原始力/滑速/点数/接触点均作为监督诊断保存，不进入推理特征。
- 诊断同时比较 URDF 选定点的 FK 位置、Jacobian 速度与仿真在同一点的值，不把“假设脚静止”误差混入 Jacobian 验证。

## 模型和训练

配置文件：`configs/state_estimation/p2_pelvis.json`。

默认 25 帧 × 50 Hz、batch 256、AdamW 3e-4；Contact MLP129→128 / GRU128 / MLP128→64→2。融合 MLP144→256 / 2-layer GRU256，输出 delta/free 各4维。

`kin = sum(p_i * candidate_i)/(sum(p_i)+eps)`，`c = 1-(1-pL)(1-pR)`；两个候选冲突时，根据标准化差值和 `pL*pR` 降低 c。单脚可靠时不因摆动脚候选任意变化而受惩罚。最终 `c*(kin+delta)+(1-c)*free`。纯 GRU baseline 不将任何 FK/c/contact 特征送入状态编码器；同样训练 contact head 便于诊断。

三种可训练模式：`baseline`、`oracle`、`fusion`。Oracle 必须显式传真实 contact，实时接口拒绝加载 Oracle 权重。可部署模式拒绝传入 contact_override。另提供完全无网络的 physics-oracle，只在有可靠支撑的帧统计误差，同时报告支持覆盖率；腾空帧不能用该结果代表全时域性能。

loss 为归一化高度/速度 Huber 加 `0.2 * BCEWithLogitsLoss`。target scale 默认 `[0.5,1,1,1]`（m / m/s），输入 normalization 只由 train 帧计算，保存于权重。关节噪声、IMU噪声/固定episode bias、力矩估计误差、0–1帧传感器延迟在 train 数据加载时施加，物理候选从同一份受扰观测重新计算。噪声为每个 episode 一次可复现采样；并非每个 epoch 重采样。

采集默认用已有 domain-randomization API 随机化摩擦并施加线/角速度脉冲。这里的 push 是仿真速度脉冲近似，不是测力学的有限时长外力。可用 `--no-randomization` 先采干净 oracle 数据。

## 数据与拆分

每个 episode 一个 `.pt`，`index.json` 记录文件、原动作名/ID、group_id、category、split、dt、contract、采集来源。

必须给所有镜像、裁剪、重复/重定向版本指定同一个源 **group_id**。用 `SHA256(seed,group_id)` 固定分配 train/validation；不会随机划分相邻窗口，也不会跨 episode 拼历史。manifest 工具只自动去掉 `_M` 镜像后缀，无法自动识别所有源动作变体，需人工检查 group。数据加载再次拒绝跨 split 重复的 group。验证外部数据集时也检查它是否包含 checkpoint 的 train groups。

动作类别需要显式审核，不用文件名猜测动作覆盖。允许：`stand, walk_forward, walk_backward, walk_sideways, turn, stop, squat, rise, run, jump, push, slip`。采集额外记录 force-based airborne/landing/slip 事件及仿真实际施加的 push 脉冲事件。类别是 clip/实验标签；例如 push/slip 类别应对应专门扰动实验，不能仅重命名普通走路 clip 当成完成覆盖。

历史初始化：训练只取完整有效窗口；实时接口 reset 后前24帧返回 `None`，第25帧首次预测；时间戳倒退或间隔偏离周期25%以上会清空历史重新 warmup。GRU 每次重算固定25帧，与训练窗口一致，不将无限历史 hidden state 带入部署。

## 运行

以下命令在4090主机 repository root、Isaac Python/PYTHONPATH 环境下执行。建议先在容器内设置：

```bash
cd /home/getting/ProtoMotions
export PYTHONPATH=.
PY=/isaac-sim/python.sh
```

宿主机启动已有容器内命令的方式：

```bash
docker exec -e PYTHONPATH=. -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -w /home/getting/ProtoMotions getting-protomotions-isaaclab \
  /isaac-sim/python.sh scripts/check_p2_estimator_isaac.py \
  --output results/p2_estimator_sensor_check_new
```

输出目录要求不存在，防止覆盖旧实验。无模型代码检查：

```bash
$PY scripts/check_p2_estimator_isaac.py --output results/p2_sensor_check_new
$PY scripts/smoke_p2_estimator.py --output results/p2_synthetic_new --device cpu
$PY -m pytest protomotions/tests/test_p2_state_estimation.py -q
```

当前容器没有独立 pytest 安装，本次通过已有 `/data/getting/.venvs/unimate/lib/python3.10/site-packages` 的 pytest 包运行，并设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`；普通开发环境安装 pytest 即可。

**后续模型及动作文件可用后**（本轮未复制/运行）：

```bash
$PY scripts/prepare_p2_estimator_manifest.py \
  --motion-file /path/to/reference.pt --include 'walk|jump|idle|squat' \
  --output /path/to/motions.json
# 编辑 motions.json：审核 category 和 group_id；unclassified 会被采集器拒绝。
$PY scripts/collect_p2_estimator.py \
  --checkpoint /path/to/inference_score_based.ckpt \
  --motion-file /path/to/reference.pt --manifest /path/to/motions.json \
  --num-envs 8 --episodes 100 --max-steps 500 --output results/p2_dataset

$PY -m protomotions.state_estimation.evaluate --data results/p2_dataset \
  --physics-oracle --output results/p2_fk_oracle

$PY -m protomotions.state_estimation.train --data results/p2_dataset \
  --mode baseline --output results/p2_baseline
$PY -m protomotions.state_estimation.train --data results/p2_dataset \
  --mode oracle --output results/p2_oracle
$PY -m protomotions.state_estimation.train --data results/p2_dataset \
  --mode fusion --output results/p2_fusion

$PY -m protomotions.state_estimation.evaluate --data results/p2_dataset \
  --checkpoint results/p2_fusion/best.pt --output results/p2_eval_clean
$PY -m protomotions.state_estimation.evaluate --data results/p2_dataset \
  --checkpoint results/p2_fusion/best.pt --noise-config configs/state_estimation/p2_pelvis.json \
  --output results/p2_eval_noisy
```

对 baseline/oracle 重复 evaluate 比较相同 held-out motion groups。clean/noisy 需要分别报告，不要拿验证集调参后再当最终测试集；本版只实现 train/validation 两个 split，正式验收应另留独立 motion-family 测试集。

评估输出 `metrics.json`、所有帧的 `curves.pt`、每类一个 episode 的 height/velocity/error/contact PNG。逐类检查 height MAE≤0.03m、水平两分量 pooled RMSE≤0.10m/s（同时报告 vx/vy 独立 RMSE 与二维向量 RMSE，便于统一验收口径）、vz RMSE≤0.15m/s；报告 contact false-positive rate、腾空/落地/滑动/推搡事件误差。完整真实数据类别/事件覆盖、误差、P95≤20ms 同时满足才标记 acceptance_passed。合成数据、smoke训练、Oracle 永远不标记部署验收通过。

真实机器人调用：

```python
from protomotions.state_estimation.inference import RealtimeEstimator
estimator = RealtimeEstimator('results/p2_fusion/best.pt', device='cpu')
estimator.reset()  # 启动、episode边界或姿态解算重置
result = estimator.update(measured_proprio_129, timestamp_seconds)
if result is not None:
    pelvis_h_v = result['state']  # [4]，SI单位，waist heading
```

运行时不接受真实接触或基座 h/v。还需硬件端提供共同时间基准、joint order、校准的 pelvis IMU 外参、PD target 和 torque estimate。

## Stage 1 候选来源（仅只读确认）

2号机：`172.16.9.7`。

`/data/chenguanting/ProtoMotions/results/p2_fsq_bones_8gpu_lift_foot_smooth/inference_score_based.ckpt`

同目录 `resolved_configs_inference.pt`；reference 数据位于 `/data/chenguanting/datasets/BONES-SEED/p2_motionlib_lze_contact_orin/`。

选用 reference-conditioned FSQ tracker 的 inference checkpoint，epoch 12800；checkpoint 中 best_evaluated_score 为 `0.990966796875`。这是已有记录，不是本轮重评估，更不是状态估计器精度。用户要求暂不复制，因此本轮没有模型副本。

## 本轮实测

结果摘要见 `docs/p2_pelvis_estimator_verification.json`。

- 回归测试20项通过（新增估计器16项、已有URDF模块4项）；包含Jacobian有限差分、IMU杆臂/比力、COM转换、接触速度、反向传播、数据拆分、实时窗口等。
- 4090主机当前运行时：torch 2.11.0+cu128、NumPy2.3.1、SciPy1.17.0、IsaacLab package12.0.0（仓库称新版Warp/PhysX接口）、MuJoCo3.15.0。
- Isaac PD fixture：2×120帧，FK点位置最大绝对误差 `7.15e-7 m`，点速度误差 `1.28e-6 m/s`；448个足/帧存在真实接触，446个满足当前可靠支撑阈值。此小范围结果不能外推到所有动作。
- 随机摩擦/推搡fixture另运行2×120帧，记录8个环境帧的实际推搡；FK点速度最大误差1.14e-6m/s，集成检查通过。
- 完整融合实时CPU/2线程：P95约3.46ms，包括每帧FK和25帧网络窗口；权重来自合成 smoke，只验证结构计算成本。
- 本轮没有测得真实 tracker held-out 状态估计 MAE/RMSE，没有训练可部署精度模型。
