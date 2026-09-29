# SOMA → Astro P2：bone 模式 PyRoki

服务器新增文件：

- `/data/chenguanting/ProtoMotions/pyroki/batch_retarget_soma_to_astro_p2_bone.py`
- `/data/chenguanting/ProtoMotions/pyroki/soma_p2_bone_calibration.yaml`
- `/data/chenguanting/ProtoMotions/pyroki/README_SOMA_P2_BONE.md`

## 当前默认配置（用户确认，2026-09-28）

本次用户提交的参数已完整内置到 `batch_retarget_soma_to_astro_p2_bone.py` 的 `APPROVED_CALIBRATION`。普通运行不会重新拟合，也不依赖旁边 YAML 是否被修改。可用 `--calibration-config /path/to/custom.yaml` 显式覆盖。

默认输入为 `/data/chenguanting/datasets/BONES-SEED/soma_uniform`，当存在 `bvh/` 时只扫描该子目录；已枚举 **142,220 个原始 BVH**。默认输出是 `/data/chenguanting/datasets/BONES-SEED/p2_bone_pyroki`。未启动全量转换。

直接处理该数据集（约 120 FPS 的 BVH 每四帧取一帧，输出 30 FPS）：

```bash
cd /data/chenguanting/ProtoMotions
~/.venvs/astro-p2-retarget/bin/python pyroki/batch_retarget_soma_to_astro_p2_bone.py \
  --subsample-factor 4 --align-ground --skip-existing
```

运行前只检查输入数量：`--list-inputs`。小范围验证可加 `--limit-motions 1 --max-frames 80`，并指定独立验证输出目录。默认无这两个限制，完整处理选中动作。分片用 `--num-shards N --shard-index I`。

## 其他输入的运行方式

原始 SOMA77 NPZ，使用服务器现有样例：

```bash
cd /data/chenguanting/ProtoMotions
~/.venvs/astro-p2-retarget/bin/python pyroki/batch_retarget_soma_to_astro_p2_bone.py \
  --input data/soma-kimodo-generated/output.npz \
  --input-fps 30 \
  --output-dir /data/chenguanting/workspace/calibration-viewer-run/soma_bone_output \
  --align-ground --save-targets --skip-existing
```

已经转换好的 SOMA23 ProtoMotions `.motion`（读取文件中的 FPS）：

```bash
cd /data/chenguanting/ProtoMotions
~/.venvs/astro-p2-retarget/bin/python pyroki/batch_retarget_soma_to_astro_p2_bone.py \
  --input data/soma-kimodo-generated/proto/output.motion \
  --output-dir /data/chenguanting/workspace/calibration-viewer-run/soma_motion_bone_output \
  --align-ground --save-targets --skip-existing
```

`--input` 可传目录，递归处理 `.bvh/.npz/.motion` 并保留子目录；当传入 BONES-SEED 根目录且存在 `bvh/` 时，优先选择原始 BVH 子目录。目录中若同时存在同一动作的原始文件和转换文件，两者都会处理；选择所需的数据目录即可。原始 NPZ 未提供 FPS 时必须传 `--input-fps`，不能仅凭扩展名推断帧率。

默认完整处理每条动作。支持 `--subsample-factor`、`--start-frame`、`--max-frames`、`--num-shards` 和 `--shard-index`。默认每 120 帧一段、15 帧重叠，末段补帧并剔除补帧结果；位置/关节角线性拼接，根四元数 SLERP。`--prepare-only` 仅保存标定后目标，不运行 IK。

## 输入约定

**原始 BONES-SEED BVH**：

- 直接解析 SOMA77 层级与每个关节声明的 Euler 顺序，厘米转米。
- 使用仓库 `standard_t_pose_global_offsets_rots.p` 中的 77 个旋转矩阵，将 BVH 骨轴零姿态转为标准 T-pose，再按 SOMA23 MJCF 做 FK。该张量通过 `torch.load(weights_only=True)` 读取。
- 保留仓库已有 BVH 转换器的 Hips 绝对平移约定。对于非零或有动画的外层 Root，明确报错，不丢弃该变换。
- 帧率读取 `Frame Time`。如 `0.008333` 这类打印截断，在相对误差小于 1e-4 时恢复到最近整数帧率 120；`--input-fps` 可显式覆盖。抽帧后自动调整输出 FPS。
- 不调用旧转换器的动作过滤器，不依赖预先转换的 `.motion`；失败文件进入 manifest，并返回非零退出码。


**原始 NPZ**：

- `posed_joints [T,77,3]` 或 `[T,23,3]`，单位米，SOMA 原生 Y-up。
- `global_rot_mats [T,J,3,3]`，或 `local_rot_mats [T,J,3,3]`。优先使用 global；只有 local 时沿 SOMA23 树递推。
- 77 点选择与仓库 `convert_soma23_to_proto.py` 一致。23 点必须按 SOMA23 MJCF 的树遍历顺序；若提供 `joint_names`，按名字重新排序。
- 默认世界变换为 `[-x,z,y]`，与仓库现有 raw SOMA→`.motion` 的 rot2 变换一致。其他世界约定可通过 `--raw-world-axes z,x,y` 等显式配置。
- 局部 canonical 坐标统一为 `[forward,left,up]`：原生 SOMA 模板使用 `[z,x,y]`，MJCF SOMA23 使用 `[-y,x,z]`。

**SOMA23 `.motion`**：要求 COMMON 顺序、`rigid_body_pos [T,23,3]`、`rigid_body_rot [T,23,4]`（XYZW）、`fps`。世界坐标已经 Z-up，不再重复变换。使用 PyTorch `weights_only=True`，只额外允许仓库的 `StateConversion` 枚举。

不支持在此脚本中直接读取打包 MotionLib `.pt`、SMPL 参数或旧版 18 点 keypoints。`.motion` 必须确实来自 SOMA23，不能仅凭其他模型恰好有 23 个刚体就套用。

## SOMA 专用 bone 标定

默认使用脚本内置的用户确认标定，与本次提交的 `soma_p2_bone_calibration.yaml` 数值完全一致，不会复用 SMPL 的骨长比例。该 YAML 保留为可读副本。

附带配置由 SOMA23 MJCF 中性骨架与 P2 参考姿态自动拟合生成：仅从用户原来的 `astro_p2_calibration.yaml` 获取机器人点映射和 `t_pose_joint_positions`，不使用其中 SMPL 的缩放或偏移。需要另行拟合时，必须明确提供输出 YAML 路径；这不会修改已内置的默认值，使用新配置时需显式传 `--calibration-config`：

```bash
cd /data/chenguanting/ProtoMotions
~/.venvs/astro-p2-retarget/bin/python pyroki/batch_retarget_soma_to_astro_p2_bone.py \
  --fit-calibration \
  --calibration-config pyroki/soma_p2_bone_calibration.yaml
```

初始比例约为：torso **0.858235**、upper_arm **0.906402**、forearm **0.759862**、thigh **0.695201**、shank **0.634235**。肩、肘、髋三维偏移见 YAML。

这是**自动初始标定**：单个中性姿态的 15 点坐标分量 RMSE 约 14.68 mm，原始 Jacobian 秩 12 / 参数数 14，没有参数碰到约束边界。静态姿态存在参数耦合，使用解剖长度先验和正则项约束；它不能替代多姿态标定及动态动作检查。

运行时直接遍历 SOMA 的 23 节点骨树：

- torso 缩放 Spine1、Spine2、Chest、Neck1、Neck2 的入骨段。
- upper_arm / forearm 缩放 Arm→ForeArm / ForeArm→Hand。
- thigh / shank 缩放 Leg→Shin / Shin→Foot。
- shoulder 偏移施加在 Arm，elbow 在 ForeArm，hip 在 Leg，并沿子树传播。右侧只镜像偏移 Y，偏移随根姿态旋转。
- 支持 asymmetric 分侧比例；旧的 upper_scale、lower_scale、shoulder_offset、elbow_offset 在 bone 模式不参与计算。
- 根轨迹独立应用 `root_trajectory`，初始配置为单位缩放和零偏移。
- 两个手辅助点沿 SOMA 的手部纵向局部轴生成，长度为 P2 的 0.11 m；保留 0.18 m 躯干朝向辅助点。

直接使用原始身体点，无额外 0.07 m pelvis 后移、踝部前移或旧版 Cartesian 缩放。骨架拓扑和 MJCF 哈希会校验；SMPL 标定会明确拒绝。

## 足接触与地面

`.motion` 直接读取对应脚刚体的接触标签。原始 NPZ 的 `foot_contacts [T,4]` 不含列名时，不猜测四列顺序；默认从脚点高度、速度推断接触。已知顺序时可传：

```bash
--foot-contact-order left_ankle,left_foot,right_ankle,right_foot
```

上面只是参数写法，只有数据确实采用该顺序时才能如此指定。默认推断阈值为高度 0.08 m、速度 0.2 m/s，随后五帧平滑；空中片段及楼梯动作需检查接触质量。

`--align-ground` 是可选的整段 Z 常量平移，将标定后脚点低分位对齐场景 Z=0，并保留配置的 root Z offset。默认关闭。原 PyRoki 足接触成本限制滑移、高差和倾斜，并不锁定绝对地面高度。

## 输出与验证

主输出为 `原始文件名_retargeted.npz`，包括 `base_frame_pos`、WXYZ 的 `base_frame_wxyz`、30 DOF `joint_angles`、`joint_names`、FPS、源帧编号、接触和配置元数据。`--save-targets` 另存 18 点目标。`.report.json` 检查最终拼接轨迹的胶囊穿透、软关节限位超出和四元数模长。

`--skip-existing` 比较源文件信息、标定、骨架、URDF、脚本和参数指纹。失败文件写入 `run_shard_I.json` 并返回非零退出码。

2026-09-28 最新 BVH 接入验证：

- 原 11 项测试加上 5 项 BVH/默认参数测试，合计 **16 项通过**。
- 内置标定与用户提交 YAML 完全一致；默认扫描枚举 **142,220** 个 BVH。
- `210531/jump_and_land_heavy_001__A001.bvh` 共 1463 帧、约 120 FPS；每四帧取一帧后完整生成 366 帧目标，末尾源帧编号 1460，没有按固定求解窗口截断。
- 对照仓库原始 BVH 转换函数，前 80 帧最大位置差异约 **7.88e-7 m**，旋转矩阵最大差异约 **1.02e-6**。
- 前 80 原始帧抽为 20 帧，CPU 两段 PyRoki 实测输出 20×30 关节轨迹、30 FPS；胶囊最小间距约 **19.72 mm**，该短片段未发现胶囊穿透，关节软限位最大超出约 **4.61e-5 rad**。这不代表整个数据集无碰撞。
- 另一条此前未生成 `.motion` 的 BVH `210707/choreography1_injured_L_leg_001__A005.bvh` 也通过直接读取测试。
- 全量文件仅枚举，**尚未进行全量 retarget**。

此前 NPZ / `.motion` 接口验证记录：

- 11 项测试通过：骨段比例、偏移传播/镜像、左右独立比例、旧参数不生效、转身等变性、原始/转换格式坐标一致性、global/local 旋转、接触列顺序、帧率/长度及非法输入拒绝。
- 原始 NPZ 与对应 `.motion` 的完整 150 帧目标均生成成功；两种输入的身体点最大差异约 **0.163 mm**，旋转矩阵最大差异约 **2.33e-6**，与 MJCF 偏移四位小数精度相符。
- 真实 SOMA NPZ 前 40 原始帧、每 2 帧取一帧，经 CPU PyRoki 两段求解输出 **20 帧 / 30 DOF / 15 FPS**。
- 最终拼接轨迹最大软胶囊穿透约 **6.31 mm**，3/20 帧超过 1 mm；最大关节限位超出约 **4.89e-5 rad**。

求解器沿用原脚本的权重和局部比例优化变量，每段一次全身 PyRoki；没有额外 IK 修正。当前结果是可运行的初始版本，未做大规模视觉验收或动力学可执行性验证，不能称为零碰撞。GPU 未在本次验证。

依赖同目录已有的 `batch_retarget_to_astro_p2_bone.py` 和原 `batch_retarget_to_astro_p2_from_keypoints.py`、calibration viewer 及仓库 P2/SOMA 资产。部署时保持这些文件位于已有 ProtoMotions checkout 的 `pyroki/`。

求解方式：每条选中的动作整段送入 PyRoki，只调用一次优化器，无分块、补帧或重叠插值。`--chunk-frames` 和 `--overlap` 已移除。长动作需要更多内存，失败会记录，不自动改用分块。

Bone 模式前向辅助点已统一使用源骨盆和机器人骨盆坐标系（前方 0.18 m），不再将源骨盆目标匹配至机器人胸部；共用求解器需使用本次更新版本。

新版用户标定：`human.format: soma`、`layout: soma23`、`units: m`，新增 `neck` 对应 SOMA `Neck1` 与 P2 `head_link`。默认目标数组为 19 点：原 15 点、neck、左手辅助点、右手辅助点、骨盆前向辅助点。neck 同时参与全局位置和骨盆/双肩相对向量约束；NPZ 的机器人状态仍为 30 个关节。

若 calibration_viewer 被移至独立工程，可用 `CALIBRATION_VIEWER_SRC` 指定其 src 目录；也支持当前服务器 `/data/chenguanting/workspace/calibration-viewer-soma-test/src`。

## 2026-09-29：最新标定与地面、背手约束

当前内置参数与 `soma_p2_bone_calibration.yaml` 已同步用户最新配置：
`torso=0.950070275306741`、`upper_arm=0.9994727232391987`、
`forearm=0.779078544121117`、`thigh=0.6997741356596586`、`shank=0.73`。
三组 joint_offsets 也完整更新，保留 `Neck1 -> head_link`。

使用单次整段 PyRoki 优化。以源上臂、前臂方向初始化肩和肘，其余关节从零位开始，
不再用全部关节限位的中点初始化。额外匹配骨盆坐标系中的左右手腕位置。
脚部使用 URDF 视觉网格凸包的最低点，添加世界 Z=0 地面软约束，目标余量 2 mm；
接触帧保持脚部静止并贴近地面，不再强迫脚踝和脚尖 link 原点等高。
当前场景地面为 Z=0，请搭配 `--align-ground` 使用。

报告新增 `min_sole_height_m`、`ground_penetrating_frames_1mm`、
`wrist_pelvis_rmse_m`、`wrist_front_when_target_behind_frames` 和 `quality_pass`。
`quality_pass` 要求地面及胶囊自碰撞穿透不超过 1 mm、两侧手腕局部 RMSE 均小于 8 cm。
导出成功不等于质量通过；约束为软代价，必须检查报告。
视频通过 MuJoCo 的 `mj_forward` 回放 qpos，不运行动力学或控制器。
