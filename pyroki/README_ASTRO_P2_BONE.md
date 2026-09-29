# Astro P2：直接从完整 SMPL 动作做 bone retarget

服务器脚本：`/data/chenguanting/ProtoMotions/pyroki/batch_retarget_to_astro_p2_bone.py`。

脚本新增，原有 `batch_retarget_to_astro_p2_from_keypoints.py` 保持不变。新脚本直接复用它的 PyRoki 全身轨迹优化、关节限制、速度约束、足接触和软胶囊自碰撞成本，不增加第二阶段 IK。

## 直接运行

```bash
cd /data/chenguanting/ProtoMotions
~/.venvs/astro-p2-retarget/bin/python pyroki/batch_retarget_to_astro_p2_bone.py \
  --input /data/chenguanting/datasets/AMASS/raw_SMPL+H/DFaust_67/50026/50026_shake_shoulders_poses.npz \
  --output-dir /data/chenguanting/workspace/calibration-viewer-run/bone_output \
  --calibration-config /data/chenguanting/workspace/calibration-viewer-run/astro_p2_calibration.yaml \
  --smpl-rest-pose /data/chenguanting/workspace/calibration-viewer-run/smpl_neutral_viewer.pkl \
  --align-ground --save-targets --skip-existing
```

`--input` 也可以传一个目录，递归处理 `.npz/.pkl/.npy`，保留相对目录，排除 AMASS 的 `shape.npz`。只加载可信的 pickle/npy 输入。批量目录中的非动作文件会明确报错并记入失败清单。

默认处理动作全部帧；`--subsample-factor N` 按 N 抽帧并相应调整 FPS。`--start-frame`、`--max-frames` 是显式裁剪选项。长动作默认每 120 帧求解一次，相邻段重叠 15 帧；位置/关节角线性融合，根四元数用 SLERP，末段补帧后移除，不截断长动作。每段仍只有原有的一次全身求解。

## 输入与体型约定

输入字典支持：

- `poses [T,72]`：SMPL；`poses [T,156]`：SMPL-H，使用前 66 维身体姿态；也支持身体姿态 `poses [T,66]`。
- 或 `global_orient [T,3]` 与 `body_pose [T,63或69]`。
- `trans` 或 `transl [T,3]`：米。姿态为弧度轴角。
- `mocap_framerate`、`mocap_frame_rate` 或 `fps`；缺失时必须传 `--input-fps`。
- 可选 `left_foot_contacts` / `right_foot_contacts`，形状 `[T]`、`[T,1]` 或 `[T,2]`，值在 `[0,1]`。

**默认体型是本次标定配套的 neutral 骨架** `smpl_neutral_viewer.pkl`，从其中的 24 个关节进行 FK。源动作的 betas、DMPL 和 SMPL-H 手指姿态不会应用。这保证输入骨架与标定参考体型一致，但并非恢复原演员形状的 SMPL 网格。无需额外安装 SMPL-X/chumpy 或加载旧 SMPL 模型。需要其他体型时，应提供该体型的 neutral 24 点骨架并重新标定。SMPL-X 165 维输入明确拒绝。

## 坐标与标定语义

- `human.axes: [z,x,y]` 定义 **neutral SMPL 模板** 到机器人 `[forward,left,up]` 的坐标变换。
- AMASS 的动态世界通常已经 Z-up。`--input-axes auto` 检测 AMASS 的 `mocap_framerate/mocap_frame_rate` 字段后保留世界 XYZ，否则使用 YAML 的 human.axes。自定义数据有不同世界坐标时，明确传 `--input-axes x,y,z`、`--input-axes z,x,y` 或 `--input-axes calibration`。不要把已经 Z-up 的 AMASS 世界再次变为 ZXY。
- 根局部姿态采用 `R_body_to_world = A_world × R_SMPL × B_template.T`，所以偏移在转身时随人体旋转。
- 直接复用 calibration viewer 的 `CalibrationParams` 与 `HumanMotion.calibrated()`：torso 只缩放 spine1/spine2/spine3/neck 骨段；上臂、前臂、大腿、小腿分别缩放；肩、肘、髋偏移沿父子链传播，右侧只镜像偏移 Y。支持 asymmetric 的分侧比例。
- bone 模式下 YAML 残留的 `upper_scale/lower_scale/shoulder_offset/elbow_offset` 不参与变换。
- `root_trajectory` 独立生效：XY 缩放相对于选中动作首帧，Z 缩放绝对高度，再加 offset。本次 YAML 是单位缩放、零偏移；不会拿腿部比例缩放全局行走距离。
- YAML 的 robot keypoint_links 用于 15 点映射；t_pose_joint_positions 是标定时的参考姿态，不是每帧机器人关节命令。求解器保留原脚本默认初始化及优化变量，包括局部 alignment 的比例变量。
- 两只手辅助点沿 FK 的腕→手方向重建为 0.11 m；躯干辅助点沿用原 P2 求解器的 0.18 m 朝前提示。

原始 AMASS 文件的地面未必是 Z=0；原 PyRoki 足接触成本仅约束滑移、高差及倾斜，不锁定绝对地面高度。示例命令显式加 `--align-ground`，将标定后脚点 Z 的第 2 百分位对齐到零，保留 YAML 的 root Z offset，并在 metadata 记录 `ground_alignment_m`。这是整条轨迹的常量平移；不改变 XY 或帧间根位移。默认不开启，以保留严格的 YAML 根轨迹结果。已经对齐的动作、空中片段、台阶动作应自行选择是否使用。

未提供足接触时，通过源骨架脚点高度与速度估计，再做五帧平滑。默认高度阈值 0.08 m、速度阈值 0.2 m/s；地面取脚点 Z 的第 2 百分位，可用 `--ground-height` 明确指定。空中动作、楼梯或不平地面需要检查接触标签。

## 输出及验证

每个动作输出 `原始文件名_retargeted.npz`，主要字段：

- `base_frame_pos [T,3]`、`base_frame_wxyz [T,4]`（WXYZ）、`joint_angles [T,30]`、`joint_names`。
- `fps/source_fps/subsample_factor/source_frames`、左右足接触。
- `metadata_json`：标定全文、参数、源帧、坐标和体型策略；`signature` 用于可靠跳过。

`--save-targets` 另存 18 点校准目标；`--prepare-only` 只进行 FK 和标定，不运行 IK。`--skip-existing` 比对配置、实现、参考骨架、URDF、运行参数及源文件路径/大小/mtime，参数变更不会误跳过旧结果。支持 `--num-shards N --shard-index I` 文件级分片。

`.report.json` 在**最终拼接轨迹**上检查胶囊碰撞、关节限位误差和四元数模长。轨迹融合未增加第二次优化，段间速度、脚滑和真实网格碰撞仍需评估；软约束不等于零碰撞或物理可执行性。

2026-09-28 验证：10 项数值测试通过（含精确骨链、偏移随转身旋转、AMASS 坐标、独立根轨迹、可选地面对齐、长序列覆盖和四元数拼接）。实际 SMPL-H AMASS `50026_shake_shoulders_poses.npz`，前 40 原始帧、每 2 帧取一帧，20 帧/30 DOF，CPU 单阶段 PyRoki 两段求解完成；开启及关闭地面对齐均完成验证。完整 270 帧的目标生成与重复运行跳过也已通过。最大胶囊穿透约 1.36 mm，2/20 帧超过 1 mm；最大关节限位超出约 4.09e-5 rad。尚未做大批量动作视觉或动力学验收。

本文件和脚本放在已有 ProtoMotions checkout 的 `pyroki/` 下使用；依赖同仓库原 retarget 脚本、calibration viewer、P2 资产及已配置的 PyRoki 环境。CPU 已验证；`--device cuda` 需要可用的 JAX CUDA 环境，本次未验证 GPU。
