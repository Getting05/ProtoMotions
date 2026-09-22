# 恢复单次 PyRoki 的 P2 手腕修复

2026-09-22，服务器仓库 `/data/chenguanting/ProtoMotions`。

## 当前管线

SMPL 关键点 → PyRoki 全身 IK（含手部辅助点修复与自碰撞项）→ 原有 ProtoMotions 格式转换 → MotionLib 打包。

没有额外手臂 IK。`scripts/retarget_amass_to_robot.sh` 已恢复原样，git diff 为空。下列额外阶段文件已移出仓库，保存在 `/data/chenguanting/workspace/astro_p2_retarget/retired_arm_refinement_20260922/`，可恢复：

- `refine_astro_p2_arms.py`
- `test_p2_arm_refinement.py`
- `p2_anatomical_arm_refinement.md`

旧试验数据和视频未删除。此前 P2 默认姿态 PD 中心修改保留，URDF/MJCF 和真实限位未修改。

## 修改位置

`pyroki/batch_retarget_to_astro_p2_from_keypoints.py`：

1. 新增 `p2_smpl_hand_aux`：用 canonical SMPL 的真实 wrist→hand 局部偏移方向构造 P2 手掌中心辅助点。左偏移 (-0.0149,0.084,-0.0082)，右偏移 (-0.0103,-0.0846,-0.0061)，归一化后旋转到世界系；P2 长度为 0.11 m。只在 P2 载入层重建两个手部辅助点，不修改共享关键点缓存。
2. 保留原 PyRoki 肩肘关键点、缩放、局部/全局对齐权重、rest 和平滑项。没有添加上一版掌面法线约束，也不固定身体再单独求解手臂。
3. 用 PyRoki `RobotCollision.from_urdf` 构建碰撞模型，启用原生 `pk.costs.self_collision_cost`，weight=20、margin=0.005 m。
4. 求解后调用同一个 PyRoki 碰撞模型的 `compute_self_collision_distance`，输出 `.collision.json`。渲染器只用于显示，不提供碰撞验证结果或二次优化。
5. 新输出带 `retarget_version=p2-pyroki-wrist-v3`。后续 `--skip-existing` 只跳过当前版本；旧版本会重新求解，避免修复被旧缓存绕过。本次没有在旧训练输出目录执行批量重生成。

## 碰撞模型与限制

PyRoki 为每个 URDF link 拟合一个包围胶囊，并非精确三角网格检测。P2 含无碰撞几何的中间 yaw/pitch/fixed links：排除这些零尺寸占位体，沿 URDF 父链识别相邻的实际碰撞连杆；另排除腰与近端髋部在骨盆处的结构性包围胶囊重叠。共保留 118 个有效碰撞对，包括手腕—髋部、双手、左右脚等。

碰撞项是软约束，不保证无穿透。结果如下（全部来自 PyRoki，阈值为超过 1 mm 穿透）：

| 原 rank0 动作 | 帧数 | 最小有符号距离 | 超过 1 mm 穿透的帧数 |
|---|---:|---:|---:|
| motion0 | 450 | +11.41 mm | 0 |
| motion638 | 450 | -6.09 mm | 23 |
| motion190 | 86 | -1.92 mm | 4 |

motion638 的残余为左右脚胶囊；motion190 为左髋—左肘与左右脚胶囊。不能把这些结果称为零自碰撞，后续如需消除残余，应继续在 PyRoki 中调整碰撞权重/几何近似，而不是增加外部 IK。

## 验证与输出

- `test_action_functions.py` 与 `test_p2_pyroki_wrist.py` 共 26 项测试通过。包含辅助点左右方向、旋转等变性、其余关键点不变、原生碰撞对、二次 IK 已移除及 PD 回归。
- shell 语法和 `git diff --check` 通过。
- 三段完整时序通过 PyRoki 求解并转换、导出对比视频。视频列顺序：原 SMPL / 旧 PyRoki P2 / 新 PyRoki P2；不是上一版二次 IK 的结果。
- 源、目标动作按原 rank0 编号对应，保留原有前 450 帧截取规则。
- 输出目录：`/data/chenguanting/workspace/astro_p2_retarget/pyroki_wrist_v3_20260922/`，含 `retargeted/`、`motions/`、`videos/` 和 `solve.log`。
- 没有替换完整训练库、shards 或 checkpoints，也没有启动训练。原转换器的高度后处理未改，本次不宣称解决脚底穿地。
