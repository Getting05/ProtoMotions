# Astro P2 接入说明

当前更新服务器：172.16.9.7（8 卡 4090 2 号机）；项目：/data/chenguanting/ProtoMotions。
机器人名称：`astro_p2`，别名 `p2`。现有支持 `--robot-name` 的入口可使用 `--robot-name astro_p2`。

## 接入内容

- `protomotions/robot_configs/astro_p2.py`：30 个控制关节，31 个机器人刚体，+X 朝向；默认根高度 0.645 m（用户提供的初始位置）。
- `protomotions/robot_configs/factory.py`：新增 astro_p2 / p2 注册。
- `protomotions/data/assets/astro_p2/`：完整保留用户提供的 mesh、URDF 和原始 MJCF。
- `astro_p2/mjcf/astro_p2_protomotions.xml`：由 primitive_collision 版生成的仿真适配文件，保留已有自由根关节；增加 30 个 motor 和 armature，显式声明 hinge/limited 以兼容 IsaacGym 的关节限位解析。移除旧 dm_control 不识别的 mesh content_type 和 joint actuatorfrcrange，力矩限值仍由 motor 和 ControlInfo 保留。
- `scripts/smoke_astro_p2.py`：独立检查模型、限位、配置、复位及仿真步进；不需要动作数据或策略权重。

身体映射：躯干为 `waist_roll_link`，双手为 `left/right_wrist_yaw_link`，双脚为 `left/right_ankle_roll_link`，头为 `head_link`。原始 MJCF 已将固定躯干/手掌并入这些刚体，因此不使用不存在的 torso_link/hand_link 刚体。P2 的 `head_joint` 排在第 30 个控制关节，不沿用 P1 的关节顺序。

关节上下限、力矩和速度与提供的 P2 URDF 一致（髋/膝速度 23.04 rad/s，腰 yaw 为 13.04 rad/s）。2026-10-16 按用户提供的 P2 配置更新 29 个身体关节；本仓库保留第 30 个 `head_joint` 及其原有控制参数，因为参考文件没有定义头部执行器。

| 关节组 | armature (kg·m²) | 自然频率 | 阻尼比 |
|---|---:|---:|---:|
| 髋、膝、腰 yaw（85 系） | 0.04416342683946 | 10 Hz | 2.0 |
| 肩、肘、腕 roll（50 系） | 0.0087362778817 | 10 Hz | 2.0 |
| 腕 pitch/yaw（39 系） | 0.002387232 | 12 Hz | 2.0 |
| 踝 pitch/roll、腰 pitch/roll | 2 × 0.0087362778817 | 10 Hz | 2.0 |

`stiffness = armature × (2πf)²`，`damping = 2 × 2.0 × armature × (2πf)`。上述 armature 已包含减速比，踝/腰两电机系数同时用于 stiffness、damping、armature；roll 不再额外缩小。力矩/速度上限保留参考文件的逐关节值，摩擦为零。头关节保留 `armature=0.002387`、`stiffness=9.423498282160118`、`damping=0.8998777996942602`、力矩 10 Nm、速度 20.94 rad/s。

默认姿态髋 pitch −0.312、膝 0.669、踝 pitch −0.363、肘 0.6、肩 pitch 0.2、左肩 roll 0.2、右肩 roll −0.2 rad，其余为零。依赖无第三方库的 `astro_p2_pose.py` 共享该具名姿态表。

运行时控制配置和 `astro_p2_protomotions.xml` 的 armature 同步；该 MJCF 不再带入原始左髋 roll 的被动弹簧/阻尼/摩擦，避免叠加到执行器 PD。原始导入的 URDF/MJCF 保留几何源数据身份；实际机器人使用运行时适配文件。IsaacLab 的 MJCF 内容指纹会生成新的 USD 缓存路径，无需删除运行中任务的缓存。

ProtoMotions 继续使用自身动作处理：标准 normalized PD 保留关节范围缩放，并读取新的默认姿态和 PD；BM PD 的 `effort_limit/stiffness` 缩放自动随新 PD 更新。参考文件的 `build_joint_position_action_scale` 函数体没有提供，此次不推测其算法。延迟、Dobot T–N 曲线及并联执行器依赖外部 rl_lab/menagerie_x 实现，此次数值同步不移植这些执行器。

已有训练进程及 `resolved_configs.pt` 保留各自加载的旧参数。新参数面向新建配置；用已有实验名恢复训练时仍会加载冻结配置。这里没有修改已有 checkpoint 或实验目录。

## 检查命令

MuJoCo：

```bash
cd /data/chenguanting/ProtoMotions
source ~/.venvs/protomotions-mujoco/bin/activate
PYTHONPATH=. python scripts/smoke_astro_p2.py --simulator mujoco --headless --steps 200
```

IsaacLab（GPU 编号按空闲情况调整）：

```bash
cd /data/chenguanting/ProtoMotions
source /data/chenguanting/.venvs/protomotions-isaaclab/bin/activate
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python scripts/smoke_astro_p2.py \
  --simulator isaaclab --headless --contacts --num-envs 2 --steps 100
```

IsaacGym：

```bash
cd /data/chenguanting/ProtoMotions
source ~/.venvs/protomotions-isaacgym/bin/activate
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python scripts/smoke_astro_p2.py \
  --simulator isaacgym --headless --num-envs 2 --steps 100
```

小规模检查脚本为 IsaacGym 使用 1M 接触对和 1×缓冲区，避免占用训练级缓冲区显存；不修改机器人训练默认值。此服务器版本的 IsaacGym 固定使用 GPU PhysX，不支持本脚本的 `--cpu-only`。

必须激活对应虚拟环境，使 Ninja 等工具出现在 PATH 中。IsaacLab 自动生成并缓存 USD，无需手动维护另一套模型。

## 训练边界

这是机器人模型和仿真配置接入，不包含 P2 动作重定向数据或训练好的策略。固定 PD 目标的步进测试中机器人会倒下；PASS 仅代表接口和数值检查通过，不代表平衡、行走或动作跟踪性能通过。不能将现有 G1 的动作数据或权重直接用于 P2。启动 motion tracking 训练前需要按 P2 的骨架及关节顺序生成动作文件，然后在相应训练命令中指定 `--robot-name astro_p2`。

## 2026-09-20 验证结果

最终适配文件已通过：

| 后端 | 环境数 | 控制步数 | 结果 |
|---|---:|---:|---|
| MuJoCo | 1 | 200 | 通过模型、限位、复位、有限数值步进检查 |
| IsaacGym GPU | 2 | 100 | 通过，使用检查脚本的小规模接触缓冲区 |
| IsaacLab GPU | 2 | 100 | 通过，启用全部 31 个刚体的接触传感器 |

30 个 motor、37 维 qpos、36 维 qvel 与 30 个策略动作一致。检查脚本逐关节核对 URDF 位置、力矩和速度限值；默认姿态均处于关节范围内。Python 编译检查、factory 修改的 git diff 空白检查通过。未运行策略训练，未验证 Genesis/Newton。

原始 42 个资源文件保留不变。服务器完整验证日志保存于 `docs/astro_p2_validation/`。

## 2026-10-16 参数同步验证

- `python -m unittest protomotions.tests.test_astro_p2_parameters -v`：4 项通过，覆盖 29 个参考关节、保留头关节、MJCF armature、限幅、默认姿态、动作处理和 IsaacLab 执行器参数传递。
- MuJoCo：1 环境、200 控制步，`P2_SMOKE_PASS`。
- IsaacLab：CPU PhysX、2 环境、接触传感器、100 控制步；从 PhysX 读取全部关节的 stiffness/damping/armature/力矩/速度上限逐项核对。CPU 检查用于避开正在训练的 8 张 GPU；没有运行新的训练或验证策略质量。
- IsaacLab 日志有导入器 motor drive 提示和接触材料索引警告；执行器在初始化时写入 PD，实际 PhysX 属性检查及有限数值步进均须通过。

本次日志与修改前备份：`work/p2_parameters_20261016/`。
