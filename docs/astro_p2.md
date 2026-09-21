# Astro P2 接入说明

服务器：172.16.9.8；项目：/data/chenguanting/ProtoMotions。
机器人名称：`astro_p2`，别名 `p2`。现有支持 `--robot-name` 的入口可使用 `--robot-name astro_p2`。

## 接入内容

- `protomotions/robot_configs/astro_p2.py`：30 个控制关节，31 个机器人刚体，+X 朝向；默认根高度 0.6252 m（默认姿态足底距平地约 2 mm）。
- `protomotions/robot_configs/factory.py`：新增 astro_p2 / p2 注册。
- `protomotions/data/assets/astro_p2/`：完整保留用户提供的 mesh、URDF 和原始 MJCF。
- `astro_p2/mjcf/astro_p2_protomotions.xml`：由 primitive_collision 版生成的仿真适配文件，保留已有自由根关节；增加 30 个 motor 和 armature，显式声明 hinge/limited 以兼容 IsaacGym 的关节限位解析。移除旧 dm_control 不识别的 mesh content_type 和 joint actuatorfrcrange，力矩限值仍由 motor 和 ControlInfo 保留。
- `scripts/smoke_astro_p2.py`：独立检查模型、限位、配置、复位及仿真步进；不需要动作数据或策略权重。

身体映射：躯干为 `waist_roll_link`，双手为 `left/right_wrist_yaw_link`，双脚为 `left/right_ankle_roll_link`，头为 `head_link`。原始 MJCF 已将固定躯干/手掌并入这些刚体，因此不使用不存在的 torso_link/hand_link 刚体。P2 的 `head_joint` 排在第 30 个控制关节，不沿用 P1 的关节顺序。

关节上下限、力矩和速度来自提供的 P2 URDF（例如髋/膝速度 23.04 rad/s，腰 yaw 为 13.04 rad/s）。PD 和 armature 初值参考 menagerie_x 的 docs/robots/astro.md 中 Astro/P1 参数表，并非经过 P2 实机验证的参数。默认姿态髋 -0.312、膝 0.669、踝 pitch -0.357 rad，使脚底保持水平。

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
source ~/.venvs/protomotions-isaaclab/bin/activate
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
