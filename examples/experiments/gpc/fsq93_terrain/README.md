# SuperMimic FSQ93 · GPC Stage 3 地形任务

这套配置使用 5090 1 号机上 ProtoMotions 的 FSQ93 自回归 Stage 2 step 20000 和对应的冻结 SuperMimic P2 decoder，在同机现有的 MJLab/MuJoCo Warp 运行时训练地形条件适配器。论文依据为 GPC 附录 D.2、图 16 和表 12。机器人用 50 Hz policy 和 200 Hz 仿真。

| 配置 | 场景 | 目标 |
| --- | --- | --- |
| `platform_2x2.yaml` | 4 块 2.00 m 方平台，间隙 0.55 m，台面高 0.82 m | 相邻平台中心 |
| `platform_giant.yaml` | 5 × 6 块平台，行列尺寸与间隙按种子采样，间隙 0.70–1.70 m，相邻高度差至多 0.30 m | 随机正交相邻平台中心 |
| `barrier.yaml` | 13 m 走廊、5 根横梁，净空从 0.62–0.92 m 采样 | 单梁课程，随机选一根，从梁前到梁后 |
| `barrier_full.yaml` | 与单梁课程使用同一条走廊 | 走廊另一端，连续通过 5 根横梁 |

XY 局部目标与 33 × 33 的机器人朝向对齐高度图进入任务适配器。采样间隔 0.19 m，对应 6.08 m 的采样中心跨度。地图有相对地面高度、相对横梁底面高度和横梁占用三通道，由轻量 CNN 编码。冻结 prior 继续只读取原生的 10 × 93 历史。DoRA/FiLM 用 rank 64、alpha 128，冻结 prior 的采样约束为 top-p 0.9、temperature 1.0。

奖励为论文式 (15) 的 XY 距离指数奖励，位置系数 0.42 沿用论文表 11 的 Target 配置。目标半径 0.30 m。平台落入间隙、走廊出界、从横梁上方越过或机器人摔倒都会结束本回合。Barrier 的场景尺寸、净空范围、2×2 的间隙和平台基准高度是针对 P2 的设置。随机几何按启动种子生成，同一进程内的并行环境共享该布局，各环境独立采样出生点和目标。图片里的横梁立柱和起终点标记是渲染辅助，仿真碰撞用横梁、平台和地面。

## 5090 1 号机运行

从已有的 `physics-drift-supermimic93-5090` 容器运行。以下命令针对平台 2×2，替换配置路径可切换其他场景。

```bash
EXP=/root/code/ProtoMotions-GPC-FSQ93-stage2/examples/experiments/gpc/fsq93_terrain
OUT=/root/code/ProtoMotions-GPC-FSQ93-stage2/output/stage3_fsq93_terrain_20261124
CUDA_VISIBLE_DEVICES=0 "$EXP/run_5090.sh" train \
  --config "$EXP/configs/platform_2x2.yaml" --output "$OUT/verify-2x2" --smoke
```

训练入口支持 `--iterations`、`--envs` 和 `--minibatch`。单梁训练的 `adapter.pt` 可用 `--warm-start` 加载到全走廊任务。全走廊从头初始化时起点距离目标约 11.4 m，距离奖励约 2e-24，因此提供近距课程。当前启动入口为单卡。冒烟测试每个配置 3 个环境、4 个控制步、1 轮 PPO，覆盖采样、回放、一次局部重置、碰撞模型与高度图对照。场景视频展示 P2 初始姿态、地形和目标位置。

```bash
"$EXP/run_5090.sh" render --configs \
  "$EXP/configs/platform_2x2.yaml" "$EXP/configs/platform_giant.yaml" \
  "$EXP/configs/barrier.yaml" --output "$OUT/preview"
```

## OneStep · 八卡 Barrier

`barrier_onestep.yaml` 接入 PD 平滑后的 Physics-Drift OneStep prior
`step-060000`。每卡在独立的 MJLab 仿真中按 `terrain.seed + rank` 生成横梁净空，
卡内的环境共用布局并独立采样初始状态与目标。DDP 同步 PPO 的 actor 和 critic。
PPO 缓存保存原生历史、根位姿、目标和冻结 prior 的 logits，更新时按小批次重建高度图。

5090 1 号机的六组压测使用相同的 32 步 rollout 和 2 个 PPO epoch，每组预热一轮后
测第二轮。每卡 3072 环境、minibatch 8192 的实测吞吐为 27,521 步/秒，
八卡峰值为单卡 21.25 GiB。以下命令在 5090 的仿真容器内启动这组配置。

```bash
EXP=/root/code/ProtoMotions-GPC-FSQ93-stage2/examples/experiments/gpc/fsq93_terrain
"$EXP/run_onestep_5090.sh" --envs 3072 --minibatch 8192 --output /root/code/physics-drift-gpc-stage3-onestep-20261109/runs/barrier-onestep
```

`run_onestep_5090.sh` 启用 5090 1 号机的 NUMA 绑核。其他主机可直接用
`python -m torch.distributed.run --standalone --nproc-per-node=8 --module fsq93_terrain.onestep`，
在 `PYTHONPATH` 上配置本目录的父目录和 Physics-Drift OneStep 仓库的 `src`，
并使用 `--config`、`--output` 以及资产路径选项。启动需要对应的 OneStep prior、
原生 93D tracker、含历史的 reset cases 和 P2 MJCF。
