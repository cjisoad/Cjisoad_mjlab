# 鲁棒起立过渡教师（bridge v2）

任务 `pedipulation_bridge_t` 的唯一目标：从三足教师交出的真实状态完成参考引导的起立，并进入冻结双足教师能稳定接管的状态。2026-10-07 起采用独立起立参考；删除过渡中的线速度和角速度命令跟踪。中段 FR 跟踪允许偏差，终点仍需满足实际交接条件。

源教师：`outputs/new_teachers_review_20261006/quadruped/checkpoints/model_9900.pt`。
双足教师：`outputs/new_teachers_review_20261006/biped/checkpoints/model_14600.pt`。
两个教师固定，不进入 PPO。只更新过渡 actor 和新 critic。

## 引导数据如何使用

`src/assets/motions/go2/tripod_to_biped_nominal.npz` 提供完整准备、重心转移和起立参考，名义起立结束于 2.5 s。参考完成后保持终点。每次接管使用独立时间，播放速度随机 0.85–1.15 倍，完全不依赖 FR 命令高度或误差。

使用机身离地高度、机身重力方向、FL/RL/RR 九个支撑关节角、FL 相对机身的锚定坐标位置、接触/卸载阶段。高度/方向为主要跟踪，支撑关节和 FL 为弱引导。允许水平位移、改变后脚落点和速度；不要求跟踪绝对 XY、yaw、参考速度或静态力矩。

接管时捕获实际高度、机身重力、关节和 FL 相对位置，在 0.3 s 内平滑衔接参考。这只改变参考目标，不设置/重置实际姿态或关节。

`bridge_target_paths.npz` 的 12 对起终点产生平滑 FR 目标。中段跟踪尺度 0.10 m，参考进度最后 30% 内线性收紧到 0.05 m。路径库的完整 IK 姿态样本用于离线审计/预览；在线弱关节参考来自名义起立数据。现有路径库约束仍存在：不是任意 XYZ 或任意初状态的动态可行性证明。

## 控制与交接

1. 源教师零命令热身 1.5–2 s；LOW 目标保持 1–1.5 s，再平滑向上。静止/移动各 0.5；移动 vx=0.05–0.15 m/s、vy=0、wz=−0.15–0.15 rad/s。
2. FR 命令达到随机开始边界 dz=0.25–0.28 m 后，物理健康即接管。取消入口 FR<0.1 m、FL 接触和近期后脚支撑筛选；不把需要恢复的健康偏差筛掉。越界后控制锁定，直到验证成功/失败。
3. 接管保留实际速度、姿态、关节、动作历史、延迟和物理参数。默认 0.5 概率在第一步过渡控制前施加一次速度增量：世界水平各轴 ±0.10 m/s，角速度各轴 ±0.20 rad/s；只改变动量。把 `initial_impulse_probability` 设为 0 可做无额外扰动基线。源自然速度仍保留。
4. 过渡速度命令为零，但无速度跟踪奖励；实际速度观测保留。参考走完且机身高于 0.44 m、重力 x<−0.8 且 |z|<0.35、角速度范数<2 rad/s、平移速度范数<0.5 m/s、前脚离地、后脚最近 0.3 s 都有支撑、FR 误差<0.1 m、关节速度范数<20 rad/s，保持 0.1 s 后交接。
5. 同一真实状态下由双足教师验证 2.5 s，默认站立速度命令 (0,0,0)。严重碰撞、基座低于 0.12 m 或非有限状态立即失败；连续 0.2 s 不满足双足检查也失败。门控范围是启动设置，需要后续接管能力校准。

源阶段失败记录源产出率，不进入过渡 PPO。过渡/双足验证失败为 −20，验证成功为 +20，反馈加到最后一个过渡步。验证窗口视为结果测量，**不额外对验证时长折扣**；过渡步骤内 GAE 仍 gamma=0.99。参考完成或高度达标本身不算成功。

## 奖励

逐步奖励只对过渡控制计入，控制周期 0.02 s。各项乘权重再乘周期。

| 项 | 权重 | 定义 |
|---|---:|---|
| height | 3.0 | 高度误差指数奖励，尺度0.08 m |
| orientation | 5.0 | 重力方向误差指数奖励，尺度0.35 |
| support_pose | 0.5 | 九个支撑关节误差平方均值的指数奖励，尺度0.5 rad，排除FR |
| fl_position | 0.5 | FL相对机身锚定位置误差指数奖励，尺度0.10 m |
| fr_tracking | 0.5 | FR位置误差指数奖励，尺度0.10→0.05 m |
| support | 0.5 | 两后脚近期接触加阶段FL接触，按支撑权重归一化 |
| rise_progress | 2.0 | 实际接近终点高度/重力的分数增量，除以dt以避免重复缩放；停留无增量 |
| action_change | −0.025 | 相邻原始动作差平方均值 |
| slip | −0.25 | 接触脚水平速度平方之和 |
| front_contact | −0.5 | FL接触乘参考卸载系数，加FR接触 |
| joint_limits | −10.0 | 超出软限位的距离之和 |
| torque | −0.02 | 执行器力矩按23.7/35.55 Nm上限归一化后平方均值 |
| collision | −2.0 | nonfoot_contact中超过1 N的接触数量 |

## 观测、网络与兼容性

过渡 actor73：原教师51 + 22个引导值，依次为进度1、实际高度1、参考高度1、参考重力3、参考支撑关节9（相对公共默认角）、FL相对位置3、FL卸载1、FR实际误差3。critic117 保留源 critic 的精确速度及其他特权通道，共享 actor 的原48噪声值和新增22值。

冻结专家每次查询仍使用原 actor51，按自己的默认角转换关节/上一动作，输出换算为同一个公共 PD 目标。过渡 actor 均值网络从源教师扩展：首层新增22列为零，其余参数复制，初始化输出与源教师一致；新增通道随后可由 PPO 学习。critic 从头初始化。

bridge_contract v2 保存观测、教师/参考/路径/物理哈希、控制/参考/扰动/交接参数、奖励配置及终端语义。**旧 bridge v1 actor51 模型不能恢复训练或播放；应启动新 run。** 旧 `pedipulation_transition_t`、旧资产和所有旧 checkpoint 保留。

## 运行

```bash
PYTHONPATH=.:tests MPLCONFIGDIR=/tmp/bridge-mpl OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_bridge*.py' -v
PYTHONPATH=. python scripts/smoke_bridge.py --device cpu --num-envs 4 --cycles 1 --ppo \
  --output outputs/bridge_rise_guidance_cpu
PYTHONPATH=. python scripts/smoke_bridge.py --device cuda:0 --num-envs 32 --cycles 1 --ppo \
  --output outputs/bridge_rise_guidance_gpu
PYTHONPATH=. python scripts/train.py pedipulation_bridge_t \
  --env.scene.num-envs 1024 --agent.max-iterations 1000 \
  --agent.save-interval 10 --agent.run-name robust_rise_guidance_v2
PYTHONPATH=. python scripts/play.py pedipulation_bridge_t \
  --num-envs 6 --device cuda:0 --viewer native --checkpoint-file <新的v2模型路径>
```

`bridge_metrics.jsonl` 记录静止/移动源产出、双足接管尝试/成功数、失败归因、FR RMS、最高高度、直立比例，并记录接管初始线/角速度、FR误差和按初速度分组的成功数。CPU/GPU smoke 证明运行/训练接口可执行，不代表已经学习成功或证明鲁棒起立。

设计：`docs/superpowers/specs/2026-10-07-robust-rise-guidance-design.md`。

## 本次实现检查（2026-10-07）

Bridge 测试 62 项通过；原教师/旧过渡回归检查 47 项完成，其中 1 项跳过。CPU 4 个环境和 GPU 32 个环境均完成一次实际 PPO 更新：actor 更新、冻结教师不变、checkpoint 保存恢复及不兼容配置拒绝检查均通过。独立代码审查发现的自碰撞传感器覆盖和静态终点评估标签问题已修正，并补充了非有限速度状态失败检查和传感器配置契约。

上述 smoke 使用初始化策略，仅各做一次更新；CPU/GPU 起立成功数均为 0。这验证的是运行流程，尚未得到训练收敛的鲁棒起立策略。另一个静态终点检查在零初速度、4 个不同路径终点下由冻结双足教师保持 2.5 s，结果 4/4 成功；它只检查终点与验证协议，不能代表动态过渡成功。

完整记录：`docs/superpowers/robust-rise-guidance-verification-2026-10-07.md`。本次未启动长时间训练。
