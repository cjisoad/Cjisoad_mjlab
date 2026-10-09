# transition_t：Go2 单次起立过渡跟踪

注册任务：`Transition-Go2-v0`。基于本仓库 `src/tasks/tracking/` 的 BeyondMimic 实现，训练 `loco_pedipulation_t → transition_t → pedipulation_t` 的起立方向。任务独立于已有的 `pedipulation_transition_t` 和 `pedipulation_bridge_t`。

默认数据为仓库下的 `outputs/teacher_endpoint_rise_clearance20mm_20261008/tripod_to_biped_reference.npz`：50 Hz，351 帧，7 秒。SHA256：`1cb970ae8cfb107e25f3aca2e1d48b8cf9a2a145d6fac25c9aa67076c3bdd0b6`。允许显式替换同 schema 的参考文件；checkpoint 会检查其哈希。数据按关节/身体名字映射，拒绝缺字段、非法形状、非有限值及非单位四元数。

这是运动学参考，数据审计给出的动态平衡残差最大约 13.43 N、3.20 N·m。真实仿真跟踪、鲁棒性、两端教师交接需要训练后验证。当前轨迹没有反向落地动作；键盘三教师 FSM 与真实入口课程是独立入口，见下文。

## 保留的 BeyondMimic 结构

- 当前参考关节位置/速度、参考 anchor 相对位置/6D 姿态、实际机身线速度/角速度、关节位置/速度和上一动作：actor **75 维**。
- critic 增加 base_link 和四条腿 thigh/calf/foot 共 13 个身体的相对位置/姿态：**192 维**。
- canonical FL/FR/RL/RR、每条腿 hip/thigh/calf，共 **12 维**位置偏移动作，`q_requested = loco_default_angles + 0.25 * action`，关闭过渡策略的动作延迟。FR 目标经速度/加速度限制后再进入 PD。
- 六项原跟踪奖励及权重不变；位置核宽按 Go2 设为 0.15 m，姿态 0.4 rad；保留动作变化、软关节限位、自碰撞惩罚。
- 25% reset 从首帧开始，其余采用原自适应失败帧采样。每次训练 reset 都按配置直接添加机身与关节扰动。
- 普通 PPO，actor/critic 均为 ELU `[256, 128, 128]`，开启观测归一化。24 步 rollout、lr 3e-4、初始 std 0.5、gamma 0.99、lambda 0.95、clip 0.2、entropy 0.005、5 epochs、4 minibatches、adaptive KL 0.01。网络容量是四足 12 动作的起点，需学习曲线和消融验证。

## 真实初始状态随机化

所有范围在 `env_cfg.py` 的 `TransitionCommandCfg` 中配置。随机化改变真实 root/joint 状态，参考目标保持原样。

| 项目 | 默认范围 |
|---|---|
| 根位置 x/y，z | ±0.02 m，±0.01 m |
| 根姿态 roll/pitch，yaw | ±0.08 rad，±0.15 rad |
| 根线速度 x/y，z | ±0.20 m/s，±0.10 m/s |
| 根角速度三轴 | ±0.30 rad/s |
| 各关节位置 | 参考关节角上加 ±0.1 rad，裁剪到软限位 |

随机化方式与 BeyondMimic 相同：在参考根位姿和速度上加均匀扰动，在参考关节角上加均匀噪声，按机器人软关节限位裁剪后直接写入仿真。关节速度保留参考值。足端位置随根姿态和关节角变化，不再指定独立足端位置范围，也不固定支撑脚的初始高度。已删除 IK、可达性重试和 reset 前的 CPU 逐候选碰撞筛选；接触演化由训练中的物理仿真处理。

机器人采用 standing Go2 几何/Ideal PD，名义 Kp=40、Kd=1，hip/thigh effort=23.7 N·m、calf=35.55 N·m；动作零点和 armature 对齐源 loco 教师（hip/thigh 0.01、calf 0.02）。软关节限位设为真实限位的 0.98，以容纳参考中接近限位的姿态，避免 reset 静默裁剪。Ideal PD 执行器限制输出力矩，没有额外实施转速曲线/硬速度裁剪。

保留小幅 startup 随机化：base COM ±0.01 m、foot friction 0.6–1.2、encoder bias ±0.01 rad。关闭继承的运行时 push；标准训练入口没有额外恢复奖励，真实入口课程通过独立脚本启用。play/evaluation 默认关闭 startup 随机化和观测噪声，以单独测量初态鲁棒性。

## FR 关节目标限速

Teacher 训练、评估和 actor75 键盘三教师播放共用 `JointTargetLimiter`：每个
0.005 s 物理子步，限制 FR hip/thigh/calf 的**目标速度为 2 rad/s，目标加速度为
30 rad/s²**。反向动作先连续刹车再反向，教师切换保留限速器的位置和速度状态。
其余九个关节保持原始目标下发；对旧三足教师给支撑腿加 6 rad/s、120 rad/s²
限制的同种子对照明显失稳，因此本项只约束 FR。参数在 `env_cfg.py` 中配置。

模拟教师延迟保留上一条策略请求；actor 的上一动作、动作变化奖励和经理历史
使用最后实际下发的公共坐标目标，避免策略观察到未执行动作。新状态库
schema 2 保存限速器位置/速度，并校验 action 零点与 scale。旧 schema 1
状态库需重新采集。checkpoint 契约版本 3、入口 recipe 版本 4 记录这些设置；
修改限速配置或入口采样比例后，原 checkpoint 不允许作为同配置断点恢复。

旧 `model_30000.pt` 是未限速策略。课程 warm start 显式迁移其
actor/critic/normalizer，使用新优化器和课程时钟；普通加载仍拒绝旧契约。
仅在仿真中测试旧模型加限速时，键盘/入口评估需传
`--allow-legacy-transition-limiter`，输出会标记为旧策略实验。给旧模型加限速
并不代表它已经学会补偿，仍需续训及留出评估。

这些参数约束的是 PD 的关节目标，实测关节/足端速度还受惯性和接触影响。
之后蒸馏 student 应使用限速后执行动作与轨迹；蒸馏可以学习这种行为，硬边界
还需要在 student 执行端保留相同限速器。当前实现范围是 teacher 与 MuJoCo，
没有真机验证。

## 时间与切换接口

控制周期 0.02 s，物理步长 0.005 s，episode 8 s。播完参考后保持末帧，目标速度归零；不会在参考末尾重采样或瞬移。PPO runner 禁止仅随机 episode 计数而不匹配物理状态。

跟踪误差终止采用 anchor/四足高度误差 >0.15 m、完整四元数角误差 >0.8 rad。reset 后 0.2 s 内暂缓这些误差终止；计时按实际仿真步数。非有限状态及基座/头部严重触地立即失败。

切入策略时，可在同一机器人环境中调用：

```python
term = env.command_manager.get_term("motion")
term.start_from_current_state(env_ids)  # env_ids: device 上的 LongTensor
```

该接口将参考首帧的水平位置/yaw 对齐当前机身，同时旋转世界系目标速度；保留高度、倾角、关节误差。它不写 root/joint 状态，不清除速度、上一动作或执行器历史，也不重置物理 reset 宽限期。调用方负责从正确的观测和动作接口运行已训练策略，并在完成终点保持后交给站立教师。最新三教师键盘接入及实际回放结果见下文。

## 真实入口课程续训

当问题来自三足教师交接时的 FR 偏差或过大的初速度，可以用冻结的
`outputs/loco_return_20261009/checkpoints/model_14999.pt` 采集物理状态，再从
`transition_t/model_30000.pt` warm start。采集器保存根状态、关节状态、三帧公共动作历史、实际水平速度、FR 相对参考首帧误差和完整轨迹 ID；训练/留出按轨迹划分，不能把相邻帧当成独立样本。

```bash
python scripts/collect_transition_entries.py \
  --device cuda:0 --num-envs 256 --batches 16 --seconds 8 \
  --teacher-checkpoint outputs/loco_return_20261009/checkpoints/model_14999.pt \
  --transition-checkpoint logs/rsl_rl/transition_t/<run>/model_30000.pt \
  --output outputs/transition_entry_course_20261009/entries.npz

python scripts/train_transition_entry.py \
  --entry-bank outputs/transition_entry_course_20261009/entries.npz \
  --checkpoint logs/rsl_rl/transition_t/<run>/model_30000.pt \
  --device cuda:0 --num-envs 4096 --iterations 5000 \
  --output logs/rsl_rl/transition_t_entry/<run>

python scripts/evaluate_transition_entry.py \
  --entry-bank outputs/transition_entry_course_20261009/entries.npz \
  --checkpoint logs/rsl_rl/transition_t_entry/<run>/model_4999.pt \
  --level all --attempts 64 --device cuda:0 \
  --output outputs/transition_entry_course_20261009/evaluation.json
```

四个难度级别逐步放宽实际速度和 FR 误差上限。每次重置从真实采样状态开始，只对齐参考首帧的 XY/yaw；高度、倾角、速度、关节状态、动作项历史、公共动作经理历史和限速器状态保留。入口不额外获得 tracking grace。每个评估窗口要求至少 64 条独立留出轨迹、站稳率至少 80%，连续两个窗口才晋级；状态库不足时只做运行检查，不晋级。checkpoint 记录状态库、教师、参考和物理哈希，`--resume` 会拒绝不匹配的状态库及采样配置，并恢复课程状态、优化器和自适应学习率。

| 课程 | 实际水平速度上限 | FR 首帧误差上限 | 真实入口 / 名义首帧 / 自适应帧 |
|---|---|---|---|
| 0 | 0.25 m/s | 0.08 m | 20% / 40% / 40% |
| 1 | 0.45 m/s | 0.14 m | 35% / 30% / 35% |
| 2 | 0.70 m/s | 0.22 m | 50% / 25% / 25% |
| 3 | 1.20 m/s | 0.45 m | 60% / 20% / 20% |

状态采集使用名义物理。训练保留过渡任务原有的小幅随机化增强：base COM ±0.01 m、encoder bias ±0.01 rad、foot friction 0.6–1.2，并保留 actor observation corruption/noise。完整事件、范围和噪声函数签名保存在 checkpoint recipe 中；恢复时配置变化会被拒绝。

本地未限速历史验证输出位于 `outputs/transition_entry_course_20261009/`。历史平衡状态库包含 12,669 个状态和 748 条轨迹；训练集 601 条轨迹，留出集 147 条，level 0 留出只有 48 条，因此 level 0 的 64 次评估不能作为晋级证据。冻结 `model_30000.pt` 的 64 次留出基线为：level 0 完整起立 53/64、站稳 33/64；level 1 为 41/64、25/64；level 2 为 24/64、9/64；level 3 为 15/64、6/64。四档严格终点保持均为 0/64。该基线说明高速度和较大 FR 偏差确实是主要失败区域，但不代表新模型已经改善；正式大规模采集和 5000 次续训等待远端服务器恢复。

## 训练与播放

在仓库根目录，使用安装有 mjlab 的 Python：

```bash
conda activate mjlab
python scripts/train.py Transition-Go2-v0 \
  --motion-file outputs/teacher_endpoint_rise_clearance20mm_20261008/tripod_to_biped_reference.npz \
  --env.scene.num-envs 4096
```

默认日志目录 `logs/rsl_rl/transition_t/`。通用训练入口要求显式传 `--motion-file`。调试时可减少环境数；本次交付没有启动正式长训。

```bash
python scripts/play.py Transition-Go2-v0 \
  --checkpoint-file logs/rsl_rl/transition_t/<run>/model_<iteration>.pt \
  --motion-file outputs/teacher_endpoint_rise_clearance20mm_20261008/tripod_to_biped_reference.npz \
  --num-envs 1
```

checkpoint 保存/检查参考 SHA、观测布局、身体/关节顺序、动作零点/scale、名义物理参数及自适应采样统计。修改这些契约后需要重新训练，不能直接混用旧任务的 checkpoint。

## 验证与评估

```bash
python scripts/smoke_transition_t.py --device cpu --num-envs 2 --ppo-updates 1
python scripts/evaluate_transition_t.py --agent zero --episodes-per-group 1 --device cpu
python scripts/evaluate_transition_t.py --agent trained \
  --checkpoint-file logs/rsl_rl/transition_t/<run>/model_<iteration>.pt \
  --episodes-per-group 100 --device cuda:0
```

Smoke 检查真实仿真的 75/192 维有限观测、奖励、PPO 参数更新、checkpoint 回读一致性和不兼容动作配置拒绝。零策略评估只验证执行入口，不代表策略成功。

评估四组分别为 nominal/base/joints/combined，都从完整首帧开始。base 组只扰动根状态，joints 组只扰动关节角，combined 组同时扰动。JSON 包含每组完整参考完成率、连续终点保持率、失效原因、姿态/高度/四足 RMS 误差及每次实际初始位姿/速度/关节角/足端偏差。

终点保持条件连续 1 s：姿态误差 <0.25 rad、高度误差 <0.05 m、足端 RMS <0.06 m、根线速度 <0.3 m/s、角速度 <0.8 rad/s，后脚均有 >5 N 地面接触，前脚无地面接触且足中心高度 >0.032 m。终止步的状态在自动 reset 前采集，避免把新 episode 的状态算进结果。

聚焦测试：

```bash
python -m pytest -q tests/test_transition_t_*.py
```

服务器 RTX 4090 已验证 CUDA rollout/PPO/checkpoint 链路。2026-10-09 按用户要求改为 BeyondMimic 直接根状态与关节角噪声 reset，旧 IK 训练停止，新版本从头训练并完成。训练后评估见下文；旧版初始化回退率诊断不适用于当前版本。

## 最新模型的键盘三教师交接

2026-10-09 的正式训练已正常结束，最终 checkpoint 为下方 `model_30000.pt`。独立首帧评估中四组初态各 20 回合全部完成起立，但严格终点保持率均为 0%；主要问题是世界系水平漂移。完整报告见 `outputs/transition_t_review_20261009/report.md`。

使用新的 `--transition-checkpoint-file` 参数选择 BeyondMimic actor75 后端。旧 `--bridge-checkpoint-file` 选择原 actor84 bridge 后端，两者互斥，不能直接混用模型。

```bash
python scripts/keyboard_teacher.py loco_pedipulation_t \
  --workspace union --num-envs 1 --device cuda:0 \
  --checkpoint-file logs/rsl_rl/loco_pedipulation_t/2026-10-08_14-46-49_loco_return_24bcf05/model_12700.pt \
  --transition-checkpoint-file logs/rsl_rl/transition_t/2026-10-09_01-13-48_transition_t_direct_4090_20261009/model_30000.pt \
  --biped-checkpoint-file outputs/new_teachers_review_20261006/biped/checkpoints/model_14600.pt \
  --transition-margin 0.02 --allow-legacy-transition-limiter
```

需要已激活安装有 mjlab 的 Python 环境和 X11 桌面。没有可用 CUDA 时用 `--device cpu`。聚焦 MuJoCo 窗口后：WASD 平移/转向，方向键调整 FR 的 XY，Q/E 调整高度，R 恢复当前教师默认目标，Backspace 完整复位，Space 暂停。

三足教师真实执行当前命令；请求及下发 FR dz 达到 `0.35 - margin`（默认 0.33 m），且 FR 跟踪误差 <0.10 m，连续 0.10 s 才切入过渡。参考仅对齐当前机身的水平位置和 yaw；真实位置、姿态、关节、速度、上一动作和执行器历史保持连续，不将机器人写入参考首帧，也不增加 reset 宽限期。入口误差会记录在 `KEYBOARD_HANDOFF` 日志中。

过渡期间键盘目标修改冻结，参考完成且姿态、高度、速度、前脚离地和双后脚支撑等站稳门槛连续满足 1 s 后，才在同一环境切入双足教师。双足目标继承参考末端的公共坐标 FR 目标，之后恢复键盘控制。过渡超过 10 s 未站稳会暂停并显示未满足条件；物理失败/跟踪终止或完整复位返回三足。

界面区分 `Standing handoff hold` 和 `Strict endpoint hold`。世界系足端 RMS <0.06 m 是严格终点保持的额外条件，作为独立诊断显示；它不阻止已经站稳的键盘教师交接。这与旧键盘入口把 FR 跟踪误差作为诊断的语义一致。交接成功不能等同于严格定点跟踪达标。

新后端沿用过渡模型的名义物理配置并执行完整 checkpoint 契约校验。三足 actor54/旧 actor51 与双足 actor51 分别使用自己的观测坐标和关节零点，动作裁剪后再重基到公共执行器坐标；两个端点教师按其训练契约执行控制步内延迟，过渡模型无延迟。

以下为加入限速前的历史结果。本地 CPU scripted 回放使用与窗口相同的键盘控制器、策略和状态机：先稳定 1 s，再按住 Q 到触发，若交接成功则继续执行双足教师至少 3 s；失败或超时立即结束。结果保存在 `outputs/keyboard_transition_t_20261009/`。早期回放曾成功交给双足并执行超过 3 s；补齐模型加载后的完整复位后，最终版视频回放在 2.22 s 切入过渡、12.22 s 超时，未交给双足。超时瞬间足端 RMS 2.51 cm，但双后脚支撑条件不满足；未出现物理/跟踪复位。两次切入的 qpos/qvel/动作/执行器历史差值均为 0。早期成功和最终版超时使用不同的复位序列，不能合并估算成功率；目前证据不足以认为交接可靠。完整记录见 `outputs/keyboard_transition_t_20261009/report.md`。交互键盘、移动中切入及不同入口姿态仍需要实际播放评估。

2026-10-09 FR 限速版的本地验证位于 `outputs/transition_target_limiter_20261009/`：
聚焦回归 **179 passed, 32 subtests passed**；CPU 8 环境完成两次 PPO 更新并
恢复至第 3 次，Adam 步数 40→60，参数变化且标量保持有限。最终采集 smoke
含 6,128 状态、254 轨迹；仅用于链路检查，level 0 留出只有 17 条独立轨迹，
最高实际速度 1.10 m/s，正式训练需更大的重新采集状态库。

旧模型加 FR 限速的单环境播放在 2.22 s 切入过渡、12.22 s 超时，没有物理失败
或跟踪复位；未连续站稳 1 s，也未进入双足教师。FR 下发目标实测峰值
2.00001 rad/s、30.00001 rad/s²；实测关节峰值仍约 6.01 rad/s，足端约
1.57 m/s。限速已执行，但不能把目标限速等同于实测速度硬限制或性能改善。
