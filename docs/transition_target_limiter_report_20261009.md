# transition_t FR 目标限速交付记录

本次在 teacher 训练、评估和 MuJoCo actor75 三教师播放中加入相同的 FR
关节目标速度/加速度限制，最新代码已部署新服务器并开始真实入口课程续训。

## 实现与配置

- `PedipulationPositionAction` 可选使用 `JointTargetLimiter`。每 0.005 s 子步
  限制目标轨迹；FR hip/thigh/calf 上限 2 rad/s、30 rad/s²，其余九关节直通。
- 限速器保存已下发位置和速度，反向时连续刹车；物理 reset 从最终初始化
  关节姿态开始，教师切换保留限速器状态。
- 观察量、动作变化奖励、三帧动作经理历史记录实际执行的公共动作。教师延迟
  使用独立的上一请求 buffer，避免在恒定输入时每个控制周期反复刹车。
- 新入口状态库 schema 2 记录限速器位置/速度，校验动作零点/scale 和限速
  配置；schema 1 仅支持历史检查，正式训练需重新采集。
- checkpoint contract 3 / entry recipe 4 包含限速配置、状态库、物理、随机化、
  实际入口 split/比例和参考采样设置。恢复配置不一致时在加载权重前拒绝。
- 旧 `model_30000.pt` 仅通过明确迁移路径加载网络权重；课程 warm start
  保留 actor/critic/normalizer，优化器及课程时钟重新开始。

## 本地证据

聚焦最终回归记录 `outputs/transition_target_limiter_20261009/regression_final.log`：
**179 passed, 32 subtests passed**。覆盖限速反向/刹车、稳定落点、部分复位、
恒定请求延迟、真实入口恢复、独立留出评估、checkpoint 不兼容拒绝与恢复。
本次未声明全仓测试通过；原工作区 `test_keyboard_teacher.py` 的 3 条几何缓存
旧断言失败记录仍见入口课程报告。

已提交代码的隔离副本补齐历史 bridge 模型依赖后，额外覆盖 bridge 契约与
冻结教师测试，结果为 **196 passed, 60 subtests passed**。

CPU 8 环境从旧 model 30000 warm start 做两次更新，再严格恢复至第 3 次。
网络参数保持有限，最大 actor 权重变化 0.001037；Adam 步数 40→60，环境
步数 48→72，自适应学习率恢复为 1e-5，TensorBoard 标量均有限。
详情：`outputs/transition_target_limiter_20261009/ppo_verification_final.json`。

初始所有关节限速引发旧源教师失稳。同种子 64 环境、8 s 的诊断中，未限制 /
只限制 FR / 只限制支撑腿 6/120 / 全部限制分别有 56 / 55 / 14 / 9 个环境
存活。因此最终只约束 FR，不把支撑腿动态一起改掉。这个对照用于定位集成
问题，样本量不足以证明 FR 限速改善了策略性能。

修复延迟语义后的最终采集 smoke：64 环境 × 4 批次，6,128 状态、254 条轨迹，
256 条运行轨迹中 228 条撑满 8 s。实际速度最高 1.10 m/s，FR 首帧误差最高
0.528 m。level 0 训练/留出轨迹数 65/17；不足 64 条独立留出，不能晋级。
该库仅用于 CPU PPO 链路验证，正式训练重新扩大采集。

旧 model 30000 加 FR 限速的真实 keyboard/FSM 播放：2.22 s 切入过渡，12.22 s
超时；没有物理失败或跟踪复位，未连续站稳 1 s，双足教师执行时间为零。
进入过渡时 qpos/qvel/公共动作/限速器位置与速度的差值全为零。2444 个物理
子步测得目标峰值 2.00001 rad/s、30.00001 rad/s²；实测 FR 关节峰值约
6.01 rad/s、足端约 1.57 m/s。证据表明限制了下发目标，不能保证实际足端
速度或可靠交接。详情：`outputs/transition_target_limiter_20261009/legacy_chain_fr.json`。

## 蒸馏与后续验证

Student 监督应使用限速后实际执行动作和轨迹，同时复现执行环境。Student
能学习平滑行为，但蒸馏本身不提供硬约束；未来部署时仍需保留同一限速器。
当前无真机，Go2 C++ velocity 部署不在本次修改范围。课程正式长训后的
留出性能与实际双足教师接续需要单独评估。

旧 `transition_entry_deployment_20261009.tar.gz` 是未限速历史包，已被本次
FR 限速代码/部署取代；不要使用旧包内启动脚本直接运行当前工作区。

## 正式远端启动

服务器：`root@xj-member.bitahub.com:42242`，RTX 4090。代码已推送到
`origin/wsl-validation`，训练源码提交 `66effffe06e7ee1435f039ff57088e6620b5cbc0`。
远端新目录：`/pedipulation/legmanip1/transition_entry_fr_20261009_66effff/`。
部署复用文件逐项检查 SHA256；全部 1,016 个文件与部署清单一致。
GPU 8 环境 rollout/PPO/checkpoint 检查通过后启动正式任务。

采集器使用 `loco_return_20261009/model_14999.pt`、256 环境 × 16 批次 ×
8 s，得到 **99,612 个真实状态、4,043 条轨迹**；4,096 条运行轨迹中 319 条
失败，失败后的状态不进入库。实际水平速度最大 1.508 m/s，FR 首帧误差最大
0.595 m。速度分箱全部覆盖，但 >1.2 m/s 仅 13 状态，且当前课程最高纳入
1.2 m/s、0.45 m 的入口。

| 课程 | 训练独立轨迹 | 留出独立轨迹 | 续训前站稳 / 64 |
|---|---|---|---|
| 0 | 1,055 | 269 | 35 |
| 1 | 1,497 | 399 | 33 |
| 2 | 2,295 | 596 | 18 |
| 3 | 3,234 | 809 | 12 |

状态库 SHA256：`60fb91f2e92b5f099e0201977394c56aa9cf1a9542ddff2a2633332cb1730dd9`。
正式命令由 `scripts/run_transition_entry_remote.sh` 执行：4,096 环境、目标
5,000 次 PPO 更新、每次 24 控制步；每 250 次更新评估当前课程的 64 条独立
留出轨迹，连续两个窗口站稳率 ≥80% 才晋级。每 100 次更新及评估窗口保存
checkpoint，预计最终 `run/training/model_4999.pt`，结束后自动独立评估四档。

这是从旧 `transition_t/model_30000.pt` warm start 的续训，保留网络权重及
normalizer；因为执行约束改变，优化器和课程时钟从零开始，不是随机从头训，
也不是原训练优化器的严格恢复。数据采集教师保持冻结。

正式训练 PID **1509**，主流程 PID **1455**，通过独立会话后台运行。
日志 `run/training.log`，训练产物 `run/training/`，课程 `run/training/course.jsonl`。
2026-10-09 21:06 CST 检查：`model_400.pt` 参数有限，actor 权重相对 seed
最大变化 0.01785，TensorBoard 标量有限。随后已确认生成 `model_600.pt`、
两个 PID 存活，GPU 显存约 3.9 GiB、利用率 59%。

早期结果尚未改善：level 0 站稳基线 35/64，250 更新后 20/64，500 更新后
17/64；250 更新完整参考完成 53/64，高于基线 50/64，但站稳恶化。
课程正确留在 0，严格终点保持仍为 0。不能将短期训练 reward 或完成起立
等同于可靠站稳，最终需看留出评估和真实三教师接续。

部署清单、启动参数、覆盖、完整四档基线和 250 次评估已下载至本地
`outputs/transition_target_limiter_20261009/`；`startup_verification.json` 保存
正式模型有限值和初始学习变化审计。
