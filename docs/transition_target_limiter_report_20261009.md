# transition_t FR 目标限速交付记录

本次在 teacher 训练、评估和 MuJoCo actor75 三教师播放中加入相同的 FR
关节目标速度/加速度限制，最新代码将用于新服务器的入口课程续训。

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
