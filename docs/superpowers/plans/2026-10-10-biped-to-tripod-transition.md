# 双足到三足 BeyondMimic 过渡实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task after user approval. Steps use checkbox syntax for tracking. Execute inline unless the user explicitly requests agent delegation.

**Goal:** 使用当前 `transition_t` 的方法，从头训练双足到 FL/RL/RR 三足的独立过渡策略，并验证指定两位冻结教师的完整链路。

**Architecture:** 新建 `src/tasks/transition_down_t/`，继承现有参考命令、观测、PPO 与入口课程；加入双足起点稳定航向对齐和三足终点条件。数据绑定本次生成的 7 秒参考、39,040 状态入口库与两位指定教师，正向任务继续使用其原契约。

**Tech Stack:** Python 3.11、MuJoCo、mjlab、PyTorch、RSL-RL、NumPy、pytest、TensorBoard；正式训练需要可用 CUDA GPU。

**Status:** 数据准备完成，计划待用户审核。以下代码/CLI 是审核后实现目标，尚未注册新任务、创建对应训练入口或启动训练。

## 输入冻结与文件分工

数据：`outputs/biped_to_tripod_preparation_20261010/biped_to_tripod_reference.npz`，SHA `fa49119da56b0eb2ac2dc49d5b65dfe310aad8033a5e15565ba5079872bb11aa`。

入口：同目录 `biped_entry_bank.npz`，SHA `ff962e464cd928e44dea76f60363867f5361a45564ad7adbd6a43632d7193058`。512 条训练、128 条留出轨迹。所有四档都有 128 条合格独立留出轨迹。参考物理和教师契约以 `manifest.json`、`baseline_configuration.json`、bank metadata 为准。

| 文件 | 职责 |
|---|---|
| 新 `src/tasks/transition_down_t/__init__.py` | 注册 `Transition-Down-Go2-v0` |
| 新 `src/tasks/transition_down_t/commands.py` | 原参考命令的薄扩展；teacher 水平航向对齐 |
| 新 `src/tasks/transition_down_t/env_cfg.py` | 原配置的方向覆盖；反向 motion 与三足保持奖励 |
| 新 `src/tasks/transition_down_t/standing.py` | 三足接触指标、好样本条件、严格定点条件 |
| 新 `src/tasks/transition_down_t/rewards.py` | 原保持公式/计时语义下的三足保持奖励 |
| 新 `src/tasks/transition_down_t/rl.py` | 原 PPO 配置、反向契约及严格加载 |
| 新 `src/tasks/transition_down_t/entry_course.py` | 原课程采样/状态恢复，改用稳定水平航向 |
| 新 `src/tasks/transition_down_t/entry_rl.py` | warm start/resume，反向入口 recipe |
| 新 `src/tasks/transition_down_t/entry_evaluation.py` | 留出入口和三足终点评估 |
| 新 `src/tasks/transition_down_t/README.md` | 实际可运行命令与模型/数据说明 |
| 新 `scripts/smoke_transition_down.py` | CPU/GPU rollout、PPO 与 checkpoint smoke |
| 新 `scripts/evaluate_transition_down.py` | 四组完整首帧评估 |
| 新 `scripts/train_transition_down_entry.py` | 真实入口课程训练、评估与保存 |
| 新 `scripts/evaluate_transition_down_entry.py` | 四档入口独立评估 |
| 新 `scripts/evaluate_transition_down_chain.py` | 双足教师→反向策略→三足教师，保留物理状态 |
| 新 `tests/test_transition_down_*.py` | 方向适配、契约、奖励、入口与链路回归 |

复用原 `transition_t.motion.TransitionMotion`、`TransitionCommand`、tracking 观测/六项奖励、`advance_standing_hold`、`EntryStateBank`、`EntryCurriculum`、`AlignedKeyboardTeacher` 和目标限速器。通用 `scripts/train.py` 已通过命令类型识别 tracking；仅在实际 smoke 证明必要时做局部兼容改动，不改用户已有训练入口文件以方便实现。

## Task 1：冻结已审核数据并建立隔离工作区

- [ ] 运行 `git status --short`，记录当前用户修改；按 using-git-worktrees 技能建立隔离工作区，连接经过 SHA 校验的审核数据。数据在 ignored outputs 下，不能假设 worktree 自动含有数据。
- [ ] 运行 `PYTHONPATH=. python outputs/biped_to_tripod_preparation_20261010/verify_bundle.py`，要求文件 SHA、两个教师 SHA、motion/schema、bank/split、视频与逆动力学检查全部通过。
- [ ] 将 `manifest.json`、`randomization_config.json`、`baseline_configuration.json` 的 SHA 写入新任务的数据契约。验证 CPU 采集物理 SHA 与新环境名义 compiled physics 完全一致。
- [ ] 确认四足顺序 FL/FR/RL/RR、关节顺序 canonical、初/末 contact [0,0,1,1] / [1,0,1,1]。拒绝 `rejected_*` 数据作为输入。

验证命令：`python -m pytest -q tests/test_transition_t_motion.py tests/test_transition_t_entry_bank.py`。预期现有 loader/schema 行为保持通过；实际状态库还需新任务的航向契约验证。

## Task 2：双足起点水平航向与状态保持

**Files:** `src/tasks/transition_down_t/commands.py`；`tests/test_transition_down_alignment.py`。

- [ ] 先写反例测试：使用同一批双足状态，Euler yaw 的 FR 误差中位数约 0.6816 m；新航向中位数约 0.03325 m。测试包括 pitch 穿过 90° 前后不翻转、纯 roll 不改变水平前向、body-Y 与 up 平行时已有确定性 fallback。
- [ ] 实现下列水平四元数函数，复用教师已有坐标定义，不另创 heading 系统：

```python
def teacher_heading_quat(quaternion):
    basis = anchor_basis(quaternion, quaternion.new_tensor([0., 0., -1.]))
    angle = torch.atan2(basis[:, 1, 0], basis[:, 0, 0])
    result = torch.zeros_like(quaternion)
    result[:, 0] = torch.cos(angle / 2)
    result[:, 3] = torch.sin(angle / 2)
    return result
```

- [ ] `DownMotionCfg` 继承 `TransitionCommandCfg`，默认 motion 指向冻结反向数据，build 返回 `DownMotion`。`DownMotion` 继承 `TransitionCommand`，保留采样/reset/末帧行为，只覆盖 `start_from_current_state` 的水平对齐：

```python
rotation = quat_mul(teacher_heading_quat(actual_q),
                    quat_inv(teacher_heading_quat(reference_q)))
translation = actual_p - env.scene.env_origins[ids] - quat_apply(rotation, reference_p)
translation[:, 2] = 0.
```

- [ ] 该接口只设置时钟、参考 rotation/translation；不写 root/joint，不清速度、经理动作历史、限速器状态，不重置 grace。测试在调用前后用 `torch.testing.assert_close(..., rtol=0., atol=0.)` 比较所有物理/执行状态。
- [ ] 验证参考世界位置、四元数、线速度和角速度同旋转变换，子集调用不影响其他环境；对齐前后的高度与倾角误差保留。

运行 `python -m pytest -q tests/test_transition_down_alignment.py`，要求反例复现且稳定航向/状态连续性测试通过。

## Task 3：三足终点指标和原保持奖励

**Files:** `standing.py`、`rewards.py`；`tests/test_transition_down_standing.py`、`tests/test_transition_down_reward.py`。

- [ ] 测试只双后脚支撑不得成功；FL/RL/RR 接触且 >5 N、FR 无接触且足中心高于 0.032 m 才满足三足支撑条件；FR 落地不得成功。姿态、高度、线/角速度和 joint norm 阈值保持原值。
- [ ] 定义 `tripod_endpoint_metrics`、`tripod_conditions` 和 `strict_tripod_endpoint_candidate`。从真实 contact sensor 和 body_link_pos 获取结果，使用当前反向参考做姿态/高度/四足世界 RMS 对照。
- [ ] 新 `SustainedTripodReward` 复用 `advance_standing_hold`，自身只持有每环境 duration、bad_samples、last_step、value；不共享环境间状态。奖励公式保持：

```python
raw = torch.where(good,
    0.2 + 0.8 * (duration / 1.0).clamp(max=1.0), 0.0)
```

- [ ] 与正向相同：参考未完成为零，前两坏样本暂停、第三坏样本清零，物理/跟踪/nonfinite 立即清零；同 step 重复调用不重复计时，子集 reset 只清相应环境。最大 raw 为 1，权重 0.1、dt0.02 后每步最多 0.002。严格世界 RMS 6 cm 条件仅用于独立诊断，不加入交接保持奖励。
- [ ] 通过真实 2 环境 reward manager 测试检查权重、dt 缩放及 episode10s；用原保持规则单元测试的计时序列验证公式一致。

运行 `python -m pytest -q tests/test_transition_down_standing.py tests/test_transition_down_reward.py tests/test_transition_t_standing_reward.py`，要求新三足语义通过且旧双足奖励回归通过。

## Task 4：独立任务、相同训练参数与方向契约

**Files:** `env_cfg.py`、`rl.py`、`__init__.py`；`tests/test_transition_down_task.py`、`tests/test_transition_down_rl.py`。

- [ ] 新环境调用 `transition_env_cfg(play=play)` 获取原物理、观测、噪声、动作、事件、跟踪奖励和终止；替换 motion 为 `DownMotionCfg`，将 holding term 换为 `SustainedTripodReward`。保持 original 六个奖励权重、std 与惩罚参数；play 保留禁用随机化/噪声的原行为。
- [ ] actor/critic 为 75/192，action12；网络 ELU [256,128,128]；公共零点和 scale0.25；transition 无延迟；FR 限速2/30只约束FR，其他关节沿用原行为。
- [ ] `DownOnPolicyRunner` 继承 `TransitionOnPolicyRunner`，`down_ppo_runner_cfg` 调用原配置，只变类名、experiment_name=`transition_down_t`、save_interval=100。其 learn 仍禁止只随机 episode count。
- [ ] `_contract` 加入新 task、direction=`biped_to_tripod`、teacher heading alignment、末端支撑、两个教师 SHA、随机化/完整奖励 recipe 和 episode10s；保留原参考/观测/动作/物理/限速器/adaptive 统计。严格加载拒绝正向模型、错误方向/航向、教师/参考/奖励变化；新反向名义模型可作为新反向课程的 weights-only warm start。
- [ ] 在 `__init__.py` 注册 `Transition-Down-Go2-v0`，使用新 runner；不修改既有 `Transition-Go2-v0` 注册。

PPO 参数需逐项对照：rollout24，lr3e-4，adaptive KL0.01，clip0.2，gamma0.99，lambda0.95，entropy0.005，epochs5，minibatches4，max_grad_norm1，init_std0.5，actor/critic 归一化。

运行 `python -m pytest -q tests/test_transition_down_task.py tests/test_transition_down_rl.py tests/test_transition_t_task.py tests/test_transition_t_rl.py`。要求配置只出现上述方向差异，75/192/12 维断言及双向 checkpoint 拒绝成立。

## Task 5：原真实入口课程的反向适配

**Files:** `entry_course.py`、`entry_rl.py`、`entry_evaluation.py`；`tests/test_transition_down_entry.py`。

- [ ] `DownEntryCommandCfg` 继承 `EntryCommandCfg`，build 返回 `DownEntryCommand`；后者保留 occupied-bin 采样/课程和物理恢复，覆盖恢复后的参考对齐为 Task 2 稳定航向。恢复 reference/body relative 目标后才计算观测。
- [ ] 首先要求 bank metadata 的 direction 和 reference_alignment 匹配，两个教师 SHA、action/limiter 和名义物理 SHA 匹配。用真实 bank 测试每档 train/holdout 非空且至少128独立留出轨迹。
- [ ] 保留所有 root/joint/twist、raw/previous action、三帧 manager history、previous joint velocity、limiter position/velocity；不把入口 state 改为名义姿态，不增加0.2s tracking grace。
- [ ] 课程范围、混合比例保持原值：0=(.25,.08;20/40/40)，1=(.45,.14;35/30/35)，2=(.70,.22;50/25/25)，3=(1.20,.45;60/20/20)。记录后两档 FR 大偏差没有实测覆盖，不能填充人造状态冒充教师样本。
- [ ] `DownEntryOnPolicyRunner` 复用原 warm start/resume 过程，recipe 明确记录三足保持、teacher heading、数据/两教师 SHA。weights-only warm start 保留 actor/critic/normalizers/distribution，重新初始化 optimizer、adaptive 统计和课程0；严格 resume 恢复优化器、KL学习率、迭代及课程，并拒绝 recipe 改变。
- [ ] 新入口评估替换三足指标、配置与 runner，保留终止前采样和独立轨迹证据。晋级仍至少64条独立留出、≥80%、连续2窗口。

运行 `python -m pytest -q tests/test_transition_down_entry.py tests/test_transition_t_entry_bank.py tests/test_transition_t_entry_course.py tests/test_transition_t_entry_rl.py`。另用32个真实入口做零策略 smoke，明确只验证数据恢复与评估链路，不报告成功策略。

## Task 6：独立入口、完整教师链路与功能验证

- [ ] 实现新 smoke/full-start/entry-training/entry-evaluation CLI，要求显式传 motion/entry bank、checkpoint和output，不写入原正向日志目录。所有解析与实际导入路径均加针对性测试。
- [ ] 完整链路入口使用两位指定冻结教师各自原观测/动作零点与延迟；BIPED→DOWN→TRIPOD 三状态在同一环境执行。DOWN 切入只对齐参考，不写姿态。末帧后按三足好样本累计1秒才切入actor54；后者继承最终公共FR目标[.12,0,.33]并继续3秒。
- [ ] 在两次交接处捕获 qpos/qvel、root/joint、真实经理动作历史、limiter状态，要求交接函数调用前后差值为0；没有执行物理步时不得清状态或偷偷 reset。最终三足教师的FR不落地。
- [ ] 测试终止状态在自动reset前捕获、相同采样点不重复计时、BIPED/正向hold不能误触发三足交接、超时/物理/跟踪失败分别记录。
- [ ] 执行新的聚焦 tests 与现有正向/教师/目标限速回归，只在有新失败或改动时扩大测试。
- [ ] 审核后运行：`python scripts/smoke_transition_down.py --device cpu --num-envs 8 --ppo-updates 2`，检查真实有限观测/奖励、非零参数更新、checkpoint round trip和错误契约拒绝；再用 `--device cuda:0` 做8环境同等 smoke。

## Task 7：正式名义训练与入口训练

- [ ] 用审核输入 SHA、源代码清单、运行配置、PID、设备信息和新的输出路径创建运行记录。GPU smoke通过才启动正式长训；当前无可用CUDA不意味着已启动或已获得远端资源。
- [ ] 从头训练，使用新的反向任务，不给旧正向模型做未经定义的 warm start。审核后可运行的名义命令：

```bash
python scripts/train.py Transition-Down-Go2-v0 \
  --motion-file outputs/biped_to_tripod_preparation_20261010/biped_to_tripod_reference.npz \
  --env.scene.num-envs 4096 --agent.seed 42 \
  --agent.max-iterations 30001 --agent.save-interval 100 \
  --agent.run-name biped_to_tripod_nominal
```

- [ ] 保存到 `logs/rsl_rl/transition_down_t/`；最后名义 checkpoint 按现有 runner 时钟为 `model_30000.pt`。检查学习曲线、有限参数、首帧组和终点三足率，失败不冒充训练成功。
- [ ] 新名义模型完成后，用其完整路径作为 `--checkpoint`，新入口训练 CLI 使用已审核 bank、motion、4096环境、seed42、iterations10000、eval_interval250、eval_attempts64，并写入新的 `logs/rsl_rl/transition_down_t_entry/`。先评估四档基线，再开启课程0。阶段2最后模型按课程新时钟为 `model_9999.pt`。
- [ ] 每100次保存，评估每250次；训练日志/模型写入 motion/bank/两教师/config/源代码SHA。严格resume与新warm-start分开记录，不沿用正向课程晋级证据。

## Task 8：最终评估与交付

- [ ] 四组完整首帧 nominal/base/joints/combined，每组至少100回合，全部从frame0，评估默认名义物理/关闭观测噪声，分别报告直接初态扰动效果；另做 startupDR/观测噪声评估并分开统计。
- [ ] 每档入口至少64不同留出轨迹；完整链路至少128不同留出轨迹，从真实双足教师状态继续执行，终点后接管三足教师3秒。报告实际速度/FR覆盖，不把课程上限当作实测范围。
- [ ] 分别报告完整下降率、三足1秒保持率、严格世界足端RMS定点保持率、链路成功率以及物理/跟踪/超时失败。联合随机化穿地样本另列结果，reference近似动态平衡限制写入报告。
- [ ] 给出名义与扰动策略执行视频、动作/FR目标速度与加速度曲线、两次交接状态连续性记录、最终模型SHA、可重现命令和数据版本。
- [ ] 最终声明只有在新验证命令与实测评估支持时才写“可训练/已训练/接管成功”；性能未达到门槛时交付实际统计与失败分析，不擅自换算法或随机化范围。

审核重点：0.5/4.5/1.5/0.5秒参考时序、teacher heading方向适配、联合噪声穿地处理、端点约束投影及动态残差、从头30001更新后入口10000更新的训练安排。用户审核前以上任务不执行。
