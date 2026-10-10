# 双足到三足过渡：数据审核记录

状态：已完成数据准备，等待用户审核数据与实施计划；未创建反向训练任务，未执行 PPO 更新，未启动正式训练。

数据目录：`outputs/biped_to_tripod_preparation_20261010/`。该目录被仓库现有 `.gitignore` 忽略；文件实际保存在当前工作区。设计与计划分别见 `docs/superpowers/specs/2026-10-10-biped-to-tripod-transition-design.md` 和 `docs/superpowers/plans/2026-10-10-biped-to-tripod-transition.md`。

## 优先审核的文件

| 内容 | 文件 |
|---|---|
| 双视角完整动作预览，标注为运动学参考 | [reference_preview.mp4](../outputs/biped_to_tripod_preparation_20261010/reference_preview.mp4) |
| 9 个关键姿态 | [keyframes.png](../outputs/biped_to_tripod_preparation_20261010/keyframes.png) |
| 姿态、足端高度、FR 命令、支撑力、后膝净空 | [reference_curves.png](../outputs/biped_to_tripod_preparation_20261010/reference_curves.png) |
| 可直接由当前参考加载器读取的反向参考 | [biped_to_tripod_reference.npz](../outputs/biped_to_tripod_preparation_20261010/biped_to_tripod_reference.npz) |
| 两个指定教师的实测端点及选择依据 | [teacher_endpoints.json](../outputs/biped_to_tripod_preparation_20261010/teacher_endpoints.json) |
| 随机化范围、种子、初态 FK 诊断 | [randomization_config.json](../outputs/biped_to_tripod_preparation_20261010/randomization_config.json) |
| 双足教师真实入口库 | [biped_entry_bank.npz](../outputs/biped_to_tripod_preparation_20261010/biped_entry_bank.npz) |
| 入口数量、分组覆盖、训练/留出分割 | [entry_bank_audit.json](../outputs/biped_to_tripod_preparation_20261010/entry_bank_audit.json) |
| 直立姿态航向对齐问题的实测对照 | [alignment_comparison.json](../outputs/biped_to_tripod_preparation_20261010/alignment_comparison.json) |

## 教师身份与物理配置

实际加载并验证了用户指定的文件，未使用仓库中的其他默认教师代替。

| 教师 | 任务与观测 | checkpoint iteration | SHA256 |
|---|---|---:|---|
| `mark/quad_teacher.pt` | `loco_pedipulation_t`，actor54 / critic95 | 10900 | `1176fb6c71ba5c05c7d2c8ac36f6492f9b3dd37ba8dcafc172e2c3da0bc5d0fd` |
| `mark/biped_teacher.pt` | `pedipulation_t`，actor51 / critic89 | 14999 | `f919492bbb59fe29158e6aa6cc0cc62acf12bcf16cbdec70a86705c9ba084b6b` |

两个教师的关节零点不同。采集复用了 `AlignedKeyboardTeacher` 的观测构造、教师自身动作裁剪和公共坐标重基，保留教师控制步内延迟、FR 目标速度 2 rad/s / 加速度 30 rad/s² 限制。公共动作仍为 `q_target = loco_default_angles + 0.25 * action`。

参考生成与采集采用当前 `transition_t` 名义机器人。采集关闭 startup 随机化和观测噪声，训练随机化资料另行记录。采集物理 SHA256 为 `dac78f222f6f1f668c6c9b582d2a48bb277af84723de3af3324541ae5d770367`；机器人源几何 SHA256 为 `c45dbc763bbbac4703b9cfa57e0d332bb234b0230b1ecde43da69cb353ab2b0c`。完整动作与物理元数据保存在采集记录和 bank metadata。

当前本机 `torch.cuda.is_available()` 为 false，因此本轮资料全部通过 CPU 生成。未连接远端、未占用训练服务器。

## 参考数据生成

首先在当前过渡物理中运行指定教师：64 个环境，12 秒；32 个双足、32 个四/三足，保存 7,680 个时刻状态。双足初始化只使用旧起立参考末帧作为启动姿态，随后所有采样均来自新指定双足教师的真实控制结果，rollout 期间没有重写姿态。三足从默认状态开始，先稳定、再逐渐抬 FR。

端点从实际访问过的样本中选择，不平均关节和四元数。要求指定目标、姿态、高度、速度、FR 误差及相应接触门槛。三足端点实测 FR 误差 9.13 mm、根线速度 0.00186 m/s；双足端点实测 FR 误差 11.61 mm、根线速度 0.04756 m/s。双足端点筛选获得 322 个双后脚支撑且角速度低于 0.8 rad/s 的候选样本。

随后复用原 `teacher_endpoint_rise_clearance20mm_20261008/generate.py` 的 MuJoCo FK、约束 IK、固定后脚落点、FL 路径、FR 公共坐标目标、COM 转移和 20 mm 后膝净空要求，以这些实测端点重新求解路径，再进行时间反转。原生成器和适配版本均保存在 `method_snapshot/`，并记录 SHA256。

位置和接触计划反序；关节速度、MuJoCo 广义速度、身体世界线/角速度及 root twist 反序后取负；加速度和逆动力学重新计算。水平位置整体平移，使反向首帧根部 XY 为零。FR 全程悬空，RL/RR 全程保持规划支撑，FL 在 5 秒处开始规划支撑。

| 时间 | 动作与规划支撑 |
|---|---|
| 0–0.5 s | 双足保持，RL/RR 支撑 |
| 0.5–5.0 s | 降低机身、FL 向地面摆动，RL/RR 支撑 |
| 5.0–6.5 s | FL 落地加载、COM 向三足支撑转移 |
| 6.5–7.0 s | FL/RL/RR 三足保持，FR 悬空 |

完整参考为 **50 Hz、351 帧、7 秒**，含 12 个 canonical 关节、29 个身体的名字与状态、root、qpos/qvel、四足位置、接触计划、静态支撑力/扭矩及后膝净空。

参考 SHA256：`fa49119da56b0eb2ac2dc49d5b65dfe310aad8033a5e15565ba5079872bb11aa`。

### 时序调整及拒绝候选

原主体运动段 3 秒直接代入新教师端点，最大关节速度 **4.971 rad/s**、FR 最大速度 **2.135 rad/s**，未通过原数据 4 rad/s 验收线。该候选完整保存在 `rejected_original_timing/`，没有作为训练参考发布。

采用相同 IK 方法，将主体下降段延长到 4.5 秒，压缩双足起始保持段，保持总时长 7 秒和采样率 50 Hz。最终关节最大速度为 **3.433 rad/s**，FR 为 **1.473 rad/s**。这是数据时序的方向适配；PPO、网络、奖励主项和随机化范围不因此改变。

### 几何与动力学审计

| 检查 | 实测结果 |
|---|---:|
| 已存参考 FK、四元数、world 速度及广义速度差分 | 全部 351 帧通过 |
| 与当前加载器一致的插值几何检查 | 250 Hz，共 1,751 个状态 |
| 后膝视觉外壳最小净空 | 21.943 mm |
| 后小腿碰撞体最小净空 | 27.252 mm |
| 超过原深度门槛的非足部地面/自碰穿透 | 0 |
| 起始/最终根高度 | 0.49153 / 0.29848 m |
| 条件逆动力学最大扭矩 | RL calf，24.990 N·m |
| 条件逆动力学最大力矩限额占比 | RL thigh，78.12% |
| 最大未平衡浮动基座合力/力矩 | 13.617 N / 3.957 N·m |

**参考仍是运动学训练目标，不能据此认定已经可以物理执行。** 与原起立数据一样，点接触模型下存在未平衡机身力/力矩。`torque_audit.json` 明确记为动态平衡不满足，`verification.json` 明确记为没有训练策略跟踪验证。视频为 FK 回放，不是策略执行录像。

端点约束投影也不等于教师原始关节角：反向首帧 RL/RR hip 分别调整约 −0.444 / +0.489 rad，其他调整详见 `reference_audit.json`；三足末端最大关节调整约 0.00483 rad。真实教师入口课程需要覆盖这一初态差异。

另做了端点教师接续诊断：从新参考首/末姿态分别交给指定双足/三足教师，各 16 条轨迹、5 秒，32 条均没有物理失败。三足后 3 秒 FL/RL/RR 接触比例均为 100%，FR 为 0，平均线速度 0.00386 m/s；双足后脚仍有交替接触。该实验没有经过过渡策略，不能视为完整双足→三足链路成功率。

## 随机化资料

以当前运行代码为准；未沿用旧设计文档中已被移除的 reset IK。训练采用直接根位姿/速度、关节角噪声及软限位裁剪，不为足端另造独立扰动，不进行碰撞预筛选。

| 随机项 | 与当前 `transition_t` 相同的范围 |
|---|---|
| 根 XY / Z | 每轴 ±0.02 / ±0.01 m |
| 根 roll、pitch / yaw | 每轴 ±0.08 / ±0.15 rad |
| 根线速度 XY / Z | 每轴 ±0.20 / ±0.10 m/s |
| 根角速度 | 每轴 ±0.30 rad/s |
| 关节位置 | 每个关节 ±0.1 rad，按 0.98 软限位裁剪 |
| 关节速度 | 保留参考值，无额外噪声 |
| startup base COM | 每轴 ±0.01 m |
| startup 足部摩擦 | 0.6–1.2，四足共享样本 |
| startup encoder bias | 每关节 ±0.01 rad |

actor 噪声也按现有配置记录：anchor 相对位置 ±0.25、6D 姿态 ±0.05、基座线速度 ±0.5、角速度 ±0.2、关节位置 ±0.01、关节速度 ±0.5；command 和上一动作不加噪声，critic 不加噪声。运行时 push 关闭。

已保存：`reset_review_samples.npz` 共 512 个名义/base/joints/combined 首帧状态（每组 128），含实际 root/joint、FK 足端、扰动值及诊断；`startup_randomization_samples.npz` 共 4,096 组 COM/摩擦/encoder/75 维 actor 噪声样本。评审样本采用 NumPy 固定种子，表达相同分布与扰动组成，不声称与 GPU/Torch 随机数流逐位相同。

联合扰动组有 **15/128** 个非足部初始穿地诊断；其余三组为 0。所有 512 个样本均保留，关节裁剪计数为 0，完整足端实际分布在 JSON 中。这个现象需要审核：若缩小范围或加入过滤，会改变用户要求沿用的原 reset 方法。本计划默认保持原方法并单独报告这些初态的训练/评估结果。

训练仍为 25% 强制首帧，其余来自在线自适应失败帧采样；不额外设置无扰动训练 episode 比例。在线采样器的失败统计尚未产生，因为本轮没有训练。

## 双足真实入口库

128 个环境 × 5 批 × 8 秒，seed 20262010–20262014，共 640 条独立轨迹、51,200 个原始样本。包括静止、FR 目标变化以及运动命令（前进最高请求 1 m/s）的真实双足教师运行。没有向 bank 人工添加根速度或关节姿态。

筛选有限值、原轨迹存活、至少已运行 2 秒、双足姿态/高度、前脚离地和至少一只后脚接触的真实样本，保留 **39,040 个状态、640 条轨迹**。其余 12,160 个样本主要来自启动阶段的时间过滤，不能称为 12,160 个物理失败。按 trajectory_id % 5 划分：**512 条训练轨迹，128 条留出轨迹**，无轨迹交叉。

schema 2 保存 root 世界速度、关节位置/速度、当前/上一实际动作、三帧公共动作经理历史、上一关节速度、限速器位置/速度、真实水平速度、FR 偏移/误差、命令和轨迹 ID。通过当前 `EntryStateBank` 的形状、有限值、四元数、速度、执行动作及限速器一致性校验。

| 课程 | 速度 / FR 误差上限 | 训练状态数 | 留出状态数 | 合格独立留出轨迹 |
|---|---|---:|---:|---:|
| 0 | 0.25 m/s / 0.08 m | 26,391 | 6,693 | 128 |
| 1 | 0.45 m/s / 0.14 m | 28,704 | 7,185 | 128 |
| 2 | 0.70 m/s / 0.22 m | 29,748 | 7,444 | 128 |
| 3 | 1.20 m/s / 0.45 m | 31,232 | 7,808 | 128 |

真实速度最大 **1.079 m/s**；FR 首帧误差中位数 **3.325 cm**、90 分位 **6.749 cm**、最大 **13.051 cm**。每档均满足 64 条独立留出轨迹的数量条件；这不是策略晋级或成功证据。FR 误差 0.14–0.45 m 的大偏差区域没有实测覆盖，1.08–1.2 m/s 区域也未被验证；不能将课程阈值写成已覆盖的鲁棒范围。

最终 bank SHA256：`ff962e464cd928e44dea76f60363867f5361a45564ad7adbd6a43632d7193058`。

### 双足起点需要的航向适配

原任务从三足开始，`yaw_quat` 使用 Euler yaw。双足的 body +X 接近竖直，轻微过仰便会使这个 yaw 翻转约 π；本次同一组真实状态上，Euler yaw 对齐的 FR 误差中位数为 **0.6816 m**，导致课程 0 没有合格状态。

使用两个教师已采用的 `teacher_body_y_cross_up_v1` 水平航向，仅旋转参考水平航向并对齐 XY，FR 误差中位数为 **0.03325 m**。实际 root、关节、速度、动作及课程阈值全部没有变化。新 bank 的 metadata 明确要求 `teacher_body_y_cross_up_v1_xy_heading`；未来反向 command、入口恢复、评估与 FSM 接入必须采用这个定义，不能直接把新 bank 交给原正向 `EntryCommand` 的 Euler yaw 对齐。

错误对齐生成的 bank 和统计保存在 `rejected_euler_yaw_*` 作为对照，不列为训练输入。

## 复现与记录

在仓库根目录、当前 mjlab Python 环境中执行以下命令。它们全部是冻结教师推理或数据处理，不启动训练。

```bash
PYTHONPATH=. python outputs/biped_to_tripod_preparation_20261010/collect_teachers.py --cohort paired --num-envs 64 --batches 1 --seconds 12
PYTHONPATH=. python outputs/biped_to_tripod_preparation_20261010/build_reference.py --descent-seconds 4.5
PYTHONPATH=. python outputs/biped_to_tripod_preparation_20261010/collect_teachers.py --cohort source --num-envs 128 --batches 5 --seconds 8 --seed 20262010
PYTHONPATH=. python outputs/biped_to_tripod_preparation_20261010/audit_data.py --phase all
MUJOCO_GL=egl PYTHONPATH=. python outputs/biped_to_tripod_preparation_20261010/build_previews.py
PYTHONPATH=. python outputs/biped_to_tripod_preparation_20261010/collect_teachers.py --cohort endpoints --num-envs 32 --batches 1 --seconds 5 --seed 20266010
PYTHONPATH=. python outputs/biped_to_tripod_preparation_20261010/verify_bundle.py
```

原始状态、种子、生成器快照、配置、输入/输出 SHA256、失败候选、过程日志均保留。CPU 接触计算的逐位重复性并不保证；审核和训练绑定的是本次实际文件 SHA，而非仅依据种子认定结果相同。`manifest.json` 记录最终审核包及所依赖代码文件的哈希。

优先审核：下降/FL 落地动作与时序、直接随机化产生的穿地样本、双足入口航向定义、双足端点的约束投影误差、动态平衡残差，以及后续训练与完整教师链路评估计划。
