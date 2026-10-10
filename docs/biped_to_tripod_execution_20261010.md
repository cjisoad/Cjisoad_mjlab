# 双足到三足实施与验证记录

用户已批准数据和计划，并授权测试通过后合入 `wsl-validation`、推送及远端训练。新任务为 `Transition-Down-Go2-v0`，代码在 `src/tasks/transition_down_t/`，命令见其 README。

隔离工作树：`/home/eden/.codex/worktrees/8301/unitree_rl_mjlab`；基础提交 `03625f1`。仅新增反向任务、对应脚本和测试合入；工作区既有正向任务、README 等修改保留。正向保持目标正在并行修改，因此反向任务固定本次审核的三坏样本规则及入口 recipe。

本地证据位于 `outputs/transition_down_validation_20261010/`：

- `tests_final.log`：163 passed，30 subtests；包含反向任务、原正向参考/入口/奖励/契约、老师坐标和 FR 限速器回归。
- `integrated_tests.log`：合入当前工作区后33 passed，7 subtests。
- `nominal_smoke/verification.json`：CPU 8环境、两次真实 PPO 更新、有限观测/奖励、参数更新、checkpoint round trip、改变动作契约拒绝。
- `entry_smoke/verification.json` 与 `integrated_entry_smoke/verification.json`：反向名义模型 warm start、fresh optimizer、两次更新、严格 resume、课程及学习率恢复；32独立留出入口零策略评估，仅用于验证执行链路，成功0。
- `full_start_zero.json`、`full_start_trained_smoke.json`：四组frame0评估入口运行验证；smoke模型不代表成功策略。
- `chain_zero.json`、`chain_trained_fixed.json`：真实双足老师继续执行2s后切入反向任务；qpos/qvel、经理三帧动作历史、执行/请求缓存及FR limiter交接差值均0。
- `video_smoke/`：真实策略视频、动作轨迹、FR物理子步目标速度/加速度图；只验证记录入口。
- `baseline_teacher_tests.log`：4项既有教师工作空间测试在原工作区同样失败，原因是扩展范围0.39m/0.6m与旧断言0.30m/0.35m不符，未修改这些无关测试。

独立审查修复了 trained 链路中包装器的隐式reset、smoke导入路径以及同一步快照重复计时。真实入口resume在inference mode下重置传感器缓存，解决PPO之后普通reset的inference tensor写入错误。

审核输入仍为 `outputs/biped_to_tripod_preparation_20261010/` 的冻结文件。任务校验参考、入口和三个review JSON的SHA，两个指定老师另校验SHA。部署保留该目录结构，不改变数据manifest。

远端：`root@xj-member.bitahub.com:42078`；RTX4090 24GB；环境mjlab1.2.0、MuJoCo3.5.0、mujoco-warp3.5.0、Warp1.12.0、rsl-rl5.0.1、torch2.8.0+cu126。本地torch2.14.0且无CUDA，需远端GPU smoke验证。

部署目录：`/pedipulation/legmanip1/transition_down_20261010/repo`。`run_transition_down_remote.sh`先验证源码/数据SHA和环境版本，运行8环境GPU名义及入口smoke，再名义从头4096环境30001更新、保存每100更新；之后反向名义模型warm start入口10000更新，每250更新64独立留出评估，完成后进行四组100回合、增强、四档入口和128独立老师链路评估及视频。运行阶段、PID、日志、退出码与最终模型SHA均记录于新运行目录。

正式训练完成及性能结果以远端运行日志、模型和独立评估为准；本地通过不表示已学会完整下降或三足接管。

2026-10-10 远端启动核验：实现提交为 `ad8cd9a`，合入远端并发提交后的训练源码为 `4f6120a1c05bfdc13cb323aee9d9a9ba74dcd659`，已推送到 `origin/wsl-validation`。合并后的反向测试再次执行，33 passed、7 subtests passed。

初次远端流水线因容器缺少 `libEGL.so.1` 在 GPU smoke 前退出，尚未训练。安装 `libegl1`、`libopengl0` 后，MuJoCo EGL 导入和 CUDA 验证通过，使用全新运行目录重启；没有改变数据、算法或随机化。

当前运行目录为 `/pedipulation/legmanip1/transition_down_20261010/run_gpu01`，tmux 会话为 `transition_down_20261010`。627 个部署文件 SHA 校验全部通过。GPU 名义 smoke 的两次 PPO 更新、有限观测/奖励、参数更新、模型 round trip 和动作契约拒绝通过；GPU 入口 smoke 的 warm start、fresh optimizer、严格 resume、学习率/课程恢复、改变目标拒绝通过，另运行32条独立零策略入口用于检查评估执行。

名义训练进程于北京时间 `2026-10-10 15:27:57` 启动，PID `15046`，流水线 PID `14589`。北京时间 `15:32:31` 的核验快照已记录到第153次更新，所检查的损失、学习率、动作标准差、奖励和回合长度历史均为有限值；`model_100.pt` 已按100次间隔保存，actor/critic 状态均为有限值，较第0次检查点发生变化。该检查点 SHA 为 `100b0ab3adff31e71e635bfdf07fa0664d85fff3ee9453b98cf8890125460d9f`。这是启动核验，最终成功率尚待训练后的独立评估。

启动证据保存在本地 `outputs/transition_down_validation_20261010/remote_start_verification.json` 与远端 `run_gpu01/start_verification.json`，包含设备、版本、PID、源码版本、部署清单 SHA、GPU smoke 结果、学习指标和检查点 SHA。远端持久存储中的文本日志在核验时尚未显示内容，实时推进已通过 TensorBoard event 文件确认；监控时同时检查 event 文件和周期检查点。

流水线将自动依次执行名义30001次更新、名义评估、入口10000次更新以及最终评估和视频记录。预计最终文件为：

- 名义：`/pedipulation/legmanip1/transition_down_20261010/repo/logs/rsl_rl/transition_down_t/run_gpu01_nominal/model_30000.pt`。
- 入口：`/pedipulation/legmanip1/transition_down_20261010/repo/logs/rsl_rl/transition_down_t_entry/run_gpu01_entry/model_9999.pt`。
- 评估与视频：`/pedipulation/legmanip1/transition_down_20261010/run_gpu01/`；`stage` 表示阶段，`exit_code` 出现表示流水线已退出。

SSH 进入后可用以下只读命令检查最新训练指标；第一阶段完成后将目录改为上述入口日志目录：

```bash
/opt/conda/bin/python - <<'PY'
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
path = '/pedipulation/legmanip1/transition_down_20261010/repo/logs/rsl_rl/transition_down_t/run_gpu01_nominal'
events = EventAccumulator(path).Reload()
for tag in ('Loss/value', 'Loss/surrogate', 'Train/mean_reward', 'Train/mean_episode_length'):
    value = events.Scalars(tag)[-1]
    print(tag, 'iteration=', value.step, 'value=', value.value)
PY
```
