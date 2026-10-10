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
