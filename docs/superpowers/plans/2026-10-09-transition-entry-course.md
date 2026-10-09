# Transition Entry Course Implementation Plan

**Goal:** 用 model_14999 的真实三足入口继续训练 transition_t/model_30000，并按真实入口难度递进。

**Architecture:** 独立 entry_bank 模块验证和分箱抽样；entry_course 命令在物理 reset 后恢复教师状态并开始首帧参考；采集器复用 keyboard 环境的冻结三足动作；专用续训脚本执行 PPO、留出评估、课程升级及元数据保存。

**Tech Stack:** Python、PyTorch、NumPy NPZ、mjlab/MuJoCo Warp、RSL-RL、pytest、SSH。

- [x] 在 `tests/test_transition_t_entry_bank.py` 先测试状态库有限值/形状/轨迹隔离、分箱均衡、课程升级和恢复；运行失败后实现 `src/tasks/transition_t/entry_bank.py`。
- [x] 在 `tests/test_transition_t_entry_course.py` 先测试子集真实状态恢复、参考 XY/yaw 对齐、动作经理及动作项历史恢复、物理 reset 年龄与原观测契约；实现 `src/tasks/transition_t/entry_course.py`，完成 CPU reset/step。
- [x] 实现 `scripts/collect_transition_entries.py`，使用 actor54 model_14999 和共同名义物理，记录实际 FR/速度、不同轨迹、完整公共动作历史；小规模采集并校验训练/留出分组。
- [x] 实现 `scripts/train_transition_entry.py`，加载原 actor/critic/归一化，重置课程计数，保存与检查状态库 SHA/课程元数据，分批 PPO 与留出站稳评估晋级；实现真实一次 PPO 更新和 checkpoint 回读验证。
- [x] 运行旧 transition/keyboard 与新课程聚焦测试；独立审查后修复问题。保留实际失败及覆盖不足，不放宽门槛。
- [ ] 核对远端 GPU/环境/现有任务，部署仅本次所需代码及模型到新目录，采集正式状态库并启动有界续训；确认 checkpoint、TensorBoard、有限奖励及课程日志，记录复现命令与 PID。
- [x] 更新任务 README 与交付报告，给出状态覆盖、留出基线、新训练输出和状态，明确正式效果待训练后评估。

验证命令采用 `/home/eden/miniconda3/envs/mjlab/bin/python`；临时 pytest 在 `/tmp/keyboard-transition-test-deps`，缓存均在 `/tmp`。不修改 Git 索引，不提交其他既有工作。

当前本地状态库有 12,669 状态/748 轨迹。真实 CPU PPO 连续两次更新后恢复至第 3 次，优化器步数 40→60、环境步数 48→72，自适应学习率保持 1e-5。课程晋级额外要求至少 64 条独立留出轨迹；本地 level 0 只有 48 条，不允许以重复状态满足证据门槛。各档原模型留出基线已完成；远端部署等待用户重启后提供的新 SSH 命令。
