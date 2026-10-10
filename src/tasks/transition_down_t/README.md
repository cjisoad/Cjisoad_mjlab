# transition_down_t

独立双足到 FL/RL/RR 三足的 BeyondMimic 任务：`Transition-Down-Go2-v0`。沿用审核时的 `transition_t` 物理、75/192 观测、12 维动作、PPO 和入口课程；反向保持规则固定在本任务内，避免正向任务随后修改目标时改变已审核训练。

输入必须通过 SHA：`outputs/biped_to_tripod_preparation_20261010/` 的 7 秒参考、真实双足入口库、manifest 和两个配置 JSON。教师固定为 `mark/quad_teacher.pt`、`mark/biped_teacher.pt`。入口库四档各有 128 条不同留出轨迹，高 FR 误差区域没有实测覆盖。

```bash
PYTHONPATH=. python scripts/smoke_transition_down.py --device cpu --num-envs 8 --ppo-updates 2 \
  --motion-file outputs/biped_to_tripod_preparation_20261010/biped_to_tripod_reference.npz \
  --output outputs/down_smoke
PYTHONPATH=. python scripts/train_transition_down.py --device cuda:0 --num-envs 4096 --iterations 30001 --seed 42 \
  --motion-file outputs/biped_to_tripod_preparation_20261010/biped_to_tripod_reference.npz \
  --output logs/rsl_rl/transition_down_t/new_nominal
PYTHONPATH=. python scripts/train_transition_down_entry.py --device cuda:0 --num-envs 4096 --iterations 10000 --seed 42 \
  --motion-file outputs/biped_to_tripod_preparation_20261010/biped_to_tripod_reference.npz \
  --entry-bank outputs/biped_to_tripod_preparation_20261010/biped_entry_bank.npz \
  --checkpoint logs/rsl_rl/transition_down_t/new_nominal/model_30000.pt \
  --eval-interval 250 --eval-attempts 64 --output logs/rsl_rl/transition_down_t_entry/new_entry
```

入口 warm start 仅接受新的反向名义模型，加载 actor/critic/normalizers/distribution，重置优化器和课程时钟；`--resume` 严格校验 recipe 并恢复 Adam、课程和学习率。正向 checkpoint 始终被拒绝。

终点交接要求参考完成、姿态 <0.25rad、高度误差 <0.05m、线速度 <0.3m/s、角速度 <0.8rad/s、关节速度范数 <20，FL/RL/RR 各有 >5N 接触，FR 无接触且足中心 >0.032m。1s 好样本积分允许两次坏样本暂停，第三次清零；失败立即清零。严格 6cm 世界足端 RMS 与连续 1s 另列诊断。

`evaluate_transition_down.py` 报告 nominal/base/joints/combined 四组 frame0 评估；`--startup-randomization --observation-noise` 单独报告训练增强效果。`evaluate_transition_down_entry.py` 报告四档留出入口。`evaluate_transition_down_chain.py` 在同一世界继续双足老师 2s，再执行下降、三足保持和三足老师 3s；交接记录 qpos/qvel、动作历史和 limiter 的零差值。

入口 schema v2 没有保存老师未执行的请求动作，因此恢复时以保存的已执行 raw/previous action 初始化请求缓存；两次实时 ownership 切换均保持全部缓存。该初始化不会给入口额外 tracking grace。参考为受约束 IK 的运动学目标，动力学成功需以训练后实测结果判断。

`run_transition_down_remote.sh` 先 SHA/版本校验、8 环境 GPU nominal/entry smoke，再从头 30001 更新、入口 10000 更新、四组/入口/链路评估。每100更新保存模型，每250更新以64不同留出轨迹评估课程。运行目录记录 PID、阶段、日志、退出码、部署源码/数据 SHA。
