# transition_t 恢复 BeyondMimic 直接随机化

用户明确要求：取消 IK 和 CPU 碰撞筛选，按 BeyondMimic 随机化后推送并重新训练。

- [x] 停止旧训练（旧 run/checkpoint 保留）。
- [x] 直接均匀随机根位置、姿态、速度及关节角；关节角 ±0.1 rad 并按软限位裁剪；删除 IK、重试与回退。
- [x] 更新评估为 nominal/base/joints/combined，记录实际关节/足端偏差。
- [x] checkpoint 契约升级为 version 2，标明 beyondmimic_root_joint_noise_v1，防止混用旧 reset 训练。
- [x] 新配置/命令测试 45 项通过，CPU PPO/checkpoint smoke 通过。
- [ ] 提交推送并同步服务器。
- [ ] GPU smoke，通过后从头启动 4096 环境、30001 轮、每100轮保存的新训练。
- [ ] 观察实际吞吐，给出新的关机时间。

本次仅修改 transition_t 及其测试、评估、文档，保留其他任务的未提交改动。参考末端保持、状态保留切入接口、奖励、网络和 PPO 参数保持已批准配置。
