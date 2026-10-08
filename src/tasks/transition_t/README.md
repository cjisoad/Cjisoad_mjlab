# transition_t：Go2 单次起立过渡跟踪

注册任务：`Transition-Go2-v0`。基于本仓库 `src/tasks/tracking/` 的 BeyondMimic 实现，训练 `loco_pedipulation_t → transition_t → pedipulation_t` 的起立方向。任务独立于已有的 `pedipulation_transition_t` 和 `pedipulation_bridge_t`。

默认数据为仓库下的 `outputs/teacher_endpoint_rise_clearance20mm_20261008/tripod_to_biped_reference.npz`：50 Hz，351 帧，7 秒。SHA256：`1cb970ae8cfb107e25f3aca2e1d48b8cf9a2a145d6fac25c9aa67076c3bdd0b6`。允许显式替换同 schema 的参考文件；checkpoint 会检查其哈希。数据按关节/身体名字映射，拒绝缺字段、非法形状、非有限值及非单位四元数。

这是运动学参考，数据审计给出的动态平衡残差最大约 13.43 N、3.20 N·m。真实仿真跟踪、鲁棒性、两端教师交接需要训练后验证。当前轨迹没有反向落地动作，任务也不包含自动切换 FSM 或冻结教师训练。

## 保留的 BeyondMimic 结构

- 当前参考关节位置/速度、参考 anchor 相对位置/6D 姿态、实际机身线速度/角速度、关节位置/速度和上一动作：actor **75 维**。
- critic 增加 base_link 和四条腿 thigh/calf/foot 共 13 个身体的相对位置/姿态：**192 维**。
- canonical FL/FR/RL/RR、每条腿 hip/thigh/calf，共 **12 维**位置偏移动作，`q_target = loco_default_angles + 0.25 * action`，关闭动作延迟。
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

保留小幅 startup 随机化：base COM ±0.01 m、foot friction 0.6–1.2、encoder bias ±0.01 rad。关闭继承的运行时 push；无新课程或额外恢复奖励。play/evaluation 默认关闭 startup 随机化和观测噪声，以单独测量初态鲁棒性。

## 时间与切换接口

控制周期 0.02 s，物理步长 0.005 s，episode 8 s。播完参考后保持末帧，目标速度归零；不会在参考末尾重采样或瞬移。PPO runner 禁止仅随机 episode 计数而不匹配物理状态。

跟踪误差终止采用 anchor/四足高度误差 >0.15 m、完整四元数角误差 >0.8 rad。reset 后 0.2 s 内暂缓这些误差终止；计时按实际仿真步数。非有限状态及基座/头部严重触地立即失败。

切入策略时，可在同一机器人环境中调用：

```python
term = env.command_manager.get_term("motion")
term.start_from_current_state(env_ids)  # env_ids: device 上的 LongTensor
```

该接口将参考首帧的水平位置/yaw 对齐当前机身，同时旋转世界系目标速度；保留高度、倾角、关节误差。它不写 root/joint 状态，不清除速度、上一动作或执行器历史，也不重置物理 reset 宽限期。调用方负责从正确的观测和动作接口运行已训练策略，并在完成终点保持后交给站立教师。这里提供的是无瞬移的切入接口，尚未验证两教师闭环接管。

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

服务器 RTX 4090 已验证 CUDA rollout/PPO/checkpoint 链路。2026-10-09 按用户要求改为 BeyondMimic 直接根状态与关节角噪声 reset，旧 IK 训练停止，新版本从头训练。正式收敛、鲁棒成功率和教师交接仍需训练后的评估。旧版初始化回退率诊断不适用于当前版本。
