# FSM GRU student：BC → DAgger

蒸馏脚本是 [train_fsm_student.py](../scripts/train_fsm_student.py)。它使用之前生成的
[teacher 数据集](fsm_teacher_recording.md) 训练 48D GRU student，并保留现有的公共动作坐标、FR 限速和 R 操作约定。

## 第一阶段：BC

```bash
python scripts/train_fsm_student.py bc \
  --dataset outputs/fsm_teacher_dataset \
  --output outputs/fsm_student/bc.pt \
  --device cuda:0 \
  --epochs 30 \
  --batch-size 256
```

每个 batch 是一组最长 32 个真实帧的连续历史。GRU 从零 hidden 开始处理每个窗口，
**每个有效帧都计算 Smooth L1 损失**。BC 在已有完整窗口之外加入 episode 开始后的
1～31 帧前缀；短窗口在右侧补零到 32 帧，补齐帧由 boolean mask 排除，不参与损失或归一化。
窗口不跨越物理复位、终止或无效帧。

类别均衡按所选窗口中实际参与监督的每帧类别计算：窗口均匀采样，损失使用十类标签的
逐帧逆频率权重。这样上下文中重复出现的静置、行走或过渡帧也纳入均衡统计。
动作监督是 teacher 数据中公共坐标的实际动作标签，包括录制时已经发生的延迟和限速效果。

输出包括：

- `bc.pt`：student 权重、网络配置、动作／观测 contract、优化器和训练元数据。
- `bc.metrics.json`：每个 epoch 的训练和验证损失、全帧 loss 类型及参与监督的类别帧数。

输出已存在时脚本拒绝覆盖；确实要重跑时显式增加 `--overwrite`。

## 第二阶段：DAgger

```bash
python scripts/train_fsm_student.py dagger \
  --dataset outputs/fsm_teacher_dataset \
  --student-checkpoint outputs/fsm_student/bc.pt \
  --output outputs/fsm_student/dagger.pt \
  --device cuda:0 \
  --num-envs 16 \
  --iterations 3 \
  --rollout-steps 1500 \
  --warm-start-seconds 1.0 \
  --dagger-epochs 5
```

每次迭代执行以下顺序：

1. 每个物理 episode 开始后，先由 teacher 控制默认 1 秒，同时每帧运行 student 更新 GRU hidden。
   随后将动作控制权交给 student，交接时保留 hidden。
2. 对执行前的同一物理状态查询冻结的四足、双足、起身或下降 teacher。
3. 保存 student 实际访问的 48D 观测、操作端六维命令、FSM 类别和 teacher common-action 标签。
4. 用原始 BC 窗口加新 replay 窗口训练 student；默认新 replay 重复两次。

暖启动按环境独立计时：某个环境物理复位后，只清空该环境的 GRU hidden 并重新开始暖启动。
普通 FSM 支撑切换、起身和下降不会重新暖启动或清空 hidden。
`--warm-start-seconds 0` 可以关闭暖启动。若整个 rollout 都由 teacher 控制，脚本报错，提示延长
`--rollout-steps` 或缩短暖启动，避免在没有 student 访问状态时继续 DAgger。

student rollout 将 full-chain action term 的 `use_teacher_policy` 设为 `False`，并通过逐环境的
teacher mask 在暖启动期间沿用原始 endpoint teacher 动作路径；交接后 student 动作进入原有动作处理路径。
teacher 播放的默认值仍为 `True`。DAgger 保留同一个**阶段门控的** FR 限速器：初始四足行走阶段为保留原 gait 的 bypass，
FR 完全抬起后在 HOLD／后续重新定位和返回阶段启用限速器，同时保留 endpoint 延迟和 transition actor 的无延迟规则。
R 不进入观测，owner／phase／blend 只用于 teacher 标签和数据元信息。

每次迭代保存：

- `dagger.iter01.pt` 等：迭代中间 checkpoint。
- `dagger.pt`：最终 student checkpoint。
- `dagger.replay/dagger_01.npz` 等：DAgger 访问状态和 teacher 标签，以及逐帧的
  `teacher_control`、`episode_steps`，便于区分暖启动和 student 控制。
- `dagger.metrics.json`：每次 rollout 的窗口数、teacher/student 控制帧数及各自终止次数、
  有效类别帧数、teacher/student 动作差、训练损失和标签哈希。

DAgger 仍使用数据集的 32 帧窗口长度；`--sequence-length` 必须与原始 teacher 数据一致。
每个有效、非终止 replay 帧都生成一个以该帧结束的窗口，历史不足 32 帧时右侧补齐并使用 mask。
即使 episode 在 32 帧之前失败，失败前的有效帧也能训练。终止帧不参与监督；复位后从新 episode
重新积累历史，不跨越 episode boundary。暖启动帧和 student 控制帧都参与全帧监督。

## 一次运行两阶段

```bash
python scripts/train_fsm_student.py all \
  --dataset outputs/fsm_teacher_dataset \
  --output outputs/fsm_student_full_frame/student.pt \
  --device cuda:0 \
  --warm-start-seconds 1.0 \
  --epochs 30 --iterations 3 --dagger-epochs 5
```

`all` 会先生成 `student.bc.pt`，再生成最终 `student.pt` 和 DAgger replay。
旧 checkpoint 可以继续加载并立即使用推理暖启动，但已有权重不会自动获得全帧训练效果，
需要重新训练。新 loss 的监督范围、类别权重与旧末帧 loss 不同，损失数值不能直接比较。
随机训练窗口仍从零 hidden 开始；这次改动未实现跨训练窗口传递 hidden 的截断 BPTT。
在线执行持续保留 hidden，因此 32 帧训练窗口不限定在线动作的最长执行时间。

## CPU 冒烟

```bash
python scripts/train_fsm_student.py bc \
  --dataset outputs/fsm_teacher_dataset \
  --output /tmp/fsm_student_bc.pt \
  --device cpu --epochs 2 --batch-size 32 --max-windows 160

python scripts/train_fsm_student.py dagger \
  --dataset outputs/fsm_teacher_dataset \
  --student-checkpoint /tmp/fsm_student_bc.pt \
  --output /tmp/fsm_student_dagger.pt \
  --device cpu --num-envs 2 --iterations 1 --rollout-steps 80 \
  --warm-start-seconds .2 \
  --dagger-epochs 1 --batch-size 16 --max-windows 160
```

CPU 冒烟只验证数据读取、checkpoint、student 覆盖路径、冻结 teacher 查询和 replay 保存，
不代表 student 已经学会完整行走、起身或操作。正式 DAgger 应在 GPU 上使用多轮 rollout，
并另行评估四足／三足／双足、双向切换和失败恢复率。
