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

每个 batch 是 `windows.npz` 中的一组连续 32 帧。GRU 从零 hidden 开始处理整段窗口，
但 Smooth L1 损失只计算最后一帧：窗口末帧的十类标签已经均衡，前 31 帧提供状态和动作历史。
因此训练不会因静置或过渡阶段的长上下文改变类别比例。动作监督是 teacher 数据中公共坐标的实际动作标签，
包括录制时已经发生的延迟和限速效果。

输出包括：

- `bc.pt`：student 权重、网络配置、动作／观测 contract、优化器和训练元数据。
- `bc.metrics.json`：每个 epoch 的训练和验证损失。

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
  --dagger-epochs 5
```

每次迭代执行以下顺序：

1. 载入当前 student，在完整 FSM MuJoCo 环境中执行 student 动作。
2. 对执行前的同一物理状态查询冻结的四足、双足、起身或下降 teacher。
3. 保存 student 实际访问的 48D 观测、操作端六维命令、FSM 类别和 teacher common-action 标签。
4. 用原始 BC 窗口加新 replay 窗口训练 student；默认新 replay 重复两次。

student rollout 会将 full-chain action term 的 `use_teacher_policy` 设为 `False`，使 student 动作直接进入原有动作处理路径；
teacher 播放的默认值仍为 `True`。DAgger 保留同一个**阶段门控的** FR 限速器：初始四足行走阶段为保留原 gait 的 bypass，
FR 完全抬起后在 HOLD／后续重新定位和返回阶段启用限速器，同时保留 endpoint 延迟和 transition actor 的无延迟规则。
因此 student rollout 不会偷偷让 endpoint teacher 覆盖 student 输出。
R 不进入观测，owner／phase／blend 只用于 teacher 标签和数据元信息。

每次迭代保存：

- `dagger.iter01.pt` 等：迭代中间 checkpoint。
- `dagger.pt`：最终 student checkpoint。
- `dagger.replay/dagger_01.npz` 等：DAgger 访问状态和 teacher 标签。
- `dagger.metrics.json`：每次 rollout 的窗口数、teacher/student 动作差、训练损失和标签哈希。

DAgger 仍使用数据集的 32 帧窗口长度；`--sequence-length` 必须与原始 teacher 数据一致。
短于一个窗口的 CPU 冒烟 rollout 会生成 replay，但不能产生 replay training window。
失败或物理复位后的窗口会从下一帧重新开始，不跨越 episode boundary。

## 一次运行两阶段

```bash
python scripts/train_fsm_student.py all \
  --dataset outputs/fsm_teacher_dataset \
  --output outputs/fsm_student/student.pt \
  --device cuda:0 \
  --epochs 30 --iterations 3 --dagger-epochs 5
```

`all` 会先生成 `student.bc.pt`，再生成最终 `student.pt` 和 DAgger replay。

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
  --device cpu --num-envs 2 --iterations 1 --rollout-steps 1500 \
  --dagger-epochs 1 --batch-size 32
```

CPU 冒烟只验证数据读取、checkpoint、student 覆盖路径、冻结 teacher 查询和 replay 保存，
不代表 student 已经学会完整行走、起身或操作。正式 DAgger 应在 GPU 上使用多轮 rollout，
并另行评估四足／三足／双足、双向切换和失败恢复率。
