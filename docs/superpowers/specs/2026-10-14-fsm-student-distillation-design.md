# FSM GRU student 两阶段蒸馏设计

## 目标

使用已经录制的 FSM teacher 数据训练 48D GRU student：第一阶段用行为克隆（BC）学习 teacher 动作，第二阶段用 DAgger 让 student 在自己的访问状态上重新查询冻结 teacher，并将新轨迹加入训练集。

## 约定

student 输入仍是 48 维当前观测，包含操作端六维命令和上一控制周期实际公共动作；R、owner、phase、blend、teacher reference 不进入输入。动作是 12 维公共关节位置偏移，执行时复用 teacher 环境的动作 scale、阶段门控 FR 限速器、延迟和 PD 坐标。

BC 从 `outputs/fsm_teacher_dataset/windows.npz` 读取长度固定的连续窗口。窗口末帧是监督锚点类别，前面的帧只提供 GRU 上下文；损失只计算最后一步，从而不让长上下文改变十类均衡。DAgger 的 replay 使用同一窗口格式和末帧损失。

## DAgger 状态访问

每个 DAgger rollout 使用完整 FSM MuJoCo 环境和现有 `ScriptedEpisode` 命令脚本。动作项增加显式 `use_teacher_policy` 开关：BC/teacher 回放维持原有 endpoint 覆盖，student rollout 时输入动作直接进入公共动作项，但仍经过相同的阶段门控限速器；初始四足 gait 阶段的 bypass 保留，FR 完全抬起后的 HOLD、重新定位和返回阶段启用限速。每个控制周期先构造 student 观测并查询冻结 teacher common action，然后执行 student action；因此标签对应 student 当前物理状态，而不是 teacher 自己的下一状态。起身和下降也通过同一 frozen transition actor 查询。

失败 world 标记 episode start，不能跨 reset 组成窗口。每次 DAgger 迭代原子保存 replay 分片和 manifest，训练集由原始 BC 窗口与新 replay 窗口组成，并对新 replay 可配置重复采样比例。

## 命令与产物

```bash
python scripts/train_fsm_student.py bc \
  --dataset outputs/fsm_teacher_dataset \
  --output outputs/fsm_student/bc.pt \
  --epochs 30 --batch-size 256 --device cuda:0

python scripts/train_fsm_student.py dagger \
  --dataset outputs/fsm_teacher_dataset \
  --student-checkpoint outputs/fsm_student/bc.pt \
  --output outputs/fsm_student/dagger.pt \
  --iterations 3 --rollout-steps 1500 --num-envs 16 --device cuda:0
```

`all` 子命令连续运行 BC 和 DAgger。每一阶段保存模型、训练日志、随机种子、观测／动作 contract、teacher checkpoint 哈希和 replay 文件。CPU 使用少量 epoch/步数进行链路冒烟；CPU 冒烟不能证明 student 已经具备完整运动能力。

## 验收

- 合成窗口测试检查末帧损失、GRU hidden reset、BC checkpoint 和 replay 窗口边界。
- CPU BC 冒烟能从正式 dataset 读取数据、下降 loss 并恢复 checkpoint。
- CPU DAgger 冒烟能加载完整 FSM teacher，切换 student 执行路径，生成 finite replay 和 teacher labels，且不改变原 teacher 默认模式。
- 运行现有 FSM/student/recording 测试与新增 distillation 测试后，提交仅包含本功能相关文件并推送当前远端分支。
