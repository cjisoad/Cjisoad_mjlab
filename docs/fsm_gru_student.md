# FSM teacher 的 GRU student

提供网络、观测构造、带记忆的推理接口和权重保存／加载。
教师数据可通过 [FSM 录制脚本](fsm_teacher_recording.md) 自动采集并生成均衡窗口索引。
student 的两阶段 BC→DAgger 训练见 [蒸馏说明](fsm_student_distillation.md)。
`keyboard_transition_fsm.py` 可执行教师链，也可通过 `--student-checkpoint-file` 加载 student。

## 网络与动作

默认结构为：

```text
当前帧 48 维观测 → Linear(48, 128) + ELU
                   → GRU(input=128, hidden=128, layers=1)
                   → Linear(128, 128) + ELU
                   → Linear(128, 64) + ELU
                   → Linear(64, 12)
```

每个环境只保留自己的 GRU 隐藏状态 `[1, 128]`，不堆叠过去 50 帧。
输出为现有动作接口中的 12 维位置偏移，顺序为
FL、FR、RL、RR，每条腿依次 hip、thigh、calf：

```python
joint_target = default_angles + action_scale * student_action + motor_offsets
```

`action_scale` 默认 0.25，默认关节零点取自完整 FSM 世界同样使用的
Go2 `INIT_STATE`，不是单独某个端点教师的零点。创建训练模型时应从实际环境动作项读取
`default_angles` 和 `action_scale`，并在推理时保持相同的坐标约定。
输出不使用 tanh，避免限制经过教师零点转换后的监督动作范围。
保存的 checkpoint 包含这些动作参数、关节顺序和观测布局。

## 单帧观测

| 下标 | 内容 | 缩放 |
|---|---|---|
| 0–2 | 机身坐标系角速度 | 0.25 |
| 3–5 | 机身坐标系投影重力 | 1 |
| 6–11 | 用户命令 vx、vy、wz、FR dx、dy、dz | 2、2、0.25、1、1、1 |
| 12–23 | 关节位置减去公共默认角度 | 1 |
| 24–35 | 关节速度 | 0.05 |
| 36–47 | 上一个控制周期实际下发的公共动作 | 1 |

观测截断至 `[-100, 100]`。六维控制命令使用四足统一零点，
由操作端处理快捷键后明确传入。`student_observations` 不读取 command manager。
不能把教师起身／下降期间写入 `term.command` 的参考目标当作用户命令，
也不能传入 owner、phase、blend、gait phase 或参考播放进度。

R 等键盘操作由操作端处理并转译为六维控制命令，student 接收处理后的
速度和 FR 目标，不增加按键或返回请求输入。
实际上一帧动作在 teacher 数据采集时来自 teacher，student 自主执行时
来自 student 实际下发后的动作历史。

## 推理接口

```python
from src.tasks.fsm_student import (
    GRUStudent, GRUStudentConfig, GRUStudentPolicy, student_observations,
)

action_term = env.action_manager.get_term('joint_pos')
cfg = GRUStudentConfig(
    default_angles=tuple(action_term.default_angles[0].tolist()),
    action_scale=action_term.cfg.scale,
)
model = GRUStudent(cfg).to(env.device)
policy = GRUStudentPolicy(model, num_envs=env.num_envs)

# 如有环境物理复位，先清空对应环境的记忆。
policy.reset(env_ids)
# 在每个 50 Hz 控制周期开始、动作下发之前调用一次。
obs = student_observations(env, user_command)
actions = policy(obs)  # [num_envs, 12]；此处新建模型尚未训练。
```

也可以省略显式 `reset`，在该帧唯一一次推理中传入
`policy(obs, reset_mask=episode_starts)`，掩码形状为 `[num_envs]`、类型为 bool。
两种复位方式二选一；每个控制周期只调用一次 policy。

普通命令变化、支撑切换和起身／下降接管不清空记忆。
只在环境物理复位或开始独立新轨迹时清空。
键盘脚本中的 `FullChainStudentPolicy` 已接入 student 控制和逐环境 teacher 暖启动。
默认每次物理复位后的前 1 秒由 teacher 控制，student 同时更新 hidden；随后交接给 student，
交接时保留 hidden。在 50 Hz 控制频率下，这对应前 50 个控制周期。
以下命令可加载已有 BC 权重试用：

```bash
python scripts/keyboard_transition_fsm.py \
  --device cuda:0 --initial-state biped \
  --student-checkpoint-file outputs/fsm_student_analysis_20261015/results/student.bc.pt \
  --student-warm-start-seconds 1.0
```

`--student-warm-start-seconds 0` 关闭暖启动。界面显示 `TEACHER WARM-UP` 或 `STUDENT GRU`。
暖启动发生在物理复位之后，普通 FSM 状态切换不重启暖启动。
已有权重可直接使用该推理流程；全帧 loss 的学习效果需要重新训练后才会体现。

## 序列 BC 接口

```python
import torch
from src.tasks.fsm_student.distillation import masked_sequence_loss

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# obs_seq: [batch, time, 48]
# teacher_actions: [batch, time, 12]
# episode_starts: [batch, time]，bool，复位后的第一帧为 True。
# valid_mask: [batch, time]，bool；右侧补齐的帧为 False。
predicted, next_hidden = model(obs_seq, episode_starts=episode_starts)
loss = masked_sequence_loss(predicted, teacher_actions, valid_mask)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

必须保留轨迹的时间顺序，并按连续片段采样。
网络没有固定的 50 帧窗口；当前 BC／DAgger 训练器使用最长 32 帧的有效历史，
短片段补齐并掩码，所有真实帧参与监督。随机训练窗口之间不继承 hidden。
对同一环境的连续片段做截断 BPTT 时，可将 `next_hidden.detach()`
传给下一片段；随机抽取无关片段时不能复用上一批的隐藏状态。

时间对齐：先记录当前观测，其中动作历史是上一步实际动作，再执行本
控制周期的教师动作，记录本步实际下发的目标作为该观测的监督标签。
若有子步延迟／限速，必须明确采用的控制周期标签采样规则，并使用同一
时间定义训练和评估。标签应换算至公共动作坐标，不应直接使用某个教师
内部、零点不同的 actor 输出。

```python
model.save('student.pt')
restored = GRUStudent.load('student.pt', device='cuda:0')
policy = GRUStudentPolicy(restored, num_envs=env.num_envs)
```

Checkpoint 不保存任何环境的 GRU 运行记忆，加载后从零记忆开始。
当前 checkpoint 格式为 version 2，对应 48 维观测布局。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/student_mpl \
  XDG_CACHE_HOME=/tmp/student_cache python -m pytest -q \
  tests/test_fsm_student.py tests/test_fsm_student_runtime.py
```

单元测试验证当前帧相同、历史不同的序列可以学出不同动作；整段序列
计算与逐帧推理一致；环境复位不会串扰其他环境；训练梯度和 checkpoint
正常。CPU 仿真测试读取真实完整教师环境的传感器和公共动作接口，检查
student 推理及其用户命令与教师参考目标的隔离。这些验证不代表 student
已学会机器狗任务。
