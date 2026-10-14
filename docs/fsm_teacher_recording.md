# FSM teacher 数据录制

运行 `scripts/record_fsm_teacher.py`，自动执行已有四足／三足／双足教师和起身、下降教师。
输出 48 维 student 观测、公共坐标下实际动作标签、完整连续轨迹，以及类别均衡的 GRU 窗口索引。
不需要打开 viewer 或手动按键。

```bash
python scripts/record_fsm_teacher.py \
  --device cuda:0 \
  --num-envs 16 \
  --windows-per-category 2000 \
  --sequence-length 32 \
  --max-batches 100 \
  --output-dir outputs/fsm_teacher_dataset
```

使用与 `keyboard_transition_fsm.py` 相同的默认教师 checkpoint 和动作参考。
可传入 `--quad-checkpoint-file`、`--biped-checkpoint-file`、
`--rise-checkpoint-file`、`--down-checkpoint-file`、`--rise-motion-file`、
`--motion-file` 和 `--entry-bank` 覆盖路径。
输出目录必须不存在，避免覆盖之前的数据。

## 平衡方式

| ID | 类别 | 默认训练窗口 | 默认验证窗口 |
|---|---|---:|---:|
| 0 | 四足行走 `quadruped_walk` | 2000 | 400 |
| 1 | 四足静置 `quadruped_stand` | 2000 | 400 |
| 2 | 三足行走 `tripod_walk` | 2000 | 400 |
| 3 | 三足操作 `tripod_manipulation` | 2000 | 400 |
| 4 | 双足行走 `biped_walk` | 2000 | 400 |
| 5 | 双足操作 `biped_manipulation` | 2000 | 400 |
| 6 | 四足→三足 `quadruped_to_tripod` | 2000 | 400 |
| 7 | 三足→四足 `tripod_to_quadruped` | 2000 | 400 |
| 8 | 起身 `rise` | 2000 | 400 |
| 9 | 下降 `down` | 2000 | 400 |

四足状态提出 FR 抬腿目标后进入三足，所以四足操作的准备／抬腿过程归入
四足→三足，稳定后的操作归入三足操作。各个切换方向分别设配额，避免长时间
行走掩盖短暂过渡。FR 已抬起后的重新定位归入三足操作／行走，不重复算初次抬腿。

`--windows-per-category` 为**每类训练窗口数**。
验证窗口数为 `ceil(训练配额 × --validation-fraction)`，默认比例 0.2。
训练与验证按整条轨迹分开，分别达到各类别等量配额。
每个窗口长度为 `--sequence-length`；默认 32 帧在 50 Hz 下提供约 0.64 秒历史。
这是训练时展开的长度，在线 GRU 仍每步只输入一帧并维护记忆。

原始轨迹按时间顺序完整保留。使用蓄水池抽样生成 `windows.npz`，
每类按窗口**末帧**标签达到配额。窗口的前面帧可跨越状态切换，作为 GRU 上下文。
训练时均衡损失应落在末帧；若对所有上下文帧等权监督，类别占比会改变。
不同窗口可能重叠，配额不是独立轨迹数量；正式训练应提高配额、使用多个种子，
并检查 manifest 中的轨迹数和候选帧数。验证划分验证的是本脚本命令分布中的新轨迹，
不等同于独立地形、参考动作或起始状态的泛化评估。

## 自动命令和切换

脚本按当前缺口选择场景：

- 四足场景：站稳、随机速度行走、静置、FR 抬腿、三足行走、FR 操作、放下 FR。
- 起身场景：先稳定三足，再提高 FR 到 0.32 米，最后请求 0.36 米触发起身；
  完成后采集双足行走、操作、下降和回到四足。
- 双足场景：从现有物理双足入口复位，采集行走、操作、下降及放腿。

速度和低位 FR 目标在环境间随机变化；操作目标沿小幅平滑曲线移动。
默认动作幅度偏保守。四足场景交替采用静止／带速度的抬腿、返回。
类别中的“行走”按非零速度请求区分，并额外保存实际机身速度供后续质量筛选。
“操作”指 FR 空间目标跟踪，没有物体交互／接触操作场景。

同一批环境同步执行起身／下降，避免原教师世界的全局参考切换相互干扰。
完成条件结合最小等待时长与真实 FSM owner；返回阶段等待 QUAD。
若超时会在 manifest 中标记。失败环境退出本批有效采集，下一批再复位。
默认剔除终止帧、失败前 0.5 秒及本批该环境之后所有帧，可用
`--failure-tail-seconds` 调整。

R 等返回操作由操作端转译为普通六维命令，数据不增加 R 位。
起身／下降期间保留当时请求，student 不读取教师内部的参考目标。
owner、phase、blend 仅作元数据，不拼入 student 观测。

## 文件和时间对齐

- `shard_00000.npz` 等：每批连续轨迹。
- `windows.npz`：均衡训练／验证窗口索引。
- `manifest.json`：类别、配额、实际计数、候选帧数、每批失败／超时情况、
  观测／动作约定、步长、随机种子及教师文件和主要代码 SHA256。

分片使用 `[T, N, ...]` 布局：

| 字段 | 形状 | 含义 |
|---|---|---|
| `observations` | T,N,48 | 本步执行前的 student 观测，动作历史来自上一控制周期 |
| `commands` | T,N,6 | 操作端 vx、vy、wz、FR dx、dy、dz |
| `actions` | T,N,12 | 本控制周期最后一个物理子步实际应用的公共动作，BC 标签 |
| `joint_targets` | T,N,12 | 对应最后子步的实际 PD 目标 |
| `substep_actions` | T,N,S,12 | 全部物理子步实际动作；默认 S=4 |
| `motor_offsets` | T,N,12 | 电机偏移，便于核验 PD 目标与动作坐标 |
| `category` / `owner` / `phase` / `blend` | T,N | 类别及 teacher FSM 元数据 |
| `actual_velocity` | T,N,3 | 执行前机身坐标系线速度 |
| `valid` / `done` / `episode_starts` | T,N | 有效帧、终止、轨迹起点 |
| `trajectory_id` / `split` | N | 轨迹 ID 和划分；0=训练，1=验证 |

动作在每个 `_write_target` 完成后捕获，因此包含已有延迟／限速器的结果，
而且不会被 `env.step` 最后自动复位清零。窗口不跨物理复位或无效帧。

窗口索引包含 `shard`、`env`、`start`、`stop`、`category`、`split`、`trajectory_id`。
`stop` 不包含在切片内，末帧位置为 `stop - 1`。例如：

```python
import numpy as np
import torch
import torch.nn.functional as F

with np.load('outputs/fsm_teacher_dataset/windows.npz') as index:
    row = np.flatnonzero(index['split'] == 0)[0]
    shard, env, start, stop = [int(index[key][row])
        for key in ('shard', 'env', 'start', 'stop')]
with np.load(f'outputs/fsm_teacher_dataset/shard_{shard:05d}.npz') as data:
    obs = torch.from_numpy(data['observations'][start:stop, env].copy())[None]
    target = torch.from_numpy(data['actions'][stop-1, env].copy())[None]
predicted, hidden = model(obs)  # 随机窗口从零 hidden 开始，前面帧提供上下文。
loss = F.smooth_l1_loss(predicted[:, -1], target)
```

这是读取／监督约定示例，训练器还需要批量采样、优化和评估。

## 小规模检查与中断

```bash
python scripts/record_fsm_teacher.py \
  --device cpu --num-envs 2 \
  --windows-per-category 4 --sequence-length 8 \
  --max-batches 12 \
  --output-dir outputs/fsm_teacher_recording_smoke
```

每批完成后原子写入分片、索引和 manifest。Ctrl+C 会保留当前已完成的
控制周期并更新索引，`interrupted=true`。达到全部配额时 `complete=true` 且返回 0。
达到批次数上限或被中断而配额不足时打印各类别缺口，返回 2；
`--allow-partial` 可显式允许不完整数据返回 0，manifest 仍记录真实完整性。
当前脚本不支持在同一目录恢复录制。

## 本次实现验证

2026-10-14 在 CPU 上执行了实际采集：2 个环境、每类 120 个训练窗口、
每类 24 个验证窗口、8 帧历史。保存 4 个分片、8 条轨迹、6996 个环境帧，
均衡索引共 1440 个窗口，全部十类配额达成，零失败、零超时。
其中带速度的四足→三足、三足→四足分别包含 104、111 个有效帧，
根据机身速度与投影重力计算的实际水平速度中位数为 0.158、0.142 m/s。

保存后独立核验了：文件 SHA256、有限数值、动作与子步末帧一致、
观测动作历史与上一帧标签一致、PD 目标换算、窗口边界、轨迹划分及每类配额。
`tests/test_fsm_teacher_recording.py`、`tests/test_fsm_student.py` 和
`tests/test_fsm_student_runtime.py` 合计 35 项测试通过。
本次运行验证了采集和数据约定；尚未用这些数据训练 student。
