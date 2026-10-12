# 四足教师单独播放与全链路行走差异

日期：2026-10-12。

## 结论

两个入口使用同一个 `mark/quad_teacher.pt`，观测公式和网络计算一致。
主要差异发生在动作执行：完整链路原先把过渡策略的 FR 关节目标限速器
也施加给四足教师，而单独教师没有限速器。这改变四足行走中 FR 的运动，
使 base 产生显著上下波动。

已在完整播放器内修复源教师的动作执行，并对齐键盘速度范围和源命令的
速度渐变。起身、下降、双足教师继续走原有执行路径，训练配置和旧版
`transition_t.keyboard` 播放器没有修改。

## 模型和输入核对

核对入口：

```bash
python scripts/keyboard_teacher.py loco_pedipulation_t \
  --checkpoint-file mark/quad_teacher.pt --device cuda:0 \
  --num-envs 1 --workspace quadruped

python scripts/keyboard_transition_fsm.py \
  --device cuda:0 --initial-state quadruped
```

两个 checkpoint 均解析到本仓库的 `mark/quad_teacher.pt`，SHA-256：

```text
1176fb6c71ba5c05c7d2c8ac36f6492f9b3dd37ba8dcafc172e2c3da0bc5d0fd
```

四足教师实际输入为 54 维：机身角速度 3、投影重力 3、六维命令 6、
相对关节位置 12、关节速度 12、上一动作 12、锚定坐标系机身线速度 3、
步态 sin/cos 2、支撑 blend 1。完整链路的过渡 actor75 观测没有直接
送给四足教师，四足动作由 `AlignedKeyboardTeacher` 单独查询 54 维输入。

实际 GPU 运行中，在同一状态上比较原生 `loco_policy_state(add_noise=False)`
与 `teacher_observations()`，所有 54 维的最大差为 0；单独教师 runner
与冻结教师 actor 对相同输入的输出最大差也为 0。新回归测试另外使用
非零命令、动作历史、步态相位和 blend 检查逐项一致。

这不表示两个独立运行过程中的每一帧输入都相同：限速器改变实际关节目标，
使后续关节状态、机身速度和上一动作历史发生变化，造成闭环轨迹偏离。

## 具体执行差异和修复

| 项目 | 单独四足教师 | 完整链路修复前 | 完整链路修复后 |
|---|---|---|---|
| FR 目标速度限制 | 无 | 2 rad/s | 四足行走不施加；三足及其他 actor 保留 2 rad/s |
| FR 目标加速度限制 | 无 | 30 rad/s² | 四足行走不施加；三足及其他 actor 保留 30 rad/s² |
| 其他腿的目标限制 | 无 | 无 | 无 |
| vx 键盘范围 | [-0.5, 1.0] | [-0.2, 0.6] | [-0.5, 1.0] |
| wz 键盘范围 | [-0.8, 0.8] | [-1.0, 1.0] | [-0.8, 0.8] |
| 四足速度命令 | 源命令渐变 | 提交时直接覆盖输出 | 使用源命令渐变 |
| 控制周期 / 物理周期 | 0.02 / 0.005 s | 0.02 / 0.005 s | 相同 |

完整播放器使用局部 `FullChainAction`，按 world 的 owner 和源教师支撑状态
选择行为。只有 `owner == QUADRUPED && phase == QUAD && !enabled` 时，
经过原有子步延迟选择的教师目标才直接写入 PD；同时用同一个
governor 缓存最后实际写入的目标位置，并记录受速度界限约束的差分估计。
估计用于下个受限 actor 的 governor 重入，不代表源教师期间的目标运动
始终受限。切换时没有复位姿态、动作历史或 governor 位置；三足／起身／下降
从最近实际目标继续按原限速、限加速度规则执行。

按使用反馈，FR 操作期间保留限速：新的非零 FR 请求即时启用，覆盖
PREPARE、REACH、HOLD，以及 R 落脚后的 RETURN／RECOVER；只有源状态机
确认恢复 QUAD 才解除。不能用瞬时 FR 接触状态启停，因为正常四足步态中
FR 也会周期性离地。其他腿仍不限速，起身、下降和双足执行方式不变。

两入口的仿真环境仍非完全相同。单独教师保留源任务启动随机化，包括质量、
质心、PD 参数、motor offset、关节摩擦及阻尼；全链路 play 使用过渡任务的
名义参数，没有这些启动随机化。实际对照中固定 armature 和周期相同，
质量、质心、摩擦、阻尼和 motor offset 有差异。本次没有修改这些物理
参数；在同一个全链路物理环境中仅移除 FR governor 就消除了主要抖动，
因此无需通过修改这些参数来解释或补偿本次故障。

## 实际 GPU 对照

全部使用实际 checkpoint 和 cuda:0，无窗口，每组运行 12 秒。静置 2 秒，
命令前进 0.4 m/s 至第 10 秒，再停止；统计第 4 至 10 秒的稳定行走区间。
单独教师初始朝向有随机化，日志中的世界 X 速度不应直接与全链路的
世界 X 速度比较；这里比较世界 Z 高度和垂直速度。

| 运行方式 | base 高度标准差 | 高度峰峰值 | 垂直速度 RMS |
|---|---:|---:|---:|
| 单独四足教师 | 3.19 mm | 10.41 mm | 0.0378 m/s |
| 完整链路修复前 | 14.18 mm | 46.77 mm | 0.1669 m/s |
| 完整链路仅诊断绕过限速器 | 2.65 mm | 8.42 mm | 0.0318 m/s |
| 完整链路正式修复 | 2.65 mm | 8.42 mm | 0.0318 m/s |

正式修复相对原完整链路，高度峰峰值降低约 82%，垂直速度 RMS 降低约 81%。
所有对照运行都没有自动复位。

修复后额外验证：

- FR 抬脚 → 三脚保持 → R 恢复四脚，共 8.40 秒；最终四脚接触，最后
  半秒 FR 接触率 100%，未复位。
- 四足 → 起身 → 双足 → R 下降 → 四足，共 17.10 秒；最终继续四足保持
  3 秒，phase=QUAD、blend=0、四脚接触，未复位。
- 两次过渡启动前后 qpos、qvel、当前／前一动作、governor 位置和速度的
  最大差值均为 0。
- 相关回归 49 项通过，一条第三方 torch.jit.script 弃用警告。新测试
  覆盖实际 PD 目标、子步延迟、动作历史、governor 重入、速度范围和渐变、
  原生／适配器 54 维观测一致，以及 FR 请求首帧、三足操作各阶段、R 返回
  期间继续限速和恢复 QUAD 后解除。

三足选择性限速更新后，额外使用真实 GPU 模型连续运行 14 秒：四足前进、
Q 抬 FR 至 0.145 m、三足以 0.2 m/s 命令前进、R 落脚恢复四足。逐子步
检查 1504 次受限执行，FR 目标速度最大 2.0 rad/s，目标加速度最大
30.000007 rad/s²（浮点误差）；另有 1296 次四足子步确认原始目标直接
到达 PD。三足稳定行走统计区间内前进 0.597 m，FR 接触率为 0%，最终
恢复 QUAD、blend=0、四脚接触，没有自动复位。新逻辑下重新运行的四足
12 秒对照与上述正式修复数值一致；完整起身／下降 17.10 秒回归也再次通过。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/keyboard_full_chain_mpl \
python -m pytest tests/test_keyboard_full_chain_handoff.py \
  tests/test_keyboard_full_chain.py tests/test_keyboard_transition_t.py -q
```

原始日志、JSON、诊断脚本和高度／垂直速度对比图保存在：
`outputs/keyboard_teacher_walk_parity_20261012/`。
图为 `base_height_comparison.png`；汇总为 `summary.json`。
本次选择性限速记录为 `tripod_limits_result.json`、`tripod_limits_cuda.log`、
`tripod_limits_tests.log`；复现脚本为 `verify_tripod_limits.py`。
