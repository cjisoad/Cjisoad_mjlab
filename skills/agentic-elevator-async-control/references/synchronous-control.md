---
name: agentic-elevator-control
description: 使用第一视角 RGBD 和 fsm-teacher 运控工具，在 unitree_rl_mjlab 的三层楼房 MuJoCo 仿真中实际控制机器狗移动、乘电梯、操作按钮和灯光。适用于要求机器狗执行楼房任务的指令；不用于实物机器人、策略训练、普通代码修改或仅查看仿真画面。
---

# MuJoCo 机器狗楼房操作

将用户任务转成「观测 → 单次有界运控 → 检查反馈 → 再观测」的执行循环，直到任务完成或出现明确阻塞。使用已有 fsm-teacher；它联合处理锚点系速度与右前肢目标，并自动调整四足、三足、双足及起立/落地过渡。

## 接入已有环境

在包含本技能的 `unitree_rl_mjlab` checkout 根目录工作；可用 `git rev-parse --show-toplevel` 定位，用户指定其他 checkout 时使用指定路径。先读取 [本地接口参考](references/local-api.md)，核对实际服务地址及返回字段。

优先复用 `http://127.0.0.1:8766` 的已有服务，通过 `GET /health` 检查。服务未启动且当前任务需要运行环境时，从仓库根目录启动：

```bash
python scripts/run_agentic_elevator.py --device cuda:0 --initial-state quadruped
```

可按用户要求使用 `biped` 或其他设备。正确拼写是 `quadruped`；描述文件中的 `quadrupe` 是笔误。`scripts/keyboard_transition_fsm.py` 是键盘测试入口，模型操作应接入上述同一运控链的 HTTP 服务。

## 每轮执行

1. 读取 `/observe` 或上一次 `/action` 返回的 `observation`。使用 `robot.position_w`、实时 `robot.anchor`、`robot.fr_position_w`、`robot.fsm` 和 `controls`，结合最近至多两张新的第一视角 RGBD。图像每 **0.5 秒仿真时间**采样，返回 0 或 1 帧属于正常情况。区分图像捕获时间与当前结构化观测时间。
2. 通过资源 URI 读取新帧，让 RGB 和必要的深度可视化实际进入模型上下文；需要测距时读取米制深度。按 `episode_id` 与 `frame_id` 去重。成功输入上下文后才 `/ack`，使用该批次所属的 consumer 和 `next_cursor`。当前实现 `/action` 返回的帧属于 `default` consumer，推荐整个操作会话都用 `default`，避免重复消费。
3. 确定当前子目标：接近目标、对齐、接触、释放、进出轿厢或等待电梯。控件位置来自仿真世界坐标真值，可能在画面外；不能仅因列表中存在就认定可见或可触达。移动控件的位置应每轮更新。
4. 生成一个只含 `cmd_v`、`cmd_p`、`t` 的命令，并实际调用 `/action`。每个运动决策都明确这三个值。速度/位置都要是三个有限数，执行时长满足 **`0 < t <= 1` 秒**且至少覆盖一个控制步。靠近控件、门口或发生姿态过渡时缩短单次动作，根据反馈继续。
5. 检查 `command_accepted`、`owner_before`、`owner_after`、`executed_duration`、`stop_reason`、`events`、`elevator` 和新观测。请求值不代表实际执行结果；下一轮使用返回的实际状态规划。

正常服务在两次 `/action` 之间仍运行后台循环：`QUADRUPED`/`BIPED` 时保持机器狗的位置、姿态、速度及关节状态，电梯、门和按钮继续推进；`RISE`/`DOWN` 时原 FSM 策略会自主推进过渡，完成后再保持机器狗。读观测、查看图像和现实时间等待期间，场景时间及过渡状态可能变化。`t<=1` 只限制本次动作请求，不限制整个起立/落地过程。下一步必须使用最新观测，不能假定上一条动作返回的 owner 或移动按钮位置仍有效。`--smoke` 不启动后台循环。稳定等待如需提交动作，令 `cmd_v=[0,0,0]`，`cmd_p` 使用适合当前状态的保持目标。

## 坐标与命令

| 字段 | 实际约定 |
| --- | --- |
| `cmd_v` | 锚点系 `[v_x, v_y, v_yaw]`，线速度单位 m/s，偏航角速度 rad/s |
| `cmd_p` | 锚点系右前肢末端相对**四足站姿零点**的 `[dx,dy,dz]`，单位 m；不是世界坐标，也不是上一轮目标的增量 |
| `t` | 本轮请求推进的仿真秒数，实际时长以 `executed_duration` 为准 |

设 `anchor=observation.robot.anchor`，世界系右前肢目标为 `target_w`：

```python
import numpy as np

R = np.asarray(anchor['R_world_from_anchor'])
origin = np.asarray(anchor['origin_w'])
zero = np.asarray(anchor['fr_zero_anchor'])
cmd_p = (R.T @ (np.asarray(target_w) - origin) - zero).tolist()
# 反向关系：target_w = origin + R @ (zero + np.asarray(cmd_p))
```

仓库已提供等价函数 `src.agentic_robot.observation.state.world_to_command`。始终使用最新锚点矩阵，不能用身体姿态或固定世界轴代替；双足状态也沿用四足零点。`cmd_p=[0,0,0]` 表示默认零点目标，可能影响姿态模式，不表示保持当前末端位置。需要保持实际末端时，将当前 `fr_position_w` 按上述公式转换，并核对 FSM 状态。

`RISE`/`DOWN` 期间外部命令可能不被接受，后台仍会推进过渡。先重新观测 owner；确需动作时使用短时长，每轮检查反馈。不能因 `command_accepted=false` 断言仿真未推进，也不能假设提交的足端目标生效。

## 完成楼房任务

依据 `controls[].id/type/floor/target_floor` 选择呼梯、轿厢楼层或灯光开关，使用 `surface_position_w`、`normal_out_w` 和 `press_direction_w` 规划接触。先到外侧预接触位置，再以小步沿按压方向接触，看到对应 `button_press` 事件或目标状态变化后释放。不要把控件世界位置原样作为 `cmd_p`。

用户要求机器狗操作时，通过右前肢的运控动作完成。`/button`、`/scene-click` 是场景直接测试入口，不能代替机器狗接触按钮；只有用户明确要求直接测试场景按钮时才适用。

电梯任务按实际状态完成呼梯、等待开门、进入、选择楼层、等待到站、出轿厢。使用图像检查门和通路；`elevator.floor` 表示电梯楼层，不能单独证明机器狗已到达目标楼层。灯光开关是切换操作：先核对 `elevator.lamps` 或视觉证据，目标灯已处于用户要求的状态时不要再按一次：关灯目标对应 `false`，开灯目标对应 `true`。

以可观测结果判断完成：机器狗实际位置与目标楼层一致、所需进出轿厢动作已完成、目标灯状态正确。任务结束时如 FSM 可接受命令，提交一次零速度的短动作并检查结果；只在观察支持时报告成功。最终简要说明已完成内容、最近命令与关键结果，未完成时指出具体阻塞。

遇到 `stop_reason`、动作状态不明的通信中断、连续无进展或无法获得必要观测时，保留最新反馈并报告阻塞；不要盲目重放可能已执行的动作。需要恢复时先重新观测。重置会丢失当前任务进度，除非用户已要求重试或重置，不自动调用 `/reset`。
