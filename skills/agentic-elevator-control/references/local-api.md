# 本地机器人接口

执行技能前读取本文件。接口默认根地址为 `http://127.0.0.1:8766`，服务绑定本机回环地址；用户指定端口时相应替换。请求使用 JSON，动作循环串行执行，读到反馈后再决定下一步。

## 请求与返回

| 请求 | JSON 请求体 | 用途 |
| --- | --- | --- |
| `GET /health` | 无 | 返回 `{"ok":true}` 表示服务可响应 |
| `POST /observe` | `{"consumer_id":"default","max_frames":2}` | 当前结构化状态和至多两帧未确认 RGBD |
| `GET /resources/<resource_id>` | 无 | 按资源记录中的 URI 获取 PNG、NPY 或 mask |
| `POST /ack` | `{"consumer_id":"default","cursor":<该批 next_cursor>}` | 确认已经输入上下文的帧 |
| `POST /action` | `{"cmd_v":[0.2,0,0],"cmd_p":[0,0,0],"t":0.5}` | 推进本轮物理，并返回执行结果与观测 |
| `POST /reset` | `{}` | 重置机器狗和电梯，创建新 episode；仅按已授权的重置需求使用 |

示例只展示传输格式；动作数值须根据当前任务与观测决定。可以使用可用工具桥接 HTTP；没有专用工具时，用 shell 的 `curl` 或 Python 标准库发起调用。仅输出 JSON 文本不等于调用运控。

```bash
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8766/health
curl --fail --silent --show-error --max-time 30 \
  -H 'Content-Type: application/json' \
  --data '{"consumer_id":"default","max_frames":2}' \
  http://127.0.0.1:8766/observe
```

真实动作的传输形式：

```bash
curl --fail --silent --show-error --max-time 60 \
  -H 'Content-Type: application/json' \
  --data '{"cmd_v":[0.2,0,0],"cmd_p":[0,0,0],"t":0.5}' \
  http://127.0.0.1:8766/action
```

不要给动作 HTTP 调用开启自动重试。传输超时不证明动作未执行，先观察实际状态。

正常服务会在请求间推进场景：QUADRUPED/BIPED 保持机器狗状态，电梯、门和按钮继续运行；RISE/DOWN 自主执行原 FSM 过渡。`t` 和 `executed_duration` 只描述该次请求，不能作为整个过渡的时限。后续观测可能具有更新的仿真时间、FSM owner 和移动控件位置。`--smoke` 不启动后台循环。

## 观测与图像交付

观测包包括：

- `episode_id`、`snapshot_id`、`sim_time`。
- `robot.position_w`、`orientation_wxyz`、`fr_position_w`，以及 `robot.anchor.origin_w/R_world_from_anchor/fr_zero_anchor/contract`；当前锚点契约为 `teacher_body_y_cross_up_v1`。
- `robot.fsm.owner/owner_id/command_accepted/support_phase/support_blend`；owner 为 `QUADRUPED`、`RISE`、`BIPED` 或 `DOWN`。
- `controls[]`：`id`、`type`、`floor` 或 `target_floor`、`parent`、`surface_position_w`、`press_direction_w`、`normal_out_w`、`position_source=sim_ground_truth`。
- `frames[]`、`next_cursor`、`dropped_frame_count`、`image_status`，渲染失败时还有 `image_error`。

单帧带 `frame_id/frame_seq`、`capture_sim_time`、捕获时机器人状态、相机内外参与 `resources`。资源键为 `rgb`、`depth_visualization`、`depth`、`depth_mask`，每项包含 `resource_id/uri/mime_type`。

将 RGB PNG 和需要的深度可视化下载到临时目录，再用可用图像查看工具输入上下文；不要把原始图像字节当作文本打印。深度 `depth` 是 `float32` NPY，单位为米，含义为相机光轴 Z；用 `np.load(..., allow_pickle=False)` 读取。深度可视化灰度图不是原始深度数值。相机坐标轴是 `x_right,y_down,z_forward`；需要反投影时使用该帧内参和光学相机位姿。

服务会保留未确认批次，确认前再次观察可能返回同样的旧帧。`next_cursor` 是 `{episode_id,frame_seq}` 对象，只有成功输入模型上下文的批次才能确认：

```text
{"consumer_id":"default","cursor":{"episode_id":"实际 episode_id","frame_seq":实际序号}}
```

保持同一个 consumer：`/action` 内嵌观测固定使用 `default`；`/observe` 推荐显式指定 `default`。包中不保证带 `consumer_id`，所以要记住请求使用的值。若接入已有自定义 consumer，不能把它收到的游标拿去确认 `default` 的批次。`/observe` 的可选 `cursor` 只能是已确认游标，不能用它跳过 `/ack`。

本地按 `(episode_id, frame_id)` 记录已输入帧；ACK 重试不再次输入图像。episode 改变时丢弃旧游标和旧场景假设。`dropped_frame_count>0` 表示有遗漏，不能假设观测覆盖了全部中间过程。`no_new_frame` 正常；`disabled`、`unavailable`、`error` 需要按当前任务判断是否缺少必要视觉信息，不能声称看过图像。

## 动作结果与交互反馈

`/action` 返回 `cmd_v`、`cmd_p`、`requested_duration`、`executed_duration`、`steps`、`command_accepted`、`owner_before/owner_after`、`stop_reason`、`events`、`elevator`、`observation`。请求时长会量化到控制步长，必须以返回时长为准。

`events` 中按钮事件形如 `{"type":"button_press","button":"light_f2","sim_time":...}`。`elevator` 含 `floor`、`target`、`moving`、`closing`、`lamps`；`lamps` 以字符串楼层号为键，例如 `lamps["2"]`。按钮触发只能证明发生交互，任务结果仍需用目标楼层、实际机器人位置和灯状态验证。

当前控件 ID：

- 灯：`light_f1`、`light_f2`、`light_f3`。
- 上行呼梯：`call_up_f1`、`call_up_f2`。
- 下行呼梯：`call_down_f2`、`call_down_f3`。
- 轿厢目标楼层：`cab_f1`、`cab_f2`、`cab_f3`。

使用实时位置，不缓存轿厢按钮的世界坐标。预接触点可取 `surface_position_w + clearance * normal_out_w`；按压沿 `press_direction_w` 小步推进，间距和推进量按可达性、深度与接触结果调整。

## 需要核对实现时

从当前仓库读取对应文件，避免把参考说明当作运行状态：

- `scripts/run_agentic_elevator.py`：启动参数、单世界约束。
- `src/agentic_robot/server.py`：HTTP 路由及请求字段。
- `src/agentic_robot/runtime.py`：命令校验、时长量化、FSM owner、冻结及失败逻辑。
- `src/agentic_robot/observation/state.py`：锚点契约和 `world_to_command`。
- `src/agentic_robot/observation/service.py`、`buffer.py`、`rgbd.py`：帧采样、确认、资源及深度格式。
- `src/agentic_robot/observation/controls.py`、`src/agentic_robot/elevator_adapter.py`：控件语义和场景反馈。

本技能来源是仓库的 `skil-description.md`（用户称为 `skill_description.md`），接口细节根据以上现有实现补全。
