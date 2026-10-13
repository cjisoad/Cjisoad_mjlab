# 异步规划接口

运行启用 `--experimental-pipeline` 的服务，默认 `http://127.0.0.1:8766`。
统一使用仓库脚本 `scripts/agentic_pipeline_client.py`；技能包装器
`scripts/client.py` 可从任意目录调用同一实现。

```bash
python scripts/agentic_pipeline_client.py start \
  --objective '在已检查的入口平台向前走一小段，然后停止' \
  --max-actions 10 --bounds '[[3.3,4.5],[-3.5,-1.65],[0.2,0.6]]'
python scripts/agentic_pipeline_client.py observe
python scripts/agentic_pipeline_client.py observe --after 12 --wait 1
python scripts/agentic_pipeline_client.py observe --after-frame 0 --wait 2
python scripts/agentic_pipeline_client.py ack /tmp/elevator_async_planner/request_ID.json
python scripts/agentic_pipeline_client.py submit /tmp/plan.json
python scripts/agentic_pipeline_client.py status
python scripts/agentic_pipeline_client.py stop
```

示例边界仅适用于已确认的入口平台，不可用于整个楼房任务。客户端不生成模型
决策、不自动 ACK、不重试动作或计划。

| 路由 | 请求 | 说明 |
| --- | --- | --- |
| POST `/pipeline/start` | `objective,max_actions,position_bounds` | 创建任务；无计划时保持机器狗，场景与活动过渡继续运行 |
| POST `/pipeline/observe` | 可选 `after_snapshot_id,after_frame_seq,wait_s` | 独立快照；等待最多15秒；返回 request_id、task、observation、execution |
| POST `/pipeline/ack` | `request_id,frame_ids` | 确认模型实际读取的该请求全部帧，重复确认安全 |
| POST `/pipeline/plan` | 下方 schema | 提交计划，不等待动作执行完成；只保留一份待接替计划 |
| GET `/pipeline/status` | 无 | 状态、执行动作数和最多1000条时间事件 |
| POST `/pipeline/stop` | `{}` | 请求控制步边界停止，不等待整段动作锁 |
| GET `/pipeline/resources/<resource_id>` | 无 | 已发布快照/规划请求持有的图像、深度资源 |

每次请求保留至多两张最新 RGBD，资源按最新快照与最近8个请求租约保留。
第9个新请求会驱逐最老请求，所以单规划器不要积累未完成请求。无新图像时
frame ID 可以重复，按已看过的帧去重，但每份请求仍需自己的 ACK。
`published_wall_ns` 是状态发布时间，`delivered_wall_ns` 是服务交付规划包的时间，
图像有自己的 `capture_sim_time`。它们不代表模型服务端真正收到输入的时间。
等待超时会返回当前快照，因此仍需比较帧序号，不能把超时返回当成新帧。

## 计划 schema

```json
{
  "request_id": "本次规划请求的真实ID",
  "expected_owner": "QUADRUPED",
  "expires_sim_time": 6.0,
  "terminal": false,
  "commands": [
    {"cmd_v": [0.2, 0, 0], "cmd_p": [0, 0, 0], "t": 0.5},
    {"cmd_v": [0, 0, 0], "foot_target_w": [4.0, -2.8, 0.2], "t": 0.2}
  ]
}
```

第二条仅展示世界足端目标格式，不是推荐执行目标。实际坐标由新观测决定。
`cmd_p` 与 `foot_target_w` 二选一；本实验只接受零 cmd_p，非零使用世界目标。
TTL 从本请求的源 sim_time 算，最多6秒；命令本身不能执行到 TTL 之后。
每份计划1–5条，t至少一个控制步且<=1秒。平移分量<=.6、偏航<=1。

计划按请求序号拒绝迟到旧结果；episode、FSM、源位置偏差>2m、源锚点朝向
偏差>.5rad、墙钟年龄>120秒或任务边界变化使计划失效。因失效耗尽队列后
在手动 owner 保持机器狗，场景和活动过渡继续运行；规划器读取新快照重新生成。新计划只在动作边界替换剩余后缀。
`terminal=true` 最后一条必须零速度。预算耗尽也会结束，但不等于目标完成。

`status.events` 包括 observation_delivered、plan_accepted、action_dispatched、
action_returned、plan_invalidated、executor_finished。plan_accepted 中
observation_to_plan_s 包含客户端下载、图像工具、模型推理和提交传输；不能
直接称为纯模型推理时间。需结合会话日志拆出实际图像工具返回→计划输出。
