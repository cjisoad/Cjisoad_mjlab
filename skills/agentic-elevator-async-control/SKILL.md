---
name: agentic-elevator-async-control
description: 在 unitree_rl_mjlab 的 MuJoCo 楼房中进行异步机器狗控制实验。当用户要求常驻规划智能体、推理与动作执行重叠或测试异步控制延迟时使用。使用第一视角 RGBD 和原有 teacher/FSM；普通同步楼房任务使用原操作技能。
---

# 实验性异步机器狗控制

这是 `agentic-elevator-control` 的实验副本。机器人、坐标、按钮接触和任务完成
语义沿用 [原控制约定](references/synchronous-control.md) 与
[原接口参考](references/local-api.md)；本技能的异步流程替代原约定中的
“每步等待模型再下发”。操作前读取这些机器人约定，以及
[异步接口](references/async-api.md)。

使用 **一个常驻模型规划智能体 + 一个本地执行器**。模型给出短段计划，本地
执行器独占物理，逐条检查反馈。不要让多个智能体直接调用 `/action`，也不要
每段重新 spawn。规划器是 Codex 子智能体，不是新增的外部模型 API 服务。

## 启动与执行权

从包含本技能的 checkout 根目录运行，可用 `git rev-parse --show-toplevel` 定位。复用已有 GPU 服务，先检查 `/health`
及 `/pipeline/status`。实验接口返回 404 时，按已授权的实验需求重启为：

```bash
python scripts/run_agentic_elevator.py --device cuda:0 --initial-state quadruped \
  --viewer native --experimental-pipeline
```

按用户偏好选择 `native`、`web` 或 `none`，只选一种人类第三视角显示。
重启会创建新 episode；不要为执行新计划自动 `/reset`。

确定当前任务与可行空间边界后，由主智能体用客户端 `start` 建立有动作预算的
任务。启动时已有物理动作在执行会被拒绝，先等待实际反馈。实验启用但没有
活动任务时仍可用原同步接口；任务活动时直接动作、reset、按钮测试入口返回
409，本地执行线程拥有唯一动作执行权。

## 常驻规划器

主智能体只创建一个 `collaboration.spawn_agent`，使用 `fork_turns="none"`，
传入目标、仓库/技能绝对路径、服务地址和动作预算。让它读取本技能及
`references/planner.md`，按下面流程持续规划到完成/停止。以后复用该智能体的
`followup_task`。规划器自身不创建子智能体；主智能体保持可接收用户修改或停止。

1. 用客户端 `observe` 获得规划请求和下载的图像。它读取独立快照，不等待
   当前 `/action` 锁；`--after`/`--wait` 可等待新控制步发布的快照，
   `--after-frame` 可等待更新的 RGBD，而不只是文字状态更新。
2. 把最新 RGB 和必要的 depth visualization 实际输入模型上下文；测距读取
   米制 depth NPY。按 episode/frame ID 去重。**下载文件不等于模型已经看过。**
   核对图像 capture_sim_time 与当前结构化 sim_time。渲染不可用或缺少必要视觉
   时停止并报告，不能把服务真值控件列表当成已看到目标。
3. 看过该请求中的所有帧后，用 `ack <request_file>` 确认该请求的完整 frame ID
   列表。实验 ACK 属于规划请求，不推进原 `/observe` 的 default consumer。
4. 只生成符合 schema 的 JSON 计划，不生成执行代码或长解释。每份计划 1–5
   条，每条 `0<t<=1`。用 `request_id` 绑定已交付观测；TTL 至多为源 sim_time+6，
   且计划超过 120 秒墙钟会过期。任务边界、位移及朝向偏差仍可能使它提前失效。
5. 用客户端 `submit <plan_file>` 提交，然后**立即读取下一份快照并开始下一轮
   推理**。不在每步间等待/调用模型，不等整批反馈才重规划。客户端返回 400
   表示计划未接纳，重新观察；请求结果不明确时查 status/events，不自动重放。
6. 下次计划在动作边界接替当前未执行后缀，不中断正在完成的物理步。每次规划
   结合 `execution` 中当前进度，避免重复执行已完成子目标。接近按钮、门口和
   姿态过渡时缩短计划，目标变化及时重新规划。

执行器会检查 owner、episode、计划来源/过期、位置边界和实际失败。有效计划
耗尽则在 QUADRUPED/BIPED 保持机器狗，等待规划；场景和 RISE/DOWN 过渡仍继续运行，不会为填满队列盲目延续旧速度。主智能体可随时
`stop`：取消标志在控制步边界截断当前动作及剩余队列，随后保留真实状态。

## 动作和完成语义

- `cmd_v` 沿用实时锚点系速度。此实验限制平移分量 <=.6 m/s，偏航 <=1 rad/s。
- 默认足端用 `cmd_p=[0,0,0]`。有非零足端目标时输出 `foot_target_w` 世界坐标，
  执行器在每条动作开始时使用**最新锚点**转换；不要重放旧锚点下的非零 cmd_p。
- `expected_owner` 与源观测 FSM 一致，变化后该计划失效。RISE/DOWN 中命令可能
  未被接受，后台仍会推进过渡；以最新快照核对状态，不能宣称外部目标生效。
- 按钮仍必须由机器狗右前肢实际接触、观察事件/灯状态后释放；不用 `/button`
  或 `/scene-click` 代替机器狗操作。
- 最后一份计划可 `terminal=true`，最后一条必须零速度。它表示规划器认为任务
  已完成，本地执行完成不证明用户目标成立；主智能体需核对实际位置/灯状态。
- `reason=max_actions` 只表示预算耗尽；`failed`/`stopped` 不是任务成功。
  不确定状态先观察，不能自动重试可能已经执行的动作。

最终报告任务状态、最后有效反馈、观测→计划时间、执行器停顿和推理/执行重叠。
模型耗时用完整状态/图像交付后到计划就绪的日志衡量，单列下载/查看工具开销。
未做同轨迹对照时不报告加速倍数。
