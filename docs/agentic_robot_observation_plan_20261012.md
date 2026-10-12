# Agentic Robot 观测工具开发计划（待评审）

日期：2026-10-12

## 1. 目标与已确认范围

根据 `skil-description.md`，为大模型调用现有 fsm-teacher 提供观测：

1. 最近两组未交付的机器狗第一视角 RGBD，采样周期为 **0.5 秒仿真时间**；每次允许返回 0、1、2 组。
2. 机器狗当前世界坐标位置。
3. 全部可操纵控件当前的世界坐标位置。

用户已确认：第一版提供全部控件的仿真真值坐标，包括视野外控件，用于先验证 agent 调用控制工具的闭环。接口明确标记 `position_source=sim_ground_truth`，不把该结果作为视觉识别能力评测。

为使世界坐标能转成现有控制命令，增加必要的位姿、锚点和时间元数据。灯、电梯、按钮触发的内部状态默认进入评测日志；不自动扩充为 agent 可见的语义状态。

本文件原为方案草案；用户授权后已按 P0–P4 实施，当前完成状态与验证证据见仓库根目录的最终交付说明。

## 2. 仓库现状与接入约束

- `scripts/keyboard_transition_fsm.py:load_full_chain()` 创建 mjlab 环境并加载四足、双足、起身和下降策略。`_FullChainBehavior.submit()` 接收六维命令，过渡阶段可能忽略手工目标更新。
- `simulate/mujoco/model/three_floor_elevator/run.py` 创建独立的 `MjModel/MjData`。当前场景不含机器人，不能仅通过并行启动两个命令实现接触交互。
- 电梯控制器 `ThreeFloorElevatorController` 提供稳定的 10 个按钮逻辑名、物理按压去抖和电梯/门控制逻辑，可直接作为控件注册表的数据来源。
- 当前相机 `front_overview` 是场景总览相机，需要新增机器人机载相机。
- 现有前肢命令是相对四足默认 FR 落足点的锚点系偏移，不能把按钮的世界坐标或绝对锚点坐标原样传入。
- `src/tasks/teacher_common/anchor_frame.py` 已定义统一锚点：以重力确定竖直方向，由机身 +Y 与竖直方向确定水平前向；应复用该定义。
- 本地 mjlab 支持 `SceneCfg.spec_fn` 修改组合场景，也提供 RGB/depth 相机传感器及 GPU 状态复制到 CPU 后离屏渲染的现有实现，可作为接入基础。具体步进钩子和同步开销需在第一阶段实测。

## 3. 路线选择

| 路线 | 优点 | 代价与适用范围 |
| --- | --- | --- |
| **A：保留 mjlab/FSM，把电梯场景接入同一个物理世界（推荐）** | 最大程度复用现有策略观测、动作历史、限幅器和模态切换 | 需要将电梯控制逻辑适配到 mjlab 的状态与子步进；先验证 GPU 状态读写及渲染同步 |
| B：以独立 MuJoCo 入口为主，迁移 FSM 与策略执行 | 可直接沿用现有电梯步进和原生 RGBD 渲染 | 需重建策略观测、动作历史及关节控制契约，控制迁移工作量更大 |
| C：先在独立场景加一个可移动相机与假机器人位姿 | 能快速验证 RGBD、控件位置和帧游标接口 | 只适合作为观测模块测试夹具，不能作为真实机器狗闭环验收 |

建议采用 A，观测层保持后端适配边界。优先用冻结状态的原生 MuJoCo 离屏渲染实现 2 Hz RGBD；若实测表明复制开销或渲染兼容性不合适，再在同一接口下使用 mjlab 相机传感器。

## 4. 模块与数据流

```text
统一仿真 Runtime：机器人 + 房屋 + 电梯 + 门 + 按钮
  ├─ 现有 FSM / teacher 策略 → 机器人执行
  ├─ 电梯控制器适配 → 场景执行
  └─ 完整步进后的只读状态快照
       ├─ StateReader：机器人位姿、锚点、FR 实际位置
       ├─ ControlRegistry：全部控件接触表面与按压方向
       └─ RGBDCapture：机器人第一视角 RGB + metric depth
            → FrameBuffer：按仿真时间采样与保存
            → ObservationService.observe(cursor)
            → agent 的结构化数据 + 图像附件
```

Runtime 是唯一物理步进所有者。观测调用不自行推进仿真、不写控制量、不触发按钮。渲染快照可以有独立的 CPU `MjData`，但它只镜像当前物理状态，不进行第二套物理积分。

## 5. 观测数据定义

### 5.1 机器人与坐标系

必需输出：

- `position_w`：`base_link` 原点世界坐标，单位米。
- `orientation_wxyz`：机身四元数，顺序为 w、x、y、z。
- `anchor_origin_w`、`R_world_from_anchor`：与现有 teacher 相同的锚点原点及旋转矩阵。
- `fr_zero_anchor`：现有命令零点 `term.zero`。
- `fr_position_w`：实际 FR site 世界位置，用于核对到达情况。
- `fsm_owner` 和支撑阶段：区分四足/三足支撑以及 RISE、BIPED、DOWN，避免将过渡中未执行的新指令误判为观测错误。
- `schema_version`、`episode_id`、`snapshot_id`、`sim_time`。

给定期望 FR site 世界位置 `p_target_w`，现有命令的换算关系为：

```text
cmd_p = transpose(R_world_from_anchor)
        @ (p_target_w - anchor_origin_w)
        - fr_zero_anchor
```

这里 `p_target_w` 是足端 site 目标；按压按钮时应结合脚掌碰撞半径、按钮表面及压入方向规划接近/压入目标，不能把表面接触点直接当成足端中心目标。

速度命令沿用现有锚点系定义，单位分别为 m/s、m/s、rad/s；本阶段只记录契约，不改 teacher 的解释方式。

### 5.2 控件注册与动态坐标

共 10 个控件：

- 灯开关：`light_f1`、`light_f2`、`light_f3`。
- 候梯呼叫：`call_up_f1`、`call_up_f2`、`call_down_f2`、`call_down_f3`。
- 轿厢目的层：`cab_f1`、`cab_f2`、`cab_f3`。

每项输出稳定 ID、类型、语义楼层/目的楼层、父对象、`position_source`、可接触表面中心 `surface_position_w`、向外单位法向 `normal_out_w` 和压入方向 `press_direction_w=-normal_out_w`。

位置来自当前物理快照的变换，不能读取 XML 局部位置后直接当世界坐标，也不能只在启动时缓存。轿厢三个按钮始终随轿厢升降；`cab_f1` 的 1 是目的层，不代表按钮当前位于一楼。

为每个按钮添加命名的表面 site，或使用显式注册的局部表面偏移并通过当前 geom 变换计算。不能使用 button body 原点作为接触点：现有灯按钮和候梯按钮的 geom 都有局部偏移。注册时逐个核对法向；未来添加控件时沿用同一契约。

### 5.3 第一视角 RGBD

- 增加 `robot_ego` 相机，固定在经核对的机器人头部刚体上，朝机器狗自身前方；保持真实机载视角，在起身后随机身姿态变化。
- 第一版建议 640×480、垂直视场角 70°，均可配置；先验证四足/双足视野、自遮挡、按钮可见性，再固定默认值。
- RGB：`uint8` RGB，使用 PNG 保存。
- Depth：与 RGB 对齐的 `float32` 米制深度，用 NPY 保存并带有效像素 mask；约定为相机光轴方向 Z 深度。实现阶段用已知平面验证后端返回的是 Z 深度还是射线距离，必要时统一换算。
- 每组帧包含同一状态的 RGB/depth、采样时间、机器人位姿、相机内参 K 和 `T_world_from_camera`。相机坐标约定为 x 向右、y 向下、z 向前，并显式处理与 MuJoCo 相机轴向的转换。
- 标准 agent 图像输入返回 RGB 和带固定量程、图例的深度可视化；原始深度通过数据附件/资源读取接口访问。深度可视化不能替代原始 metric depth。
- 使用现有场景可见 geom 分组作为默认值，保留隐藏碰撞边界的现有设定。检查透明轿厢面板在 RGB 与 depth 中的行为；验收包含材质和灯光状态同步。
- 第一版不加入视觉识别、SLAM、点云重建或自动导航。

### 5.4 时间、去重与重置

- 正常采样点为 episode 内 `0, 0.5, 1.0, ...` 秒；初始化完成后采第 0 帧。
- 必须在执行动作期间按周期采样，不能等 `observe()` 被调用后才渲染，否则一次 1 秒动作无法保留中间 0.5 秒帧。
- 单次返回截至当前快照最近两组、且尚未交付给本 consumer 的帧，按时间先后排列；没有新帧时返回空数组，机器人与控件状态仍返回当前快照。
- 游标为 `(episode_id, frame_seq)`，按消费者保存。采用确认后的游标推进：返回 `next_cursor`，调用方在成功输入 agent 上下文后提交；传输失败可以重试同一批数据，并按帧 ID 避免重复输入模型。
- 慢消费者落后时返回最新两组，并报告 `dropped_frame_count`；完整历史可保存在评测日志中。
- reset 或策略环境自动 reset 必须生成新的 `episode_id`，使旧游标和旧帧失效；不能继续拼接上一个 episode 的视觉历史。
- 快照先冻结再渲染，避免 RGB、depth 和位姿来自不同物理时刻。当前结构化状态时间与历史帧时间分别标记。

## 6. 工具接口草案

建议先实现进程内接口，再加本机服务传输层。HTTP/MCP 只作为薄封装，避免绑定某个 agent SDK。

```python
observe(consumer_id, cursor=None, max_frames=2) -> ObservationPacket
```

`ObservationPacket` 顶层字段：

```text
schema_version / episode_id / snapshot_id / sim_time
robot: position_w / orientation_wxyz / anchor / fr_position_w / fsm
controls: 10 个控件的当前表面位置、法向和语义标识
frames: 0~2 个 RGBD 帧及各自的 capture_sim_time / K / camera_pose
next_cursor / dropped_frame_count / status
```

RGB、depth 和 mask 使用图像附件或资源引用，不把原始像素数组展开到文本 JSON。每个资源带 episode/frame ID、MIME/type、shape 和单位，并提供真实可读取的本地服务资源；仅返回仿真进程内文件路径不足以供远程 agent 使用。

错误处理：缺少机器人/相机/控件名称时启动失败并列明缺项；渲染失败时保留结构化观测并显式返回 `image_status=error`；无新帧标记 `no_new_frame`；两者不能混淆。观测成功与帧交付确认分离。

## 7. 分阶段实施与验收

| 阶段 | 工作 | 验收条件 |
| --- | --- | --- |
| P0：统一运行底座与可行性验证 | 将房屋、电梯、机器人合入 mjlab 单环境；适配电梯控制；接入每个物理子步的控制/更新；检查传感器匹配 | 同一世界中机器人能站立/行走并与场景碰撞；按钮真实接触可触发；机器人 actuator 写入不覆盖电梯/门，电梯受力不被清零 |
| P1：结构化观测 | 实现快照、机器人位姿/锚点、10 个控件注册与表面坐标 | 坐标换算往返通过；按钮压入、轿厢升降、机器人转向/起身后观测随实际状态更新；不重复套用 env origin |
| P2：机载 RGBD | 增加相机、固定快照渲染、内外参和原始深度保存 | 四足与双足都得到真实机载视角；RGB/depth 同帧；已知距离平面深度正确；截图中目标投影与内外参一致 |
| P3：采样与交付 | 0.5 秒调度、帧缓存、消费者游标、reset 和资源读取 | 正常返回 2/1/0 帧；重复调用无重复交付；重试不丢帧；慢消费者丢帧计数正确；跨 episode 不混帧 |
| P4：与控制工具闭环 | 增加薄的命令适配和本机观测服务；日志关联 cmd_v/cmd_p/t、执行状态和观测 | agent 或脚本能执行≤1秒动作后读观测；准确记录实际执行时长、FSM 过渡中未采纳的命令及物理失败；结果可回放 |

P0 的接触传感器要覆盖房屋各层地面、候梯平台和轿厢地板。现有 `_ground_sensor()` 只匹配 `terrain`，不能直接沿用后假定电梯里的支撑判断正确。移除不合适的默认无限地面，避免遮住场景空隙；按组合模型检查碰撞容量和现有教师/过渡跟踪对楼层高度、移动平台的假设。

P0 还需要处理电梯控制器的全量 reset、初始 ctrl/qpos 以及 CPU/GPU 状态同步。现有 `controller.reset()` 会重置整个 data，组合环境中必须采用统一 reset 顺序，不能在运行时意外清掉机器人状态。CPU 渲染镜像不能被误用为电梯物理状态的唯一来源；GPU 仿真中的 ctrl、qfrc_applied 和灯光材质变更必须实际生效。

建议第一版采用轮次推进：每次命令执行 `0<t<=1` 秒仿真时间，期间低层策略及电梯持续步进并采样，执行结束冻结世界等待下一轮。这样模型推理延迟不会导致观测陈旧或机器人在无新命令时继续运动。持续实时运行和 wall-clock 推理延迟作为后续模式；两种模式必须明确区分。

## 8. 文件组织建议

以下均为拟新增文件，最终根据 P0 接入结果调整：

```text
src/agentic_robot/
  runtime.py               # 统一步进、reset、控制边界、快照
  elevator_adapter.py      # 现有电梯逻辑与 mjlab 的适配
  observation/schema.py    # 版本化观测契约
  observation/state.py     # 机器人和锚点读取
  observation/controls.py  # 控件注册与表面变换
  observation/rgbd.py      # 第一视角渲染与标定信息
  observation/buffer.py    # 仿真时间调度和游标
  observation/service.py   # observe 与资源读取
scripts/run_agentic_elevator.py
tests/test_agentic_observation_state.py
tests/test_agentic_observation_rgbd.py
tests/test_agentic_observation_delivery.py
```

优先复用 `load_full_chain()` 中的策略加载和命令提交，不通过模拟键盘/X11 驱动 agent。仅把确实需要共享的初始化或步进接口抽出，避免伴随重构整套控制器。

## 9. 验证顺序与完成标准

1. CPU 可运行的确定性数据测试：控件注册、表面坐标、锚点转换、游标/重试/reset 语义。
2. EGL 离屏渲染集成测试：RGBD 对齐、深度单位/轴向、透明面板、自遮挡、灯开关对渲染的影响；依赖缺失时明确报告未运行。
3. `cuda:0` 单机器人接入冒烟：站立、平移、转向、三足操作、起身/下降；检查与现有教师控制行为的一致性。
4. 用脚本先验证一次真实 FR 触碰按钮，使观测、接触事件和灯/电梯变化在日志中对应，再接大模型。
5. 大模型闭环任务依次增加难度：走到一楼呼叫面板附近 → 按一楼上行按钮并等待 → 进入轿厢并选择目的层 → 到站离开 → 操作对应楼层灯开关。

脚本触碰通过是观测/物理接入的验证，大模型任务成功率单独统计。不能因为工具数据正确就宣称现有策略已经具备所有按钮的实际可达性或跨层泛化能力。

完成标准：agent 能可靠读取真实机载 RGBD、当前机器人位姿与全部控件真值坐标；帧不重复、不跨 episode 混用；世界坐标能正确转换到现有控制契约；至少一次真实接触交互可以从命令、观测和事件日志中完整复核。

## 10. 工作量预估与首个交付物

按熟悉本仓库的单人开发估算：观测模块 P1–P3 约 3–5 个工作日；统一仿真 P0 约 2–4 个工作日；闭环适配 P4 约 1–2 个工作日。合计约 6–11 个工作日，需根据 P0 中 GPU 步进、接触传感器和电梯适配结果修订。这是规划估计，不包含策略重新训练或修复策略在新场景中的能力缺口。

第一个可评审交付物建议限定为 P0/P1：一个组合场景能运行，并输出包含机器人、锚点、10 个控件动态表面位置的 `observation.json`。这个阶段通过后再加入图像采样，可尽早发现接入和坐标定义问题。
