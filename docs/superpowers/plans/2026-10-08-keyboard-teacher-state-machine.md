# 键盘三教师状态机实施计划

> 按已确认设计在当前工作区顺序实施。用户已选择0.33m，不再请求执行方式确认。

**Goal:** 在三足键盘union模式实现三足→过渡→站稳1s→双足，并保留同一仿真与动作历史。

**Architecture:** 新增仅供键盘播放的BridgeCommand/BridgePositionAction子类；保留过渡checkpoint原有契约校验，独立加载用户选择的三足/双足教师。由仿真控制步推进状态，viewer只处理键盘输入和显示。

**Tech Stack:** Python、PyTorch、mjlab、MuJoCo native viewer、X11。

## Task 1：接口与诊断

- [x] `src/tasks/pedipulation_bridge_t/experts.py`：显式支持键盘加载actor51/54，原冻结pair继续默认actor51。54维输入额外提供gait_phase/blend，验证原critic95及坐标契约。critic共享观测始终取前48维，不因actor增添gait而扩维。
- [x] `commands.py`：将已有站稳合取判据提取为具名条件字典，保留原阈值；供键盘窗口显示未满足条件。

## Task 2：键盘运行状态

- [x] 新建 `src/tasks/pedipulation_bridge_t/keyboard.py`：专用命令/动作子类；0.33m、FR误差0.10m和0.10s触发保持；过渡参考结束后连续站稳1s接管。
- [x] 三足保留原操作/回落逻辑；过渡冻结手动目标与速度命令，保留实际动量；双足从过渡最终目标恢复输入。超时标志触发viewer暂停，物理失败重置。
- [x] 每世界独立reset revision，自动复位及显式复位同步键盘目标。正常三足/双足播放不沿用训练定时终止。

## Task 3：键盘入口与界面

- [x] 新建 `scripts/keyboard_teacher_fsm.py`：加载三教师，启动native viewer，按状态切换键盘范围及速度限制，显示进度/条件并记录切换原因。
- [x] 修改 `scripts/keyboard_teacher.py`：union+loco自动进入状态机，新增bridge/biped模型与margin参数；其他模式沿用原入口。

## Task 4：文档与检查

- [x] 更新 `src/tasks/teacher_common/README.md` 与桥接任务README，给出单环境完整命令和上行模式说明。
- [x] 进行源码/语法与CLI检查，核对模型契约及变更范围。遵守本会话开发指令，不新增或运行测试；GUI实播与运行效果如未执行须明确报告。

## 完成记录

- 触发高度按用户确认设为0.33m（margin默认0.02m）。
- 只读代码审查发现并修复：失焦触发计时、超时同帧继续物理步、三足统一回落配置。复审无新的Critical/Important问题。
- 多环境诊断跟随当前查看的环境；超时必须完整复位后才能继续。
- 已执行Python语法编译、CLI帮助和git差异空白检查；未运行测试或GUI实播，因此尚未验证当前模型的实际起立成功率。

## 实播故障修正：FR误差阻止双足接管

- 用户截图：RUNNING、TRANSITION、参考100%、站稳计时0、唯一未满足条件fr_error。
- 根因：键盘接管复用了训练成功合取判据，FR误差大于0.10m持续清零站稳计时并冻结键盘；与鲁棒起立优先、允许FR跟踪偏差的用途不符。
- [x] 仅键盘子类提供handoff_conditions，排除fr_error；接管逻辑和未满足项显示使用同一组条件。
- [x] 保留FR跟踪目标和三足进入过渡的误差门槛；训练/批量评估成功判据不变。
- [x] 窗口及切换日志显示FR误差诊断值，更新使用文档。
- 修正后执行语法与差异检查、只读代码复审；未运行测试或GUI实播。
