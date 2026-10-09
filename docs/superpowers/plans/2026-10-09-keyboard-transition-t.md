# Keyboard transition_t Implementation Plan

**Goal:** 用既有键盘入口真实运行用户指定的 loco → transition_t → biped 策略链。

**Architecture:** 新增互斥 checkpoint 参数，旧 bridge 入口保持兼容。新环境以 transition_t 契约为基础；motion 命令空闲时暂停，twist 命令负责逐世界状态机，动作项负责教师观测/重基与按策略执行延迟；原 runner 校验并加载新模型。独立 viewer 及 scripted headless 回放复用同一个环境和策略。

**Tech Stack:** Python、mjlab、MuJoCo/Warp、PyTorch、pytest。

- [x] 在 `tests/test_keyboard_transition_t.py` 添加 CLI 互斥、75/192 环境布局、参考暂停、真实状态切入不改物理/历史、教师观测重基、站稳/严格误差区分、超时与复位行为测试；先运行看到缺失功能的失败。
- [x] 在 `scripts/keyboard_teacher.py` 添加 `--transition-checkpoint-file` 并选择新后端；旧参数仍选择旧 FSM。
- [x] 在 `src/tasks/transition_t/keyboard.py` 实现 `keyboard_transition_env_cfg`、暂停 motion、手动 twist FSM、教师动作路由。原 checkpoint 契约由 `TransitionOnPolicyRunner.load` 原样校验。
- [x] 在 `scripts/keyboard_transition_fsm.py` 实现同一策略链的原生键盘 viewer 和 bounded headless 回放；按仿真步推进计时；显示加载模型、状态、参考进度、站稳门槛、严格足端误差、失败原因。
- [x] 运行 `PYTHONPATH=.:/tmp/keyboard-transition-test-deps python -m pytest -q tests/test_keyboard_transition_t.py tests/test_keyboard_teacher.py tests/test_transition_t_*.py`，验证已有键盘命令与新过渡契约。
- [x] 用用户指定三个真实 checkpoint 跑 CPU scripted 回放，保存结果；检查触发前后 root/joint/qvel/上一动作连续性及真实接管/失败行为。
- [x] 请求独立代码审查，处理问题，补充 README 和可直接执行的用户命令。

验证记录：65 tests + 3 subtests 通过；新增局部 reset 计时回归已先失败再修复。独立审查的 gravity 子集切片及局部 reset 全局计时问题均已处理。早期真实 checkpoint CPU 回放曾完成 TRIPOD → TRANSITION → BIPED，双足接管后运行超过 3 s 无复位；最终版补齐加载后完整 reset，视频回放改为过渡超时，不能将早期结果视为最终版成功证据。最终版 2.22 s 触发、12.22 s 超时，最后缺失双后脚支撑门槛；未发生物理/跟踪复位。切入 qpos/qvel/raw_action/previous_action/delay_steps/motor_offsets 差值均为 0。报告保留成功与失败及不同复位序列的区别，未放宽门槛。
