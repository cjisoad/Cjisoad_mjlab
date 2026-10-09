# 最新 transition_t 键盘三教师接入

用户于本轮已确认：在现有键盘入口接入最新 `transition_t/model_30000`，三足教师达到触发高度后，从当前真实状态启动过渡参考，在同一环境运行，满足站稳条件后交给双足教师，显示实际模型及切换诊断。

使用新增互斥参数 `--transition-checkpoint-file` 选择新后端；原 `--bridge-checkpoint-file` 保持旧后端。新后端基于 transition_t play 配置，保留 actor75/critic192、归一化、参考 SHA、名义物理参数与原 runner 的完整 checkpoint 校验。额外添加手动 loco 命令及教师动作路由，按各教师自己的默认关节角构造 51/54 维观测并重基动作。过渡阶段无动作延迟，端点教师按自身延迟契约执行。

TRIPOD 阶段使用真实 loco 教师和原有阶段/gait/blend。请求及实际下发 dz 达到 `0.35 - margin`，FR 跟踪误差 <0.10 m，连续 0.10 s 后切入 TRANSITION。调用现有 `start_from_current_state`，仅对齐参考水平位置/yaw；保留真实 root/joint 状态、速度和动作/执行器历史，不增加 reset 宽限期，不修改参考轨迹。记录入口误差以揭示 0.33 m 触发与固定参考首帧之间的不匹配。

参考完成后，姿态 <0.25 rad、高度 <0.05 m、线速度 <0.3 m/s、角速度 <0.8 rad/s、后脚均有 >5 N 支撑、前脚离地且足高 >0.032 m、关节速度范数 <20 rad/s，连续 1 s 才切入 BIPED。世界系足端 RMS <0.06 m 单独作为严格终点保持诊断；与原键盘语义一致，脚端跟踪误差不阻止站稳后的教师接管。界面和日志须明确区分两者。BIPED 继承过渡末帧的公共坐标 FR 目标，并恢复键盘操作。TRANSITION 10 s 超时暂停，不强制接管。失败、Backspace 和自动 reset 返回 TRIPOD。

提供无窗口 scripted 输入回放：真实三足策略起步，按相同 Q 键语义抬高，走相同触发/动作/状态机代码，保存事件、末态及失败信息；用于验证链路，不替代用户交互播放。CPU 为当前机器可用验证后端，CLI 保留 cuda:0。
