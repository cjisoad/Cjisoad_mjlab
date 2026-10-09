# transition_t 真实入口课程交付记录

本报告记录 FR 限速加入前的历史基线。最新 schema 2 状态库、目标限速及验证
见 `docs/transition_target_limiter_report_20261009.md`；下方旧部署包不适用于限速版。

## 数据与契约

- 三足冻结教师：`outputs/loco_return_20261009/checkpoints/model_14999.pt`
- 过渡 warm start：`logs/rsl_rl/transition_t/2026-10-09_01-13-48_transition_t_direct_4090_20261009/model_30000.pt`
- 参考 SHA256：`1cb970ae8cfb107e25f3aca2e1d48b8cf9a2a145d6fac25c9aa67076c3bdd0b6`
- 平衡状态库：`outputs/transition_entry_course_20261009/entries_balanced.npz`
- 状态库 SHA256：`07e1f7a76adbdf5e6c69afba5793de539765a7bf82e3f673545d3c78b053c94f`
- 采样状态：12,669；不同轨迹：748；训练轨迹：601；留出轨迹：147

采集状态来自实际三足教师在共同过渡物理环境中的运行，记录 root 世界速度、FR 实际误差、关节状态和三帧公共动作历史。入口恢复后参考从首帧开始，参考 XY/yaw 与当前 root 对齐，接触由物理仿真重算；没有为采样入口增加 tracking grace。

## 本地证据

`regression_final_v4.log`：聚焦 transition/keyboard/entry 测试结果为 `102 passed, 4 subtests passed`。另外，`regression_full_v1.log` 包含当前工作区已有的 3 个 `tests/test_keyboard_teacher.py` 断言失败，原因是 commit `0311aa8` 将矩形目标区域改为几何缓存；未修改该用户变更。

继续后的验证记录在 `regression_continuation.log`：包含新增采集/评估/训练入口、键盘交接、课程与原 transition 测试的 13 个测试文件，结果 `121 passed, 4 subtests passed`，退出码 0。此轮没有运行上述存在旧断言失败的 `test_keyboard_teacher.py`；不能据此声明全仓测试通过。

`ppo_verification.json`：在 CPU、8 环境下从 model 30000 warm start 做两次 PPO 更新，actor 参数发生变化且保持有限；随后从 checkpoint 恢复并完成第 3 次更新，优化器步数 40→60、环境步数 48→72，自适应学习率恢复前后均为 `1e-5`，TensorBoard 标量有限。

审查修复了评估轨迹重复造成晋级证据不足、断点恢复遗漏自适应学习率、PPO 缓存的非叶张量导致评估 deepcopy 失败，以及训练加载报错时的清理逻辑。恢复目标更新数不得小于等于 checkpoint 已完成数；不会报告不存在的新 checkpoint。键盘动作经理现保存端点教师实际执行的公共动作，确保过渡 first observation 的 last_action 与采集恢复坐标一致。

状态采集使用名义物理；训练保留过渡任务已有的 base COM、编码器 bias、足端摩擦及 actor 观测噪声增强。Checkpoint recipe 显式记录所有事件、参数范围、噪声函数签名与名义 reset 扰动，恢复时比对完整 recipe。模型是在名义采样状态上接受原有启动增强的训练设计，采集和训练随机化并不完全相同。

`baseline_model30000.json`：原模型每档 64 次留出评估，独立轨迹数为 level 0 的 48、level 1/2/3 的 64。完整起立/站稳分别为 level 0 `53/64, 33/64`，level 1 `41/64, 25/64`，level 2 `24/64, 9/64`，level 3 `15/64, 6/64`；严格终点保持四档均为 `0/64`。level 0 的独立轨迹不足 64，因此不会触发晋级。

评估只执行过渡策略并测量站稳交接资格，不执行双足教师链；strict endpoint 是独立诊断。以上结果是续训前基线，不是收益声明。

## 部署包核验

- 文件：`outputs/transition_entry_course_20261009/transition_entry_deployment_20261009.tar.gz`
- 大小：127,325,276 bytes（约 121.43 MiB）
- SHA256：`cd3f7a8c70573095e4d99fdb0e315d3e035404d75cdfc413ce083b4e89115356`
- 清单：522 个内容文件，另含 `deployment_manifest.json` 和 `deployment_files.sha256`，总计 524 个归档文件。

从部署目录运行归档的 `sha256sum --check` 返回成功。归档解压到独立目录 `/tmp/transition_entry_deployment_verify.efKohp` 后，包内清单的 522 个文件全部通过 SHA256 校验；没有缺失文件或内容哈希不匹配。包内 `entry_rl.py` 已核实包含 version 3 recipe、完整训练事件和观测噪声配置；`remote_run.sh` 通过 `bash -n`。本节为打包后的本地补记，包内报告保留打包时的快照，不在归档内记录归档自身的哈希。

## 远端启动

原端口 `xj-member.bitahub.com:42125` 在服务器重启期间拒绝 SSH。拿到新的 SSH 命令后，在新目录完成 GPU/磁盘/依赖预检，重新采集正式状态库，再按 README 命令启动有界续训并记录 PID、日志、checkpoint 和课程事件。正式训练的完整评估必须使用同一状态库和参考/物理哈希。

预备脚本固定使用 256 环境、16 批次采集；每个训练/留出难度至少需要 64 条独立轨迹，否则终止而不启动正式训练。通过 GPU、依赖版本和至少 5 GiB 磁盘预检后，用 4096 环境续训 5000 次更新，每 250 次更新运行留出评估；至少 80% 的连续站稳率并连续两个窗口达标后才晋级。训练结束后再独立评估 `model_4999.pt`。当前没有远端续训 PID、checkpoint 或性能提升结果；等待用户提供服务器重启后的 SSH 命令，不重试旧端口。
