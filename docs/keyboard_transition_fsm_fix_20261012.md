# 键盘四足／双足切换修复记录

日期：2026-10-12。

启动命令保持不变：

```bash
python scripts/keyboard_transition_fsm.py --device cuda:0 --initial-state biped
```

## 起身完成后卡死的原因与修复

起身参考有 351 帧，下降参考有 186 帧。原实现完成起身后，在
`mark_rise_complete()` 内立即切换到下降参考，却保留起身末帧索引 350。
同一步的 FR 参考计算和后续观测读取因此越界。

实际加载四个 checkpoint 后，将命令管理器置于起身末帧的交接边界，
CPU 复现 `IndexError: index 350 is out of bounds ... size 186`；
本机 `cuda:0` 复现 `index out of bounds` 和 `device-side assert triggered`。

修复后，双足接管时继续保留可读的起身末帧；真正触发 DOWN 才选择下降参考。
选择另一份参考时同时重置参考帧索引、完成标记和对应分箱统计张量。
完成状态切换后重新计算当前控制者掩码，避免继续执行旧状态的参考更新。
环境复位先恢复下降参考，再恢复物理初始状态；局部复位保留其他环境正在
执行的参考与时钟。切换不会重置物理姿态、速度、动作历史或关节目标限幅器。

## FR 的统一零点与键盘行为

所有状态中的 `[dx, dy, dz]` 都是 FR 相对机身的向量在教师水平锚定坐标系中
的表示，减去默认四足姿态下的 FR 向量。切换姿态不会重新定义零点，也不会
重复叠加双足默认偏移。两个端点教师 checkpoint 的 `foot_zero` 与命令项
`zero` 一致，已加入回归断言。

- 四足默认命令为 `(0, 0, 0)`。首次按 Q 从四足默认目标开始，不再跳到双足高度。
- 双足默认命令约为 `(0.13349, 0.03060, 0.47804)` m，仍相对同一个四足零点。
- 双足状态点按 R 请求经下降策略返回四足，结束后恢复四足零偏移和默认支撑；
  四足状态 R 直接恢复默认目标。双足的默认姿态不显示为新的零点。
- 起身／下降过程中冻结键盘目标修改；端点接管时把参考末帧 FR 偏移同步到
  执行命令和键盘命令，下一次按键从该目标继续调整。R 请求的下降结束时，
  改为执行四足默认零偏移。

原有 union 工作空间的界限保持如下，单位为米：

| 区段 | dx | dy | dz |
|---|---|---|---|
| 四足操作区 | [-0.30, 0.30] | [-0.12, 0.12] | [0.05, 0.35] |
| 双足操作区 | [-0.05, 0.30] | [-0.10, 0.10] | [0.25, 0.72] |

它是两个三维区段的并集。四足 `(0, 0, 0)` 是默认支撑命令；非零操作目标
仍应用原有 5 cm 最小抬升界限。本次修正默认目标和状态交接，没有扩大训练目标范围。

## 验证结果

相关回归共 **39 项通过**，仅一条第三方 `torch.jit.script` 弃用警告：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/keyboard_full_chain_mpl \
python -m pytest \
  tests/test_keyboard_full_chain_handoff.py \
  tests/test_keyboard_full_chain.py \
  tests/test_keyboard_transition_t.py -q
```

禁用 pytest 自动插件加载是因为本机 ROS 的 `launch_testing` 插件依赖缺失的
`lark`；本次测试不需要该插件。

新增 16 个测试用例覆盖起身末帧可读性、参考切换时钟和分箱张量、四足默认目标、
过渡中输入冻结、起身和下降接管后的连续目标、起身后复位、局部复位隔离，
以及实际双足启动后的首次按键、R 和共同零点；另验证脚高跟踪不再终止
键盘播放，同时训练环境仍保留原判定。R 返回测试包含单次短按、松键后请求
保留、起身后返回的物理连续性、四足默认目标、暂停／失焦和复位清理。
支撑模式测试直接检查 54 维教师观测末位，并覆盖 FR 激活、保持中重设目标、
R 后支撑权重平滑恢复和最终四脚支撑。

GPU 验证使用实际 checkpoint、同一物理环境及 `FullKeyboardInputs` /
`FullChainPolicy`，无窗口运行：

1. 起身末帧交接定点复现脚本：修改前触发 CUDA 索引断言；修改后接管双足，
   继续运行 150 步／3 秒，未重置。
2. 从四足默认物理状态开始，静置 1 秒后持续按 Q，完成
   `QUADRUPED → RISE → BIPED`，接管后继续运行 3 秒；总仿真时间 12.36 秒，
   全程未发生物理／跟踪重置。起身切换前后 qpos、qvel、当前／前一动作、
   限幅器位置和速度的最大差值均为 0。

日志和结果保存在 `outputs/keyboard_transition_fsm_fix_20261012/`。

## 键盘观察模式：取消脚高跟踪终止

首次调查时，从双足默认状态开始，静置 1 秒后持续按 E，完整往返链在下降开始后发生
`ee_body_pos` 足端跟踪终止，自动复位到双足。修改版本约在仿真 2.08 秒触发；
从 HEAD 提取的未修改播放器约在 2.10 秒触发同一终止。

随后按用户要求，从全链键盘播放器的终止配置中移除 `ee_body_pos`，起身和
下降都不再因四只脚与参考的高度误差超过 15 cm 而复位。训练环境的脚高
跟踪终止保持原配置；该修改用于观察实际动作效果。

取消后在本机 `cuda:0` 重跑同一路径，完成
`BIPED → DOWN → QUADRUPED → RISE → BIPED`，总仿真时间 16.98 秒，
最后双足接管后继续运行 3 秒，全程无自动复位。两次过渡启动前后 qpos、
qvel、动作历史和限幅器状态的最大差值均为 0。新日志和结果：
`outputs/keyboard_transition_fsm_fix_20261012/no_foot_tracking_cycle_cuda.log`、
`no_foot_tracking_cycle_result.json`、`no_foot_tracking_tests.log`。

## 当前全部重置条件与入口

全链键盘播放器当前安装 3 个自动终止项，任一为真，框架在同一步自动复位
对应环境：

| 终止项 | 判定条件 | 生效范围 |
|---|---|---|
| `anchor_pos` | 实际机身锚点世界 Z 与参考 Z 的绝对差 > 0.15 m | 仅 RISE／DOWN |
| `anchor_ori` | 实际机身姿态与参考姿态的旋转角误差 > 0.8 rad，约 45.8° | 仅 RISE／DOWN |
| `physical_failure`：非有限数 | 关节位置／速度，或跟踪机体世界位置／姿态／线速度／角速度出现 NaN／Inf | 所有状态 |
| `physical_failure`：无效姿态 | 实际机身锚点四元数范数 < 1e-6 | 所有状态 |
| `physical_failure`：机身／头部触地 | `base_ground_contact` 中任一机身／头部接触力向量模长 > 10 N | 所有状态 |

前两项有物理复位后 0.2 秒宽限期，进入过渡不会重新计时。物理失败立即
生效。第三项传感器检查的部位为 `base_link`、`Head_upper`、`Head_lower`。

手动和初始化入口：

- Backspace：播放器显式处理，完整重置环境。
- Enter：继承的 NativeMujocoViewer 快捷键，同样完整重置环境。
- 启动／加载策略时：包装器和播放器调用 `reset()`，初始化所选状态。
- 命令定时重采样：motion 和 twist 均配置为 1e9 秒，正常键盘观察中不会到期；
  motion 的重采样函数本身会恢复初始物理状态。

以上环境复位均恢复 `--initial-state` 指定状态：`biped` 恢复固定 entry-bank
样本 16508 的物理状态及动作历史，`quadruped` 恢复默认四足状态。

本播放器没有安装 `time_out`，所以继承的 5.5 秒 episode 参数不触发定时复位。
也没有安装训练用 `landing_safety`：膝部间隙、其他腿壳触地和 FR 触地监测
不会自行触发复位。脚高跟踪 `ee_body_pos` 已取消。
R 在双足状态请求下降并返回四足，在四足状态恢复默认 FR 命令；它不调用
环境复位。Space、失焦、松键，以及参考轨迹正常结束也不触发环境复位。

## R 返回四足的语义修正

先前双足状态按 R 会把目标恢复到双足默认 dz≈0.478 m，高于下降阈值
0.25 m，因此不会进入 DOWN。实际 GPU 复现中，点按 R 后超过 2 秒仍保持
BIPED。这个“恢复当前姿态默认目标”的语义与用户要求的“返回四足”不一致。

现在双足 R 使用单独的返回请求：先保持当前 FR 目标并停止速度命令，在
下一次仿真命令更新时启动下降策略，不需要先把双足教师的目标突变为零，
也不依赖按住 R 或降低到高度阈值。请求会一直保留到下降结束；四足教师
接管时恢复 `(vx, vy, wz, dx, dy, dz)=(0,0,0,0,0,0)`，退出 FR 操作目标。
普通 E 降低目标触发下降的路径仍保留末帧目标，过渡中继续冻结手动修改。

实际 GPU 验证日志和结果保存在同一输出目录，文件名前缀为
`r_return_`。直接双足启动、点按 R、返回四足并继续运行 3 秒的测试通过，
总仿真时间 7.74 秒；启动下降前后的物理状态和动作历史最大差值均为 0。
另一路从四足开始，完成起身、双足保持、点按 R 下降、四足保持 3 秒，
总仿真时间 17.10 秒，状态序列为 QUADRUPED → RISE → BIPED → DOWN →
QUADRUPED，最终六维命令全零，两次过渡启动时的状态和动作历史均连续。
记录分别为 `r_return_biped_result.json` 和
`r_return_after_rise_result.json`，两条路径均未发生环境复位。

## 四足教师内部四脚／三脚支撑修复

外层 `QUADRUPED` 表示由四足教师控制，并不表示内部已经四脚支撑。
此前完整播放器每帧无条件写入 `phase=HOLD`、`blend=1`，覆盖源教师的
四脚／三脚状态机。54 维四足教师观测最后一维就是 `blend`，因此即使 R
已经把 FR 命令清零，教师仍收到三脚支撑信号。先前仅检查外层 owner 和
六维命令的验证不能证明已经恢复四脚支撑。

修复后，提交键盘目标时调用源教师的 `set_command()`，保留激活标志和
目标变化记录；更新命令时运行源教师原有状态机，仅对 BIPED／RISE／DOWN
覆盖命令及 HOLD／blend=1。QUADRUPED 下 FR 操作沿 PREPARE／REACH／HOLD
进入三脚支撑，R 清零目标后使用原有 unified return 平滑恢复支撑、确认 FR
接触，最终进入 `phase=QUAD`、`blend=0`。此过程不调用环境复位，公共 FR
零点不变，外层 owner 在四脚／三脚切换时都保持 QUADRUPED。

同一实际 GPU 探针运行四足静置、抬 FR、保持、点按 R、继续静置，共 8.40 秒：

| 项目 | 修复前 | 修复后 |
|---|---|---|
| 最终教师 phase / blend | HOLD / 1 | QUAD / 0 |
| 最终 FR site 世界高度 | 0.1130 m | 0.0213 m，足底接触地面 |
| 最后 0.5 秒 FR 接触率 | 0% | 100% |
| 最终四足接触 | true, false, true, true | true, true, true, true |

修改后全程没有环境复位，R 输入前后物理状态、动作历史、限幅器最大差值
均为零。记录：`support_before_cuda.log`、`support_before_result.json`、
`support_after_cuda.log`、`support_after_result.json`；测试日志为
`support_tests.log`，全部位于前述输出目录。

恢复支撑状态机后另用实际 GPU 重跑两条过渡链：直接双足启动点按 R 返回
四足（7.74 秒），以及四足起身到双足再点按 R 返回（17.10 秒），均通过。
两条路径返回后都继续运行 3 秒，最终 `phase=QUAD`、`blend=0`、FR 接触
为真，未发生环境复位。结果为 `support_biped_return_result.json` 和
`support_full_cycle_result.json`。

## 其他既有测试限制

扩展检查 `tests/test_keyboard_teacher.py` 时，3 项旧测试仍以矩形边界／目标库
行为断言当前四足几何缓存，单独运行该文件也得到同样的 3 项失败。
本次未改动该播放器、工作空间缓存或这些旧测试。

## 14:52 启动失败：系统 CUDA 驱动需要主机重启

用户报告创建 Scene / terrain 时，第一次 torch.zeros(device='cuda:0')
触发 CUDA unknown error。宿主机独立复现：

- 不导入仓库代码，仅 import torch 后创建 CUDA 张量，同样失败。
- ctypes 直接调用系统 libcuda.so.1 的 cuInit(0)，返回 999 / CUDA_ERROR_UNKNOWN。
- CUDA_VISIBLE_DEVICES 未设置；在独立进程移除 LD_LIBRARY_PATH 后仍返回 999。
- nvidia-smi 能识别 RTX 4060 和 580.178.04 驱动，但 CUDA 初始化失败。
- 内核模块和实际加载的 libcuda 版本均为 580.178.04，UVM 设备节点存在。

关键内核日志（本机时间）：

```text
Oct 12 14:11:54 PM: suspend entry (s2idle)
Oct 12 14:12:44 PM: suspend exit
Oct 12 14:14:16 NVRM: Xid 31, name=modprobe, MMU Fault
Oct 12 14:14:16 NVRM: Xid 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)
```

因此当前阻塞来自系统 GPU 驱动状态。睡眠恢复与故障在时间上相邻，具体触发原因
尚未确定；不能仅据此认定是某一段 Python 代码或睡眠机制导致。
NVIDIA 文档要求遵循 Xid 154 报告的恢复动作，本机明确为主机重启：
https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html
https://docs.nvidia.com/deploy/nvidia-smi/index.html

本次保存诊断，没有修改播放器代码或驱动配置，没有重启主机或终止桌面进程。
内核原始记录为 outputs/keyboard_transition_fsm_fix_20261012/cuda_driver_failure_kernel.log。
需要用户保存工作并重启操作系统，然后先验证独立 CUDA 张量分配，再运行原命令。
此时 CUDA 尚未恢复，尚未重新验证 GUI 启动。
