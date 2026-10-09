# 鲁棒参考起立跟踪（bridge v3）

`pedipulation_bridge_t` 的主要任务是跟踪参考起立并站稳。训练从参考状态学习完整动作，再逐步加入真实三足教师交出的状态差异、初速度和随机化。右前脚保留宽松跟踪；没有导航线速度或角速度命令跟踪奖励。

## 三个课程阶段

| 阶段 | 内容 |
|---|---|
| 0 | 标准参考起立 + 25条FR轨迹；名义物理、无额外动量、噪声或延迟 |
| 1 | 阶段0 + 时间倍率0.85–1.15、初速度、真正站稳后的恢复扰动 |
| 2 | 真实三足入口 + 起立中扰动、物理随机化、观测噪声、控制延迟 |

阶段0/1的reset为50%完整三足参考起点、30%起立中间帧、20%参考终点，同时写入匹配的机身/关节状态和速度。FR关节、速度和目标使用同一参考bank。至少35%采样使用bank0；扰动阶段至少35%世界不施加额外扰动。

阶段2中真实入口占比与新增随机化幅度按25%→50%→75%→100%一起增加，这是第三阶段内部的幅度参数。晋级以至少128个合格完整起点/真实入口、最多512结果的窗口、成功率85%及至少1500控制步为初始门槛。参考终点重置成功不能抬高晋级成功率。课程及其年龄随checkpoint保存恢复。

真实入口从冻结三足教师运行到FR命令dz=0.25–0.28m时接管；保留速度、姿态、关节、上一动作、延迟队列和物理参数。0.3s混合只改变参考目标；同一episode中不把机器人写成参考姿态。参考起点写入只发生在训练reset。

物理参数从公共模型名义字段恢复后逐次采样，避免随机化累积。阶段2逐步增加观测噪声、延迟上限和起立中扰动。公共PD增益和armature来自两个教师共用的机器人配置。

## 引导数据

默认使用 `src/assets/motions/go2/tripod_to_biped_nominal.npz`：50Hz、301帧，起立约2.5s完成，之后保持终点。使用机身高度/完整姿态、九个支撑关节位置和速度、参考机身速度、FL高度及接触卸载阶段。参考速度是观测提示，没有对应速度跟踪奖励。真实入口FR路径使用已审计 `bridge_target_paths.npz`；参考支撑姿态仍来自名义起立资产，FR排除在关节模仿之外。

参考是静态/运动学引导，参考关节姿态不直接充当PD动作示范。策略学习实际控制目标。旧 `pedipulation_transition_t` 和旧参考、checkpoint保留。

## 观测和PPO

过渡actor84：公共51维 + 实际高度1、FR误差3、进度1、参考姿态误差rotation vector3、参考高度1、九支撑关节参考位置9/速度9、参考线速度3/角速度3。critic172共享全部actor噪声，并增加88维精确物理/接触/参考特权信息。两个冻结专家一直使用原actor51，不参与优化。

过渡actor和critic从头初始化，网络512/256/128 ELU。初始动作std=0.5、学习率3e-4、entropy=0.005、gamma=0.995、lambda=0.95。公共动作scale=0.25，过渡动作clip±10。日志输出12个关节std及实际PD目标std，准备/起立/终点三段奖励积分，完整/中间/终点/真实入口分组和初速度分组。

## 奖励

逐步项乘权重和控制周期dt=0.02s，只收集过渡策略控制的步骤。

| 项 | 权重 | 定义 |
|---|---:|---|
| orientation | 3.0 | 完整参考姿态角误差，尺度12°指数核；参考航向对齐入口 |
| height | 3.0 | 高度误差，尺度0.05m指数核 |
| fr_tracking | 1.0 | 位置误差平方均值，尺度0.08m指数核，全程同尺度 |
| support_pose | 0.5 | 九支撑关节误差平方均值，尺度0.30rad指数核 |
| fl_height | 0.5 | FL高度误差，尺度0.05m指数核 |
| support_loss | −0.7 | 按阶段的支撑损失 |
| front_contact | −0.5 | FL按卸载阶段加权，FR接地惩罚 |
| slip | −0.15 | 接触脚水平速度平方和 |
| excessive_force | −0.1 | 接触力超过160N后的归一化平方 |
| collision | −1.0 | 非足部触地/自碰撞，包含FR |
| action_change | −0.03 | mean(((a−a_prev)×scale/0.05)²) |
| torque | −0.02 | 按执行器力矩上限归一化平方均值 |
| joint_limits | −0.5 | 软限位越界距离 |

失败反馈为一次−5，直接加到最后一个过渡步；不再次乘dt。成功没有额外终端奖金。准备/终点奖励积分不包含单独的失败事件。删除v2耦合高度/重力的rise_progress及双足结果±20。

## 成功与真实交接

训练默认不运行双足VERIFY。过渡episode固定10s，参考完成后姿态误差<15°、高度误差<0.08m、FR误差<0.10m、前脚离地、后脚近0.3s都有接触且当前至少一脚支撑、线速度<0.5m/s、角速度<2rad/s，连续保持1s。终点基座还需>0.44m、gravity.x<−0.8、|gravity.z|<0.35、关节速度范数<20rad/s。episode结束时必须仍满足保持条件；曾经站稳后倒地不能算成功。扰动清除站稳计时，要求重新保持。报告首次站稳时间和5s内站稳次数。

基座倒地、严重非足部/自碰撞、严重关节越界或非有限状态会失败。健康的参考误差不会单独立即终止。超时未站稳为失败。源阶段失败记为源产出问题，源教师控制步不进入PPO。

独立 `live_handoff` 评估从实际站稳状态切换冻结双足教师，保留物理和执行器历史，验证2.5s。评估固定每世界配额；live条件采用偶数世界精确分配50%静止/50%移动。保存入口、真实交接和reset前终态的物理状态、参数及动作历史；报告起立后接管失败。静态终点评估仍由 `evaluate_bridge_biped.py` 单独标记。

## 使用

使用仓库mjlab环境。下面训练命令需要用户另行启动正式训练；实现检查只执行有界smoke。

```bash
PYTHONPATH=.:tests MUJOCO_GL=disable python -m unittest discover -s tests -p 'test_bridge*.py' -q
PYTHONPATH=. MUJOCO_GL=disable python scripts/smoke_bridge.py --device cpu --num-envs 4 --cycles 1 --ppo --output outputs/bridge_tracking_v3_cpu
PYTHONPATH=. MUJOCO_GL=disable python scripts/smoke_bridge.py --device cuda:0 --num-envs 16 --cycles 1 --ppo --stage 2 --live --output outputs/bridge_tracking_v3_live
PYTHONPATH=. python scripts/train.py pedipulation_bridge_t --env.scene.num-envs 1024 --agent.max-iterations 15000 --agent.save-interval 100 --agent.run-name robust_tracking_v3
PYTHONPATH=. python scripts/evaluate_bridge_tracking.py --checkpoint /path/to/v3/model.pt --case nominal --num-envs 32 --episodes-per-world 4
PYTHONPATH=. python scripts/evaluate_bridge_tracking.py --checkpoint /path/to/v3/model.pt --case live_handoff --num-envs 32 --episodes-per-world 4
PYTHONPATH=. python scripts/evaluate_bridge_tracking.py --checkpoint /path/to/v3/model.pt --case reference_disturbed --num-envs 1 --viewer native
```

评估case有 `nominal`、`fr_diversity`、`reference_disturbed`、`live_source`、`live_handoff`。native使用相同动力学控制流程和FR参考标记，非运动学qpos播放。

## 键盘三教师状态机

`scripts/keyboard_teacher.py loco_pedipulation_t --workspace union` 支持单环境
“三足 → 过渡 → 双足”。默认触发高度为FR dz0.33m；实际FR误差小于0.10m并保持0.10s后接管。
过渡阶段参考完成且站稳条件连续满足1s才切换双足，随后恢复键盘操作。
键盘接管保留姿态、高度、支撑及速度等站稳门槛；FR误差仅显示诊断值，不阻止双足接管。
三足进入过渡时的FR误差门槛仍为0.10m；训练与批量评估仍使用原完整成功判据。
切换保持同一物理状态、速度、动作历史和参数；过渡超时10s暂停并显示未满足条件。
Backspace完整复位到三足，R在稳定控制阶段恢复当前教师默认目标。

```bash
MUJOCO_GL=glfw PYTHONPATH=. python scripts/keyboard_teacher.py loco_pedipulation_t \
  --workspace union --num-envs 1 --device cuda:0 \
  --checkpoint-file outputs/new_teachers_review_20261006/quadruped/checkpoints/model_9900.pt \
  --bridge-checkpoint-file outputs/bridge_tracking_v3_recovered_20261008/model_900.pt \
  --biped-checkpoint-file outputs/new_teachers_review_20261006/biped/checkpoints/model_14600.pt
```

命令支持兼容的actor51或54三足教师；桥接actor84与双足actor51分别加载，原桥接训练契约仍严格校验。
`--transition-margin 0.02`表示距三足0.35m上限提前2cm触发。
当前model900尚未达到训练成功判据，而且0.33m高于原真实入口配置0.25–0.28m，需结合播放观察实际过渡效果。

Checkpoint契约版本3记录84/172观测、教师/参考/路径哈希、公共物理和执行器、奖励、课程控制及终端语义。旧v2模型（包括model330）以及旧过渡模型不能直接恢复训练，应开新run。

计划：`docs/superpowers/plans/2026-10-08-robust-transition-tracking.md`。
