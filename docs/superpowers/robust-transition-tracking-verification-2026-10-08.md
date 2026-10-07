# 鲁棒过渡跟踪 v3：实施与本地验证

日期：2026-10-08。任务：`pedipulation_bridge_t`。

## 已实现的三个课程

| 阶段 | 合并后的内容 |
|---|---|
| 0 | 原0/1：名义参考起立、FR轨迹多样性；关闭额外扰动、DR、噪声和延迟 |
| 1 | 原2/3：时间倍率、初始动量、实际站稳后的恢复扰动 |
| 2 | 原4/5：冻结三足教师实际入口、起立中扰动、DR、噪声和延迟；入口占比与新增幅度共同递增 |

恢复旧任务的参考状态reset和参考观测方式，使用公共教师机器人模型。奖励由姿态、高度、FR、支撑关节、FL高度跟踪以及必要物理约束组成；没有线速度或角速度跟踪奖励。训练以起立跟踪和持续站稳为目标，实际冻结双足接管作为独立评估。

过渡actor84、critic172从头初始化，冻结教师维持actor51。PPO初始std0.5、LR3e-4、entropy0.005、gamma0.995、lambda0.95。Checkpoint契约升级v3，拒绝直接恢复v2模型。

## 最终本地验证

环境：`/home/eden/miniconda3/envs/mjlab/bin/python`。最后一轮测试在参考缓存修复及评估hook异常恢复修复后执行。

| 验证 | 结果 | 记录 |
|---|---|---|
| 全部`test_bridge*.py` | 86项通过 | `/tmp/bridge_v3_review_tests.log` |
| 教师接口与旧过渡任务回归 | 59项通过 | `/tmp/bridge_v3_review_regression.log` |
| CPU，4环境，课程0，一次PPO | actor更新、冻结教师不变、checkpoint恢复和拒绝不兼容配置均通过 | `outputs/bridge_tracking_v3_cpu_review/verification.json` |
| GPU，32环境，课程0，一次PPO | 同上；1740个有效过渡样本 | `outputs/bridge_tracking_v3_gpu_review/verification.json` |
| GPU，16环境，课程2真实入口，一次PPO | 16/16源教师产出入口；1848个有效过渡样本；同上各检查通过 | `outputs/bridge_tracking_v3_live_review/verification.json` |
| 确定性名义跟踪评估，2世界各1次 | 配额完整、保存入口和reset前终态、输出分组结果 | `outputs/bridge_tracking_v3_evaluation/nominal_review.json` |

上述PPO检查均只训练一次更新，新策略起立成功数为0。它们验证代码、物理接口、样本归属和优化流程可运行，不证明策略已经学会起立或达到鲁棒性验收目标。正式训练后需按计划进行完整起点、扰动、真实入口和实际双足接管评估。

新增回归测试先重现两个问题，再验证修复：评估抛异常时必须恢复临时advance hook；实际inference模式下临时quaternion不能让参考缓存失效。世界参考按控制步/持久参考状态缓存，参考速度每次转换到当前实际anchor。

质量审查和规格审查均通过。另修复一处教师测试fixture缺少actor/critic维度的已有问题，教师生产代码没有因此改变。`git diff --check`通过。GUI播放入口已提供；本次验证使用headless MuJoCo，未以GUI播放结果作为通过证据。

## 正式训练部署

用户已授权将本次改动提交并推送至`origin/wsl-validation`，同步到`root@xj-member.bitahub.com:42288`，再从课程0启动新训练。使用独立部署目录和两个原冻结教师checkpoint，不恢复旧v2训练。

服务器部署结果另存`outputs/bridge_tracking_v3_server_start.json`，包含实际提交、目录、PID、日志位置和启动后观测到的训练进度。每100轮保存checkpoint，每500轮评估是后续训练检查安排；本地短验证不能替代这些结果。
