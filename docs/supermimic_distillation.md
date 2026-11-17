# Go2 四教师世界模型蒸馏 v2

学生、世界模型和价值网络全部随机初始化；冻结教师使用mark/{quad,biped,up,down}_teacher.pt。
沿用48D观测、GRU256、12D动作、50Hz直接PD。入口legmanip_distil_supermimic。
完整检查点版本2，拒绝旧版续训；既有独立student.pt仍能播放。

## 训练入口

WSL激活cjisoad_mjlab，服务器激活go2_supermimic，然后在仓库执行：
~~~bash
bash scripts/train_supermimic_distil.sh --agent.run-name fresh_coverage_v2
~~~
默认256环境、32控制步、60000次更新。每轮4次联合学生更新，由
--agent.supermimic.joint-updates控制；父类旧epoch/minibatch选项不控制新版联合更新。
世界模型512x3，每轮4次模型更新、batch128、连续4步。前1000次更新教师控制，
随后episode级DAgger逐步降低到10%教师保底。不加resume即从头训练。

## 覆盖采样

目标实际帧比例：四足走20%、双足走20%、三足操作20%、双足操作15%、
行走操作组合15%、完整切换10%。采样器根据已实现帧缺额修正新episode分配。
目标比例不代表短训练能精确达到。任务配额包含返回等收尾阶段，另记录stage/owner实际阶段比例，避免把收尾帧当成纯行走覆盖。

直接任务10-20秒，速度每1-4秒变化，足端保持2-5秒后连续换目标。
前10000次更新扩大速度与足端区域，保留20%容易样本。纯行走目标vx[-.2,.6]m/s、
wz[-1,1]rad/s、vy=0，包括停止、反向和原地转弯。
三足组合遵循原教师活动模式速度限幅：|vx|≤.35m/s、|wz|≤.4rad/s；目标包络不是能力保证。
双足使用已有有效状态库，保留根、关节、动作和限幅器上下文。
足端按XY内外区域和高度分层，路径经过检查的hub，有限采样运动学/碰撞验证不是动态可达性证明。
采样抬脚、收脚、键盘六零返回与完整起降。explicit legacy场景与手动播放保持原行为。

## 奖励与联合更新

速度/足端使用指数精细奖励加负Huber误差项，远误差仍有纠正信号。
线速度尺度四足.10m/s、双足.05m/s，转向尺度.30/.20rad/s，足端尺度.05m；
均为优化尺度，不是验收门槛。足端只在操作时启用，起降时关闭。
姿态连续跟随参考，支撑/滑动/步态区分四足、三足、双足。
奖励分速度、足端、姿态、正则化四组。

教师动作监督统一为12维平均L1，按实际task帧均衡。
一次学生step合并真实监督、想象教师/奖励和行走对称梯度。
世界模型和价值参数冻结，保留状态导数。镜像GRU独立维护历史，
只在物理reset清空；含操作历史的episode不施加行走对称。

真实仿真也计算同一奖励，用于价值网络校准。实际误差在supermimic/real/*。
通用Train_mean_reward仍可能为0，不代表该方法的真实任务回报。

## 可靠性与梯度检测

回放按task和接触模式分层。保留终止帧，屏蔽跨复位/加载边界和非有限数据。
预留独立仿真世界，不进入模型/价值拟合，用于多步验证。
价值网络使用真实奖励lambda目标；无效未来切断递推并保留当前有效bootstrap。
加载边界不修改物理done。

每100更新验证，8步候选终点必须有有效样本；每类任务连续3次满足内部误差条件后
才由4步扩为8步，证据不足保持4步。奖励梯度权重随模型可靠性下降，保留教师监督。
想象遵守已知时间上限和场景结束。

每100更新检查任务梯度，每500更新检查目标/奖励组梯度。
记录余弦、加权范数、累计冲突率，分all/gru/shared/head。
零范数不计算余弦，检测不写参数梯度，不自动投影梯度。

## 短训练与恢复

提前启用想象和诊断，仅检查运行链路：
~~~bash
bash scripts/train_supermimic_distil.sh --env.scene.num-envs 32 --agent.max-iterations 24 --agent.save-interval 12 --agent.supermimic.joint-updates 1 --agent.supermimic.model-updates 1 --agent.supermimic.model-batch-size 32 --agent.supermimic.imagination-start-update 0 --agent.supermimic.gradient-interval 4 --agent.supermimic.detailed-gradient-interval 8 --agent.supermimic.validation-interval 4 --agent.supermimic.validation-batch-size 16 --agent.run-name coverage_v2_startup
~~~
保存model_N.pt和独立student.pt。完整模型含优化器、回放、随机数、采样计数和检测历史。
~~~bash
bash scripts/train_supermimic_distil.sh --agent.resume True --agent.load-run '<run_directory>' --agent.load-checkpoint model_500.pt
~~~
缩小过的试运行恢复时需要相同参数。student.pt由scripts/play_legmanip_distil.py加载，
不加--scenario为手动控制。只有新一轮实际训练和测试才能判断策略效果。

指标：
- supermimic/coverage/*：实际任务帧、命令类别、足端区域、失败阶段。
- supermimic/real/task_N/*：真实跟踪误差与奖励组。
- supermimic/model_validation/*：独立多步误差、有效终点样本数。
- supermimic/grad_tasks/*、grad_objectives/*、grad_rewards/*：梯度证据。
- supermimic/imagination_confidence、active_horizon：可靠性控制。

## ClearML 云端

在服务器激活 go2_supermimic 环境后，安装 requirements-clearml.txt。
首次登录在 ClearML 网页 Settings → Workspace 创建凭据，然后在服务器执行：

```bash
clearml-init --file /home/ubun/clearml-go2.conf
chmod 600 /home/ubun/clearml-go2.conf
export CLEARML_CONFIG_FILE=/home/ubun/clearml-go2.conf
```

将网页提供的连接配置粘贴到初始化向导，等待凭据验证成功。
每次新终端启动训练前设置同一个 CLEARML_CONFIG_FILE。
配置文件独立保存在用户目录，训练参数与仓库只保存项目名等普通设置。

```bash
bash scripts/train_supermimic_distil.sh \
  --agent.clearml.enabled True \
  --agent.clearml.project Go2/SuperMimic \
  --agent.clearml.task-name fresh_supermimic \
  --agent.clearml.checkpoint-interval 1000 \
  --agent.run-name fresh_supermimic
```

曲线复用 TensorBoard；控制台输出、训练配方和软件环境也会记录。
完整检查点与部署学生模型以 checkpoint_N、student_N 两个独立产物上传到云端
文件服务。默认每 1000 次更新上传一次，前提是该更新已有本地检查点
（本地保存默认每 500 次）；正常结束额外上传最终检查点，按更新编号去重。
上传等待文件传输完成，保证被反复覆盖的 student.pt 与该次完整检查点对应。
上传失败会报告警告并保留本地文件，后续上传节点和训练结束时重试最新状态。
恢复训练会创建新的 ClearML 任务，指标沿用恢复后的更新编号。
可通过 --agent.clearml.output-uri 显式指定其他存储目的地。

配置凭据前可增加 --agent.clearml.offline True 做离线验证。
离线会话的保存路径会打印到控制台，里面包含指标和所选模型文件；
正式云端训练使用默认 offline=False。

## 冻结教师能力探测
~~~bash
PYTHONPATH=. python scripts/probe_supermimic_teachers.py --steps 500 --num-envs 2 --device cuda:0 --output outputs/teacher_quality.json
~~~
检查四足/双足速度网格24种指令，每种两个世界、10秒，报告速度误差、机身碰撞失败和有效测量时长。
无性能通过门槛；不替代足端精度、姿态和长期稳定性测试。
