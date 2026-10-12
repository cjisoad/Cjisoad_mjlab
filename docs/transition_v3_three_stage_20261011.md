# V3 三课程从零训练与 ClearML

代码已准备；正式训练未启动。服务器配置与 GPU 冒烟等用户开机后执行。旧参考训练此前已停止。

## 初始化与参考

新 actor、critic、归一化器和 Adam 从零开始，`--from-scratch` 不允许与 checkpoint、resume 或参考迁移混用。旧 `model_30000` 和续训模型不参与初始化。三足 `model_14999` 只提供真实入口状态库。

仓库默认使用 V3-1.25x 起立参考：161 帧、50 Hz、3.20 s；原始 173 帧、3.44 s 参考仍保留在 `src/tasks/transition_t/data/tripod_to_biped_v3_train.npz`。评估在参考完成后单独计保持时间。匹配的 entry bank 为 `src/tasks/transition_t/data/entries_v3_speed125_train.npz`，其 metadata 与默认参考 SHA256 一致。

## 三课程

比例顺序为参考首帧 / 参考片段 / 真实教师入口。

| 课程 | 比例 | 真实入口水平速度 / FR 误差上限 | 随机化 | 晋级或验收 |
|---|---|---|---|---|
| 0 动作学习 | 50 / 50 / 0% | 不采真实入口 | 名义物理、小初始化扰动 | 首帧完整起立至少52/64，连续两次 |
| 1 真实交接 | 40 / 20 / 40% | 0.45 m/s / 0.14 m | 原配置50% | 独立留出入口至少48/64，并且名义首帧至少52/64，连续两次 |
| 2 鲁棒起立 | 20 / 20 / 60% | 1.20 m/s / 0.45 m | 原配置100% | 独立留出入口至少103/128，名义首帧至少52/64 |

名义评估采用课程0的小初始化扰动，关闭物理随机化和观测噪声。入口评估使用对应课程的真实物理随机化，关闭 actor 观测噪声，确定性执行策略。入口按速度/FR误差分桶均衡抽样，同一评估窗每条轨迹只采一个状态。片段初始化不计完整起立成功。

课程0根 XY±5 mm、Z±2 mm、roll/pitch±0.02 rad、yaw±0.03 rad；线速度XY±0.03 m/s、Z±0.02 m/s、角速度±0.05 rad/s、关节位置±0.01 rad。课程1 COM±5 mm、摩擦0.8–1.1、encoder偏差±0.005 rad，其他原reset/观测扰动减半；课程2 COM±1 cm、摩擦0.6–1.2、encoder±0.01 rad及原始reset/观测扰动。真实入口恢复实际速度和完整执行历史，不另叠合成速度。

每250次 PPO 更新评估，不按固定训练次数强制晋级。升级时保存旧阶段边界，重建新环境使 startup 物理事件实际生效，恢复网络、归一化器、Adam、adaptive学习率、相位失败统计和累计更新计数，然后保存可直接 resume 的新阶段模型。边界测试里的人工课程变化只是重建检查，不作为能力证据。最终 `accepted` 对应最新评估模型；后续模型退化会清除此标记。

## 成功定义与奖励

参考执行完成后，连续1 s姿态误差<0.25 rad、高度误差<5 cm、固定对齐末端 XY误差≤8 cm，无物理/跟踪失败。允许原地踏步；足端跟踪、双后脚接触及速度仅作诊断。越界重新计时，没有严格足端成功率指标。默认参考最早完整首帧成功约4.20 s。

保留根位置0.5、根姿态0.5、身体位置1、身体姿态1、线速度1、角速度1、动作变化−0.1、软限位−10、自碰撞−10、平滑位置保持+0.1。终态后腿部线/角速度误差系数降为0.2，root速度误差系数仍为1，分母保持原身体数量，避免无意提高root惩罚。起立期间速度匹配不变。

新增后腿 thigh/calf 非足部触地代价−0.2：每个物理子步取最强接触力，除以50 N并截到0–1，平均4个子步；feet几何不参加。其几何范围包含后大腿和整段后小腿，不是只测膝点。轻微接触不新增即时终止。记录膝接触占比和峰力。没有增加静态 COM 支撑多边形硬门槛。FR电机目标一直保留2 rad/s、30 rad/s²限制；实测关节/足端速度并非同样硬限幅。

## 正式启动命令（待服务器配置后）

```bash
TRANSITION_PYTHON_BIN=/opt/conda/bin/python \
bash scripts/run_transition_v3_scratch.sh \
  src/tasks/transition_t/data/entries_v3_speed125_train.npz \
  /path/to/fresh/v3_speed125_scratch_run
```

默认4096环境、seed42、总预算30000次PPO更新、每100保存、每250评估。30000是本次更新总预算，不是加载30000模型后加训。每次更新包含24步rollout、5epochs、4minibatches；30000不是30000个物理步。actor75 / critic192 / action12、ELU256/128/128、初始lr3e−4、adaptive KL。

同配置继续训练使用新的输出目录和总目标更新数：

```bash
python scripts/train_transition_entry.py --curriculum three-stage --resume \
  --checkpoint /path/to/v3_scratch_run/model_9999.pt \
  --entry-bank src/tasks/transition_t/data/entries_v3_speed125_train.npz --iterations 30000 \
  --num-envs 4096 --device cuda:0 --output /path/to/fresh/v3_resume \
  --clearml --clearml-task-name 'V3 three-stage resume'
```

独立复核课程0首帧；课程2改成`--level 2 --initialization entries --attempts 128`：

```bash
python scripts/evaluate_transition_entry.py --curriculum three-stage \
  --initialization nominal --level 0 --attempts 64 \
  --checkpoint /path/to/v3_run/model_29999.pt \
  --entry-bank src/tasks/transition_t/data/entries_v3_speed125_train.npz --device cuda:0 \
  --output outputs/v3_nominal_evaluation.json
```

## ClearML

SDK已在本地 mjlab Python 环境安装，`~/clearml.conf`包含官方云服务三个地址与两项凭据，未读取或展示密钥值。启动器默认`--clearml`，项目`Go2/transition_t`，任务`V3 three-stage scratch`；可通过`TRANSITION_CLEARML_PROJECT`和`TRANSITION_CLEARML_TASK`修改。运行时使用 ClearML 配置文件或`CLEARML_API_ACCESS_KEY` / `CLEARML_API_SECRET_KEY`等环境变量。

TensorBoard训练标量自动同步：各奖励、终止原因、跟踪误差、PPO loss/学习率、速度以及新增后膝诊断。独立记录课程level/streak/acceptance、首帧和真实入口成功率、按实际速度与FR误差分桶、各脚误差、终态漂移/门控诊断。完整评估JSON、run配置、晋级事件和finished报告作为 artifacts。

本地checkpoint每100更新保存；ClearML每1000更新上传权重，晋级和训练结束强制上传。在线权重使用OutputModel，输出默认指向ClearML文件服务，避免只登记服务器本地文件路径。离线模式因SDK的DummyModel兼容问题将权重保存为artifact，已验证归档确实包含`.pt`、报告和metrics。正式在线上传权限/容量、API可达性需在服务器开机后验证。

服务器需要：Python中可导入clearml；可访问api.clear.ml、app.clear.ml、files.clear.ml；项目写入和artifact/model上传权限；有效ClearML配置。已有本地配置可在开机后安全复制到服务器，文件权限600，绝不进入git或训练配置快照。现在不需重新提供密钥。用户开机后提供最新SSH端口即可；若希望用指定项目或对象存储，请提供项目名/存储地址。

## 验证与局限

测试、scratch/resume PPO、三阶段物理重建和ClearML离线验证报告位于`outputs/transition_three_stage_20261011/`。CPU冒烟只有少量更新，随机策略成功为0是链路检查结果，不能用来判断30000更新后的表现。V3运动学已验证，但无支撑PD/IDPD/WBC动力学起立未完成；PPO的可学性和最终交接性能必须由正式训练与独立评估证明。服务器GPU冒烟及在线ClearML上传尚未执行。

本地最终验证：完整起立/过渡回归186 tests与22 subtests通过；隔离待推送checkout包含相关键盘回归171 tests与22 subtests通过。8环境从零2次PPO更新、64次首帧评估，resume到第3次通过；三阶段各一次PPO更新/8次评估，检查实际物理、PPO状态、各阶段独立新环境strict resume均通过。ClearML2.1.12离线zip含TensorBoard训练标量、课程/评估曲线、模型与报告。独立三课程CLI名义首帧8次通过执行链路。
