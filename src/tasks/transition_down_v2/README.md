# 双足到三足落地 V2

本任务训练用户已审核的 3.7 秒落地参考。任务 ID 为 `Transition-Down-Go2-v2`，入口为 `scripts/train_transition_down_v2.py`。原 `Transition-Down-Go2-v0`、7 秒数据和旧模型保留。

一个 PPO 网络、Adam 优化器、学习率及计数贯穿全部课程。开始使用 50% 标准起点和 50% 原自适应参考相位；标准起点与微扰通过后加入真实双足老师入口。每 250 次 PPO 更新评估一次，连续两次达标才晋级。40000 是默认总资源上限，可以增加；任何阶段都没有必须训练的固定轮数。

| 阶段 | 评估与默认门槛 |
|---|---|
| 参考学习 → 入口课程 | 标准起点 100 次 ≥90%；初始世界 x 速度 ±0.03m/s 各32次 ≥85% |
| 每档入口 → 下一档 | 标准起点100次 ≥90%，该档64条不同验证轨迹 ≥85% |
| 最终验收 | 标准起点 ≥95%；基座、关节、组合扰动及附加随机化各 ≥90%；四档封存入口各64条 ≥90%；128次新老师链 ≥90% |

每次晋级/验收评估要求没有观测到不安全样本。安全检查使用四腿完整大腿壳和膝中心12cm内小腿壳的真实 mesh 顶点，5ms物理子步采样；评估膝壳最低间隙40mm，训练软目标50mm、硬终止25mm。机身/非足部或FR接地力超过1N即记录为不安全。FL在2.4秒内落地且接触力>5N持续60ms。全程安全记录不会因最后站稳而被抹掉。

成功需完成参考，姿态误差<0.25rad、高度<0.05m、线速度<0.3m/s、角速度<0.8rad/s、关节速度范数<20、FL/RL/RR各>5N、FR无接触且足心高度>32mm、世界足位RMS<80mm，连续保持1秒。训练 standing 奖励仍沿用审核时的原累计规则，评估单独要求连续保持。

数据位于 `src/assets/motions/go2/landing_v2/`，参考SHA为 `a806e849906846e66bf2421afad780132478a93f71c4b4f4a15300ae7b902646`。入口保留39040个原真实物理状态及动作/限速器历史，仅重算相对新参考的FR误差、实际水平速度和版本元数据。轨迹按ID划分：512训练（%5!=0）、64验证（%10==5）、64最终（%10==0）。最终轨迹不用来决定晋级。原数据没有单独保存教师未执行的请求缓存，沿用已审核的以raw/previous action初始化请求缓存的近似，未伪造缺失历史。最大实测入口速度1.079m/s、FR误差0.1305m；难档上界更大不代表已有对应实测覆盖。

原PPO、75/192观测、12动作、2rad/s及30rad/s²的FR目标限速、动作延迟和原随机化保持，新增膝部奖励。两个老师固定SHA，V1模型不能当V2断点恢复。checkpoint保存网络/优化器、标准化、自适应采样、课程、连续计数、门槛和优化配置；改变数据、物理、目标或门槛会拒绝严格续训。

新老师链从新世界开始，先运行双足老师2秒，下降成功后让三足老师保持3秒。命令使用原source集合分布：一半静止标准目标[.12,0,.36]，四分之一随机FR偏移[.07,-.04,.30]到[.17,.04,.40]，最后四分之一再加速度指令vx[-.25,1]、vy[-.15,.15]、wz[-.6,.6]，速度1秒渐入。记录实际接管速度/FR误差以及两次接管的物理/动作缓存差值，后者必须为零。所有者切换不写qpos/qvel，不清动作或限速器历史。只有下降所有者启用参考跟踪终止；所有阶段保留物理安全检查。

## 运行

本地完整小规模执行测试（验证程序，不代表策略成功）：

```bash
MUJOCO_GL=egl PYTHONPATH=. OMP_NUM_THREADS=1 \
  /home/eden/miniconda3/envs/mjlab/bin/python scripts/train_transition_down_v2.py \
  --device cpu --num-envs 4 --smoke --max-updates 2 --eval-interval 2 \
  --output outputs/landing_v2_smoke_new
```

正式训练：

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/train_transition_down_v2.py \
  --device cuda:0 --num-envs 4096 --max-updates 40000 --eval-interval 250 \
  --gate-config src/tasks/transition_down_v2/gates_default.json \
  --output logs/rsl_rl/transition_down_v2/run01
```

续训使用相同配置加 `--resume <latest.pt或model_N.pt>`，并提供新的输出目录。预算是包括断点已有更新的总上限。若断点已处于最终验收阶段，续训执行验收并保留已有更新计数。

Ubuntu桌面直接观察正式V2模型：

```bash
MUJOCO_GL=glfw PYTHONPATH=. python scripts/play.py Transition-Down-Go2-v2 \
  --checkpoint-file <V2模型路径> --device cpu --num-envs 1 --viewer native
```

数据生成命令为 `scripts/prepare_transition_down_v2.py`；输入原冻结入口bank和已审核的参考，输出版本目录及manifest。原参考生成/动力学报告见该目录 `reference_provenance.md` 与 `physics_verified.json`。参考通过标称和低摩擦的诊断控制器测试，初始±.03m/s诊断失败；需要本次PPO训练的结果验证，不能用参考测试代替策略性能。

## 结果文件

`course.jsonl`记录每次验证、课程变化和真实PPO更新数；`evaluation_*.json`保存每次尝试的原因和膝/落地指标。每100轮保存自动模型，每次验证保存`latest.pt`，晋级保存`passed_*.pt`。完成入口课程保存`candidate.pt`后执行最终封存验收；只有全部通过才写`accepted.pt`。`finished.json`明确区分accepted、acceptance_failed、budget_exhausted和smoke_completed，不能因程序退出或预算结束声称训练成功。

远端默认启动脚本 `scripts/run_transition_down_remote.sh`会校验部署文件、生成缺失老师工作空间、执行8世界GPU全流程冒烟测试，再启动连续课程。原固定阶段脚本留档为`scripts/run_transition_down_v1_remote.sh`。
