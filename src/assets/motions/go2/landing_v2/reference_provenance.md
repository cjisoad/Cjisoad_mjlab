# 双足到三足落地参考 V2：膝部保护修订

状态：**新版参考已生成，标称 MuJoCo 力矩动力学复测通过；受扰动入口测试失败，尚未重新训练策略。**

文件 `biped_to_tripod_reference_v2.npz`，SHA256：
`a806e849906846e66bf2421afad780132478a93f71c4b4f4a15300ae7b902646`。
50 Hz、186 帧、3.7 秒，原任务 NPZ schema，12 个关节、29 个刚体。

## 改动与实测

重新求解落脚、基座、重心路径，随后用真实关节力矩执行，再将实测姿态降采样为新参考；未将旧轨迹直接加速。
候选主要下降时间为 2.0 秒；最终参考还包含初始保持与落地稳定阶段。
FL 提早落地，FR 保持悬空，FL/RL/RR 成为最终支撑。
膝部目标约束检查后腿完整大腿外壳及距膝中心 12 cm 内的小腿外壳，优化目标 52 mm；动力学控制器附加 50 mm 外壳高度屏障。
另用独立 `audit_knees.py` 复查四条腿全部对应外壳顶点，未仅检查膝关节中心或细碰撞胶囊。

| 指标 | 旧参考 | 新参考 V2 | V2 独立动力学复测 |
|---|---:|---:|---:|
| 参考/记录总时长 | 7.0 s | 3.7 s | 4.7 s（额外保持 1 s） |
| FL 首次接触 | 5.0 s（标签） | 1.66 s（源实测） | 1.585 s（实测） |
| COM 相对后脚向前增加 20 mm 的时刻 | 5.50 s | 1.38 s | 1.28 s |
| 四腿膝壳最低离地高度 | 22.00 mm | 50.92 mm | 54.14 mm |
| 膝壳低于 40 mm 的累计采样时间 | 3.64 s | 0 s | 0 s |

“后坐”主要体现为旧参考长期不向前转移 COM，并让后膝长时间保持极低姿态；旧参考 COM 世界 x 没有大幅负向移动。
新参考 COM 世界 x 最小值比起点低 2.87 mm；相对后脚的 COM x 未进一步后退，较早向前移动。
各腿膝壳最小离地高度详见 `all_knees_audit.json`。

新参考来自实际物理状态，因此如实保留了最大 **28.49 mm 后脚位移**，没有将实测脚滑动重新伪装成固定脚。
最终参考关节姿态与原教师终点的 12 关节差值二范数为 0.171 rad，根部位置差约 0.26 mm；终点姿态接近但不强制改写为旧端点。

## 动力学验证方法及边界

`replay.py` 使用原始 Go2 物理模型、5 ms 步长、30 次 MuJoCo 求解迭代、关节力矩限值 23.7/35.55 Nm。
只在起点写 qpos/qvel，之后仅关节力矩和 `mj_step`，不施加根部外力，不逐帧重置物理状态。
控制器为接触约束逆动力学诊断控制器（SLSQP），不是原 PPO 的位置动作/FR 目标限速执行链。
膝屏障中忽略膝点 Jacobian 的时间导数，不能据此给出硬件安全保证。

| 工况（新参考 SHA 相同） | 结果 | 最低膝壳间隙 | 说明 |
|---|---|---:|---|
| 摩擦系数 1.0、标准初态 | 通过 | 54.14 mm | 无非足部触地、无 FR 触地、最终 FL/RL/RR 支撑 |
| 摩擦系数 0.6、标准初态 | 通过 | 57.32 mm | 最大基座跟踪误差达 140.2 mm，跟踪精度有明显退化 |
| 初始世界 x 速度 +0.03 m/s | 失败 | 膝壳穿地 | 1.19 s 起头部/髋部触地，后续跌倒 |
| 初始世界 x 速度 −0.03 m/s | 失败 | 膝壳穿地 | 1.26 s 起小腿触地，1.28 s 终止 |

标称复测 941 个物理子步样本，膝碰撞几何最小间隙 53.10 mm。
标称最大力矩达到限值，最大瞬时关节速度 **6.10 rad/s**；FR 最大实际关节速度 **1.42 rad/s**。
参考降采样后最大关节速度 3.37 rad/s、FR 1.31 rad/s，两者不能替代 5 ms 实测冲击峰值。
3 个落地时刻的 SLSQP 返回迭代上限，但动力学等式残差小于 3e-14、力矩/膝屏障违反量为零；原始信息保存在 `physics_verified.json`。

这些结果支持“膝部间隙更大的较快标称参考”，尚不支持“真实老师入口下可直接部署的策略”。
后续训练应加入膝壳间隙/非足部触地惩罚或终止、下降阶段约束和老师入口扰动评估，并重点复查落地冲击及脚滑。
V2 与旧模型的数据 SHA 契约不同，不能直接替换冻结文件并声称旧模型训练有效。

## 在原生 MuJoCo 中观察

从 Ubuntu 图形桌面终端运行：

```bash
cd /home/eden/unitree_rl_mjlab
MUJOCO_GL=glfw PYTHONPATH=. MPLCONFIGDIR=/tmp/landing_revision_mpl \
  /home/eden/miniconda3/envs/mjlab/bin/python \
  outputs/transition_down_landing_revision_20261011/view.py --mode physics
```

`1` 旧参考，`2` 新参考，`3` 已记录的真实动力学轨迹；空格暂停，`R` 重播，左右键暂停逐帧，`[`/`]` 调节速度。
黄色标记为四腿膝关节中心，绿色点为 COM 地面投影；屏幕显示时间和后膝外壳最低间隙。
这里的模式 3 回放已记录的物理轨迹，查看器自身不重新积分动力学。

要在窗口中现场重新执行关节力矩动力学测试：

```bash
cd /home/eden/unitree_rl_mjlab
MUJOCO_GL=glfw PYTHONPATH=. OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/landing_revision_mpl \
  /home/eden/miniconda3/envs/mjlab/bin/python \
  outputs/transition_down_landing_revision_20261011/replay.py \
  --motion outputs/transition_down_landing_revision_20261011/biped_to_tripod_reference_v2.npz \
  --output outputs/transition_down_landing_revision_20261011/live_test \
  --mode wbc --viewer
```

空格暂停；仿真结束后窗口保持最终姿态，关闭窗口后写出 `live_test.npz/json`。
它是诊断控制器仿真，不是新训练策略。

## 复现与文件

所有命令使用上述 Python；建议 `MUJOCO_GL=egl PYTHONPATH=. OMP_NUM_THREADS=1` 无窗口执行。

1. `generate_selected.py --duration 2.0 --landing .72 --clearance .052 --com-start .20 --com-end .88`：选中候选的约束 IK。`generate.py` 保留后续 75 mm 候选实验，不用于复现选中候选。
2. `replay.py --motion .../candidate_2.00_land0.72_knee0.052_com0.20_0.88.npz --output .../physics_candidate04_verified --mode wbc`：生成原始实测状态。
3. `refine_from_physics.py`：从实测状态降采样，生成 `biped_to_tripod_physics_refined.npz`；其字节内容即最终 `biped_to_tripod_reference_v2.npz`。
4. `replay.py --motion .../biped_to_tripod_reference_v2.npz --output .../physics_verified --mode wbc`：独立复测。
5. 附加 `--friction .6`、`--push .03`、`--push -.03`：复现另外三个工况。
6. `verify.py` 检查 schema、SHA、有限数、单位四元数、接触、参考速度、实测膝间隙/触地/力矩和物理采样；`audit_knees.py` 独立检查四膝。

完整来源、依赖文件 SHA、Python/物理库版本、成功与失败工况汇总见 `manifest.json`。
`reference_v2_audit.json` 是新参考审计；`physics_verified.npz/json` 是独立实测；`landing_comparison.png` 是原新版对照图。
其他 `candidate_*`、`physics_candidate*`、旧控制器副本和日志保留用于追溯试错过程。

**已排除的无效试验：** `policy_probe.py` 自定义旧策略迁移入口连旧参考对照都未复现原正式评估结果，因此所有 `policy_*.json/npz` 不作为策略性能证据。
`baseline_existing_evaluator.json` 记录旧策略正式评估入口四组各一次均通过。
本次未重新训练 PPO；新数据独立保存于本目录，旧冻结数据与旧训练成果保留。
