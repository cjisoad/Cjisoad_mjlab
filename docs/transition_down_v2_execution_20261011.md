# 双足到三足落地 V2 实施记录

用户已审核新的3.7秒参考，并授权按性能门槛推进训练、合入当前分支并推送及远端训练。本次新增独立 Transition-Down-Go2-v2，V1 数据和模型保留。

默认晋级门槛：标准起点90%，微扰85%，入口85%，连续两次验证；最终标准95%、其他90%。安全评估膝壳40mm，训练软目标50mm/终止25mm；每5ms记录。每250轮评估、每100轮保存，40000是总资源预算，没有固定阶段轮数。

版本数据及来源报告已入库：src/assets/motions/go2/landing_v2。入口仍是39040真实状态/640轨迹，512训练、64验证、64封存最终。数据/随机化/PPO/老师/执行限速契约均随checkpoint记录；V1 checkpoint被拒绝。数据明细和局限见该目录及src/tasks/transition_down_v2/README.md。

隔离工作树测试和主工作区集成测试均为44 passed、7 subtests passed。测试覆盖课程、封存拆分、四膝mesh FK、真实物理子步、2次PPO参数更新、优化器/学习率/课程/计数恢复、门槛改变拒绝，以及验收阶段恢复失败时仍能写报告、计数不增加。完整CPU CLI另实际执行三个frame0评估、真实入口恢复与老师链；4个源老师世界均到达接管，qpos/qvel/三帧动作/请求及执行缓存/限速器差值全部0。两轮模型没有被标记为验收通过。

验证证据保存在 outputs/transition_down_v2_validation_20261011，包括 integrated_tests.log、cli_final/verification.json、smoke_entry.json、smoke_chain.json 和 local_validation.json。独立只读审查发现链评估timeout/所有者跟踪/初始命令及验收恢复计数边界，均已修复并补测。

远端启动脚本改为V2 GPU冒烟后单次连续课程；旧固定阶段脚本留为run_transition_down_v1_remote.sh。服务器42311已连通，GPU空闲。重启后的EGL运行库和老师工作空间正在补齐；实际远端状态会另记到本地outputs的remote_launch.json，未确认真实PPO日志前不宣称训练启动。

未改动用户正在编辑的正向任务及README等内容。
