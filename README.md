+ 启动fsm-teacher全量键盘测试：python scripts/keyboard_transition_fsm.py   --device cuda:0   --initial-state quadruped/biped

+ 启动四足键盘测试：python scripts/keyboard_teacher.py loco_pedipulation_t   --checkpoint-file mark/quad_teacher.pt   --device cuda:0   --num-envs 1   --workspace quadruped


+ 启动mimic键盘测试：python scripts/play.py Transition-Go2-v0 --checkpoint-file logs/rsl_rl/transition_t/2026-10-09_01-13-48_transition_t_direct_4090_20261009/model_30000.pt --motion-file outputs/teacher_endpoint_rise_clearance20mm_20261008/tripod_to_biped_reference.npz --num-envs 1

+ sswitch切换供应商后session手动切换：codex resume -c 'model_provider="openai/custom"'

+ 启动测试环境：
  - 平地：python simulate/mujoco/model/three_floor_elevator/run.py
    + web端：使用--web参数，默认链接：http://127.0.0.1:8765/
  - 楼房：python scripts/run_agentic_elevator.py --device cuda:0 --initial-state quadruped
    + 默认原生 MuJoCo 窗口；HTTP 运控接口：http://127.0.0.1:8766
    + 浏览器界面：启动时添加 `--viewer web`
