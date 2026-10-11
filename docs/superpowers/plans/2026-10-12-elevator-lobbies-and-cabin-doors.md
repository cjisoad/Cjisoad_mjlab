# Elevator lobbies and cabin doors implementation plan

> **For agentic workers:** Execute inline using superpowers:executing-plans. Steps use checkboxes for tracking.

**Goal:** Apply the user's furniture corrections and replace moving landing walls with a door-equipped car and three connected lobbies.

**Architecture:** Keep room floors at z=0/2.8/5.6. Shift the elevator center from x=4.15 to x=5.9; add static lobbies between x=3 and x=4.82. Replace three floor-mounted gate joints with two sliding door leaves parented to the car. The controller waits for both leaves to close before travel and opens both after alignment. Preserve all logical button names and physical press behavior.

**Tech Stack:** MJCF, MuJoCo, Python, pytest, existing GLFW/EGL browser viewer.

- [x] Update scene and controller tests to expect connected lobby floors and two car-mounted doors, then run them against the old model to confirm failure. Extend integration checks to verify doors move with the car and stay closed during travel.
- [x] Add four legs to each bench and chair, shift benches 0.5 m right, use seat/back/leg collision geoms, and clear the switch approach.
- [x] Remove `gate_body_f1..3` and `gate_f1..3`. Add `lobby_f1..3`, `lobby_floor_f1..3`, lobby guards, call panels, and a ground approach. Set car center x=5.9, platform half-depth=1.4, with framed transparent front/right panels and a roof.
- [x] Add `joint_car_door_front/back` with opposing Y axes, range 0..0.65, authored initially open. Add matching `car_door_front/back` actuators. Door leaves close the 1.3 m left-side entrance and retract behind fixed side panels.
- [x] Replace gate mappings in the controller with the two door mappings, retain close-before-travel/open-after-stop/dwell behavior, and run the scene suite including 15 kg payload.
- [x] Update README and initial preview; render the changed scene and verify browser clicks and a complete elevator trip. Restart the existing localhost viewer on the changed model.

Only task files are in scope; preserve all unrelated workspace changes. The user's current correction authorizes these changes, including removal of the previous landing-wall behavior and addition of static lobbies.

## 验证记录

22 项测试通过，包括门随轿厢运动、关闭后运行、到站打开、平台连通、整段行程避碰及 15 kg 载荷。已用 EGL 检查初始开门和运行关门画面；全部三层灯开关和候梯按钮的总览坐标拾取有效。原有楼层移动墙及门槛已删除，避免轿厢框架与静态平台边缘相撞。网页服务已在 8765 端口重启。
