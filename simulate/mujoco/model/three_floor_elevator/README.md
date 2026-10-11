# Three-floor open-room elevator scene

This standalone MuJoCo environment contains three matching furnished rooms and an adjacent elevator. The rooms show one rear wall; the front and side walls are invisible collision boundaries so a robot can enter only through the elevator opening while the camera can still see inside. The scene does not include a robot or locomotion policy.

## Run

From the repository root:

```bash
python simulate/mujoco/model/three_floor_elevator/run.py
```

An alternate MJCF can be supplied with `--xml PATH`. The window starts with the three-floor overview, and the elevator starts at floor 3 with that landing open.

Left-click a visible plunger to apply force to its physical slide joint. A robot can trigger the same input by making contact and moving the plunger past its press threshold. Right-drag rotates the view and the scroll wheel zooms.

Logical button names:

- `light_f1`, `light_f2`, `light_f3`
- `call_up_f1`, `call_up_f2`, `call_down_f2`, `call_down_f3`
- `cab_f1`, `cab_f2`, `cab_f3`

The Python boundary is `ThreeFloorElevatorController(model, data)`. A robot-composed MJCF can be loaded through the same controller as long as the environment's named joints, actuators, lamps, and indicator geoms remain available. No robot-specific code is needed here.

The lift uses an idealized speed-limited position target. Furniture is made from simple primitive geoms. The custom GLFW window is required for mouse interaction; the MJCF and controller can also be used in a separate headless simulation.
