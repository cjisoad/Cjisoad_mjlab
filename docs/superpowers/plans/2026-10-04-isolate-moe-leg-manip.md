# Isolate MLP and MoE leg manipulation tasks

Goal: Restore leg_manip to fd133c1 (the approved MLP version before MoE integration),
and preserve c2ddaed task behavior in a separate moe_leg_manip package.

Architecture: Copy the complete current leg_manip task, including its MDP, runner,
PPO, routing diagnostics, and vendored network into src/tasks/moe_leg_manip.
Rewrite its internal absolute imports. Restore leg_manip and its keyboard entry
to fd133c1. Register moe_leg_manip with four learned dense actor experts and the
original MLP critic; provide scripts/keyboard_moe_leg_manip.py for existing MoE
checkpoints. Tensor state keys remain unchanged for checkpoint compatibility.

Constraints: Preserve rewards, courses, observations, exploration, and optimizer
settings. Do not change active server training or local viewer processes.

- [x] Add failing registration and task-isolation tests.
- [x] Copy current task; restore baseline; split keyboard entry and update tests.
- [x] Run task regressions, verify exact baseline restoration and full snapshot.
- [x] Load existing MLP and MoE models through their respective keyboard scripts.
- [x] Run bounded real PPO updates for both task IDs and verify finite weights.

Validation: 150 leg_manip regression tests passed. Both registered tasks completed
two full CUDA PPO updates (64 environments, five epochs, four minibatches) with
finite outputs and successful checkpoint save/reload. Both keyboard entry points
passed 550-step command/transition/reset checks using the existing MLP model_1000
and MoE model_11300. The 22 original task files and original keyboard script match
fd133c1 exactly. The 27 copied non-registration MoE files match c2ddaed except
for namespace and task-name substitutions. Only leg_manip and moe_leg_manip are
registered. Existing task IDs and checkpoint tensor formats are documented in
src/tasks/moe_leg_manip/README.md.
