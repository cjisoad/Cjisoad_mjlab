# Transition Joint Target Limiter

The user approved adding the same joint-target velocity and acceleration limiter
to transition training and playback, then committing and pushing the changes.

The policy still emits twelve common-coordinate joint-position offsets. A batched
Torch limiter runs after action selection and any teacher delay, before the PD
position target is written at each 0.005 s physics substep. Its state is the last
commanded position and target velocity. All action histories exposed to the next
observation reflect the target actually applied, expressed in common action
coordinates. The original actor75/critic192 layout stays unchanged.

The limiter bounds target velocity and acceleration, including target reversals.
It decelerates toward a fixed target using a stopping-distance velocity envelope;
it must not snap to a target when doing so would violate acceleration. These are
command bounds, not guarantees on measured joint or Cartesian foot velocity.
Jerk constraints, automatic hardware fault handling and reference retiming are
outside this change.

The option is disabled by default for unrelated PedipulationPositionAction users.
Transition training, evaluation and the actor75 keyboard chain enable the same
per-joint configuration. Initial simulation settings in canonical FL/FR/RL/RR
order are 2 rad/s and 30 rad/s^2 for FR; the other nine joints pass requested
targets through unchanged. JSON null / Python None identifies unrestricted
joints. A matching-seed 64-world diagnostic gave 56 survivors unlimited, 55 with
FR-only limits, 14 with support-only 6/120 limits and 9 with all joints limited.
The frozen source teacher depends on fast support-leg commands, so support-leg
limits were removed from this change. The reference peaks are 1.60 rad/s / 14.38 rad/s^2 for FR and
4.00 rad/s / 54.58 rad/s^2 overall. These initial settings require hardware
validation and do not establish a safe operating envelope for the real robot.

The teacher delay holds the previous requested command; a separate executed
action history feeds observations and rewards. Reusing the executed position
as the delayed request would repeatedly brake the limiter despite constant input.
FR limiting also changes frozen endpoint-teacher execution and is recorded in
collection metadata; runtime smoke does not demonstrate adapted teacher performance.

On physical reset, limiter initialization is deferred until the final reset pose
has been written. Teacher handoffs keep its state continuous. Newly collected
entry banks store limiter position, velocity and configuration; restoration uses
those states, preserving the first command and action histories. Old nominal
entry banks are rejected for limited training rather than inventing missing
command velocities. Historical bank inspection remains available.
The bank's action zero and scale must match the current executor, and reset
mixture/split overrides are recorded and checked when resuming entry training.

New checkpoint contracts record the limiter algorithm, limits, timestep and
executed-action semantics. Strict loading and resume reject any differences.
Warm starting from an old unlimited transition checkpoint is allowed only via
an explicit migration path that verifies every other contract field, retains
actor/critic/normalizers and starts a fresh optimizer/course. A keyboard flag
permits an explicitly labelled old-policy-with-limiter experiment; default
loading never silently changes a checkpoint's execution contract.

The repository's Go2 C++ controller currently runs a velocity policy, not this
transition task. This change implements the limiter in the Python transition
execution/playback path and provides a reusable controller with explicit dt;
it does not claim the existing C++ velocity deployment runs transition_t.
Future student distillation should supervise executed, limited teacher actions
and trajectories. Distillation alone does not enforce hard command bounds; the
same limiter belongs in the student's execution path when deployment is implemented.

Validation covers config rejection, substep velocity and acceleration bounds,
reversal/braking, settling, partial resets, executed-action histories, keyboard
handoff continuity, bank roundtrips, strict checkpoint mismatch and authorized
legacy warm start. Real CPU rollout/PPO smoke checks finite targets, observations,
rewards and checkpoint restore. A bounded old-policy comparison measures FR
joint/foot speed and standing rate without treating it as a trained improvement.

Only transition entry/keyboard changes and their required dependencies are
committed. Unrelated local documents, workspace tests and README edits stay in
the worktree. Push targets the existing origin/wsl-validation branch.
