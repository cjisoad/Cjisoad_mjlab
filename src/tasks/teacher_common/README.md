# Aligned PPO teachers for later DAgger

`pedipulation_t` is the rear-leg standing/FR-operation teacher.
`loco_pedipulation_t` is the quadruped/tripod locomotion/FR-operation teacher.
These tasks prepare independently trained PPO teachers; this change does not
implement student training, online DAgger collection, or a descent teacher.
The original task packages and leg_manip/MoE implementations are unchanged.

Both actors are MLPs (512/256/128, ELU) with the **same 51D observation and 12D
action contract as the MLP leg_manip**. In order: body angular velocity (3),
projected gravity (3), velocity/FR command (6), joint positions relative to the
shared defaults (12), joint velocities (12), previous executed raw action (12),
and absolute base-origin linear velocity in the common anchor (3, scaled by 2).
The last block is not velocity tracking error and is not gated by stance.
Actions use canonical FL/FR/RL/RR hip/thigh/calf order, scale .25, the common
standing defaults, and wrapper clipping at 10. Both use the standing physical
model and its startup physical randomization, including nonzero rotor armature.
New teachers must be trained afresh; old 48D teachers have different frames,
zero points, and/or action defaults and are not compatible checkpoints.

Critics retain the respective source layouts: standing 89D (exact scaled
velocity3 + shared actor first48 + source DR34 + contacts4); locomotion 95D
(shared48 + exact unscaled velocity3 + gait2 + FR-state18 + feet24). They share
the exact actor noise draw for the first48, with separate clean velocities.
No additional privileged actor channels have been selected or enabled.

## Command geometry and the ten-centimeter overlap

Both command terms are named `twist`; commands are `[vx, vy, wz, dx, dy, dz]`.
The gravity-aligned anchor is leg_manip's horizontal projection of body X-Z.
The FR offset is relative to **one quadruped FK zero** in this frame; the world
goal is `base_position + basis @ (zero + offset)`.

| Teacher | Automatic FR dz range, meters from quadruped zero |
|---|---|
| loco_pedipulation_t | .05 to .35 (LOW bank) |
| pedipulation_t | .25 to .72 (HIGH bank) |
| Shared height band | **.25 to .35: 10 cm** |

These are vertical-offset bounds, not absolute foot heights. The two banks
include FK witnesses from the appropriate stance; LOW also checks interpolated
Cartesian IK/collision clearance. Both banks contain targets in the height
overlap. The banks need not have identical discrete xyz samples or xy ranges;
height overlap alone does not establish that every xyz target is reachable in
both stances. Given the same state and command, both tasks nevertheless decode
exactly the same physical goal. Retain leg_manip's mode hysteresis or blend
teacher actions during a later handoff; overlapping goals by themselves do not
ensure continuous actions.

Automatic targets use the fixed common bank, replacing the legacy source
workspace's joint_delta/max_offset/lift_range/foot_target_* workspace limits.
Those inherited legacy bank-generation fields do not control the new target
bank. Source velocity sampling, standing probabilities and curriculum remain;
loco advances through its original four stages (the full LOW bank is stage3).
The teachers do not run leg_manip's mixed-stance promotion exams.

For the standing teacher, source no-operation commands are translated to the
common nominal biped offset (approximately [.13349, .03060, .47804]). This
expresses biped stance to a future unified student while keeping source FR
default-pose rewards active. `has_foot_target` is false for this nominal target
and the zero sentinel. Actual operation targets enable source FR tracking.
Source rewards execute unchanged through a read-only `stand` -> `twist` view;
their weights, parameters, ordering and batch-wide height gate are preserved.
Stand symmetry preserves the nominal command while reflecting body/joint/velocity
observations; real right-only operation samples receive no symmetry loss.

## Training, play and future DAgger

Run from the repository using the configured Python environment:

```bash
python scripts/train.py pedipulation_t --env.scene.num-envs 4096
python scripts/train.py loco_pedipulation_t --env.scene.num-envs 4096
python scripts/play.py pedipulation_t --checkpoint-file /absolute/path/model.pt --viewer viser
python scripts/play.py loco_pedipulation_t --checkpoint-file /absolute/path/model.pt --viewer viser
```

Default PPO training retains physical DR, original source pushes, sensor noise,
random control delay and stochastic Gaussian exploration. Standing source
pushes set xy velocity within +/-.4 m/s and angular velocity within +/-.6 rad/s
every8s. Locomotion source pushes occur every5-6s; the source configuration
sets xy within +/-.15 m/s and yaw within +/-.2 rad/s (the resulting velocity
dictionary contains only those three axes). The original rewards are not retuned.
`play=True` removes runtime pushes and policy sensor noise, but retains physical
DR, random action delay and source reset randomness. It is **not** a fully
nominal-physics evaluation. A nominal evaluation must explicitly control these
separate sources of variation and keep physically meaningful rotor armature;
simply deleting the physics event leaves the source standing armature at zero.

DAgger does not require a disturbance-free teacher. MimicAgent trains its
privileged teacher with PPO and explicitly uses physical DR and observation
noise in both teacher/student stages. Its teacher actor and critic both receive
additional privileged state; these initial 51D actors are an interface-aligned
baseline, not a reproduction of that full privileged architecture.

Later DAgger should query the frozen teacher on the student's actual state,
command, anchor and physical instance, using the student's last executed action.
Querying clean teacher observations and using mean actions is a recommendation
to reduce label variance. Keep student-side noise and relevant physical DR;
do not advance simulation or resample commands just to query a teacher.
Labels must use the same clipped 12D action convention as execution, not PD
angle targets. Stronger actor privilege requires a separate input specification
and usually student history or a recurrent/adaptation module.

Checkpoints carry `teacher_contract` including task identity, observation
dimensions, frame, zero, overlap, joint order/defaults, actual action scale/clip
and delay. Custom experiment names do not change task identity. Incompatible
contracts are rejected before load; loco curriculum/metrics are saved/restored.
Checkpoint save uses the base runner, avoiding velocity-runner ONNX metadata's
assumption of a generic joint action. Explicit base ONNX export remains available.

## Privileged-state options (not enabled by this change)

For **pedipulation_t**:

1. Minimal critic additions: base and stand_goal heights, FR goal error,
   episode-based rear-gait sin/cos, operation flag and current height gate.
2. Recommended balance critic: option1 plus RL/RR positions relative to base
   and rear contact forces. Keeps actor51 while clarifying support and balance.
3. Strong privileged actor: append height, FR error, feet contacts and rear-foot
   positions; use option2's critic and optionally actual friction/motor offsets.
   Changes teacher actor ABI and increases the student's information gap.

For **loco_pedipulation_t**:

1. Actor51 baseline; add physical DR to the existing contact/phase/FR critic.
2. Recommended stronger teacher actor: append gait sin/cos, FR phase/reference
   state and feet contacts. Particularly useful for swing timing, three-foot
   support and landing, but future student needs history to infer hidden state.
3. Full-state actor: option2 plus foot velocities/heights/air times/forces,
   base height and actual dynamics/actuator parameters. Largest observability
   gap and engineering cost; useful as a teacher-performance upper bound.

The existing source DR34 vector is not complete physical truth: its friction
slot is zero, restitution is repeated, and motor offsets are absent. Actual
friction and motor offsets need separate extraction before recommending them
as informative privileged channels.

Sources: https://arxiv.org/html/2609.24145v1 and
https://proceedings.mlr.press/v15/ross11a.html.
