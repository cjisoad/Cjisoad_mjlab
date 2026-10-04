# Aligned teacher tasks implementation plan

> Execute inline with tests and independent review.

Goal: register pedipulation_t and loco_pedipulation_t for independent PPO teacher
training, ready for later student-state DAgger queries.

Architecture: new teacher_common holds command adapters and checkpoint handling;
the two task packages retain their respective source rewards and curricula.
Existing leg_manip, moe_leg_manip and source task behavior stays unchanged.

Constraints: actor 51, critic source layouts 89/95, MLP 512/256/128 ELU,
canonical FL/FR/RL/RR 12 actions, scale .25, wrapper clip 10, leg_manip model,
gravity-aligned body X-Z anchor and shared quadruped FK zero. LOW dz .05-.35,
HIGH dz .25-.72; preserve the .25-.35 ten-centimeter overlap. Nominal biped
offset encodes stance, not FR operation. Reward functions and weights unchanged.
Training retains physical DR, pushes and sensor noise. Play removes pushes and
sensor noise, retains physical DR and control delay; nominal evaluation must
explicitly disable DR/delay while keeping nonzero rotor armature.
Additional actor privilege and student/DAgger training are outside this change.

- [x] Write and run missing-task/ABI/overlap/reward-preservation tests (red).
- [x] Implement common command, observation, reward compatibility and PPO runner.
- [x] Register both task names and document train/play/evaluation contracts.
- [x] Run new tests, old-task regressions, real CUDA PPO and checkpoint roundtrips.
- [x] Independent review completed. Commit and verified push are final delivery steps recorded separately.


Validation: 13 teacher command/configuration/GUI/symmetry tests passed plus one
real CUDA integration test (64 environments per teacher, two PPO updates,
five epochs/four minibatches; exact policy save/load, actual action-scale/clip
mismatch rejection, custom experiment-name safety, changed curriculum state
restored). 150 leg_manip, 27 pedipulation and 22 loco_pedipulation regression
tests passed. Both train/play help commands parse, and both official train.py
entry points completed one update with configuration/log/checkpoint output.
Independent review identified and resolved nominal-command mirror semantics,
GUI/reset handling, task identification and actual checkpoint action metadata.
Existing four task packages are byte-for-byte unchanged relative to 7d2b84e.
Additional actor privilege remains a proposal; no full teacher convergence or
student DAgger training is claimed by these bounded verification runs.
