# FSM distillation: restored final-substep supervision

The native-output experiment was reverted at the user's request after local
observation. The previous supervision/control scheme is restored:

- Four frozen actors are selected by the FSM, with the original common-angle
  adapters, phase-dependent target governor and substep delay handling.
- The teacher previews four 5 ms position targets. The final applied target,
  encoded as a 12-dimensional common action, is the supervised label.
- Teacher-controlled episodes execute the four original substep targets.
- Student-controlled episodes use the existing direct 20 ms PD hold, without
  a target governor on the student path.
- The original observation/action/teacher contracts are restored, including
  `teacher_final_physics_substep_applied_common_action`. Existing models from
  that scheme retain their original semantics. Native-output experimental
  models do not match this restored contract.

The original mismatch between teacher substep execution and student holding
the last target is intentionally restored, not claimed fixed. Standalone
teacher training configurations and server jobs were not changed.

## Evaluation fixes retained

Evaluation schema version 2 derives required holds from the actual scenario
stages. Quad requires tripod operation and final quadruped return. Biped and
cycle require biped stance and final quadruped return; cycle does not require
a `tripod_operation` stage it never executes.

Biped success requires a continuous 0.5 s valid posture/height/support hold.
Each hind foot must have made contact above 5 N within the last 0.1 s of the
current biped stages. Brief contact flicker is tolerated; missing or prolonged
one-sided support and invalid posture still fail. Results disclose the exact
criteria. This scoring change does not demonstrate improved training.

The `teacher`, `teacher-hold` and `compare-teacher-timing` modes are restored:

```bash
python scripts/evaluate_legmanip_distil.py --policy teacher --scenario all --device cuda:0
python scripts/evaluate_legmanip_distil.py --policy student --checkpoint-file path/to/student.pt --scenario all --device cuda:0
python scripts/evaluate_legmanip_distil.py --policy compare-teacher-timing --scenario all --device cuda:0
```
