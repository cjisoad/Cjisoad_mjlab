# Native-action FSM distillation

The `legmanip_distil` task selects one of four frozen actors (quadruped, rise,
biped, down) with the existing FSM. The supervised target is now the actor's
native 12-joint position-offset request, expressed in the common default-angle
coordinates. Endpoint adapters only translate the default-angle reference;
they do not filter the request or select a physics substep.

Both teacher-controlled and student-controlled episodes use the same action:

```
q_target = common_default_angles + 0.25 * clamp(action, -100, 100) + motor_offsets
```

The target is held for the whole 20 ms control interval (four 5 ms physics
steps). Neither path has a target velocity/acceleration governor, a governor
velocity history, or random substep delay. The numerical action clamp, PD
gains and actuator torque limits remain physical execution safeguards. Labels
themselves retain native actor values, before that numerical clamp.

The student remains a 48-input, 12-output GRU. DAgger still selects the
controlling policy once per episode and queries the frozen teacher on the
actually visited state. Standalone teacher training configurations are not
modified. In particular, the existing rise/down checkpoints were trained with
a governor; removing it from their distillation execution changes their
training conditions and requires behavioral evaluation.

## Checkpoint semantics

The action contract label is `teacher_native_common_action`, with
`target_limiter: null`, `delay: false`, and `zero_order_hold_direct_pd` execution.
Full resume and deployment validate the action contract, rejecting older
filtered-final-substep artifacts. Their architecture is unchanged, but their
supervision semantics are different; they are not equivalent native-output
models. Do not resume the old training job as though only a scorer changed.

Transition actors are still loaded against their original checkpoint contracts
in temporary validation worlds. Those worlds are never stepped for live
distillation and do not filter the labels or action execution.

## Evaluation

Run from the repository with its configured Python environment:

```bash
python scripts/evaluate_legmanip_distil.py --policy teacher --scenario all --device cuda:0
python scripts/evaluate_legmanip_distil.py --policy student --checkpoint-file path/to/new/student.pt --scenario all --device cuda:0
```

`teacher-hold` and `compare-teacher-timing` were removed: the native teacher and
student now have the same target-hold execution, so the previous timing
comparison no longer describes this task.

Evaluation schema version 2 derives required holds from the actual scenario
stages. Quad requires tripod operation and final quadruped return; biped and
cycle require biped stance and final quadruped return. Cycle does not contain
a `tripod_operation` stage and therefore does not require its hold.

Biped success still requires a continuous 0.5 s valid posture/height/support
hold. Each hind foot must have made contact above 5 N within the last 0.1 s
of the current biped stages. This tolerates contact sampling flicker and brief
gait gaps; absent support, prolonged one-sided support, invalid posture or
scenario failure still fail. This is an explicit scoring-policy change, not a
claim that a new model has learned better. Results include the grace interval
and per-scenario required holds for reproducibility.
