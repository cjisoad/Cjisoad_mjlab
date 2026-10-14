# Balanced FSM teacher recording

**Goal:** Record actual teacher execution for the 48D GRU student, with balanced
coverage of locomotion, manipulation, and support/actor transitions.

**Design:** Preserve continuous batched episodes in compressed NumPy shards.
Create equal-size reservoirs of contiguous sequence indices for each of ten
categories, independently in training and validation. The supervised anchor is
the last frame of each indexed sequence. Earlier frames provide GRU context.
Use episode-level splits and never index across resets, invalid frames, or gaps.

Categories: quadruped_walk, quadruped_stand, tripod_walk, tripod_manipulation,
biped_walk, biped_manipulation, quadruped_to_tripod, tripod_to_quadruped, rise,
down. Four-foot FR lifting belongs to the support transition category.

The original full-chain environment has one globally selected motion reference;
rise and down are synchronized within a batch. Failed worlds are excluded until
the next batch and cannot restart a competing transition. Independent scenarios
start from quadruped or the existing physical biped entry reset.

**Files:**
- `src/tasks/fsm_student/recording.py`: classification, data validation,
  failure-tail exclusion, balanced window reservoirs, atomic archive writing.
- `scripts/record_fsm_teacher.py`: CLI, scripted operator commands, actual
  actuator capture before automatic reset, progress and dataset manifest.
- `tests/test_fsm_teacher_recording.py`: balance, boundaries, timing, CLI,
  operator scenarios, and substep capture.
- `docs/fsm_teacher_recording.md`: usage, quotas, schema, training semantics.

## Work

- [x] Write tests; run them and verify missing recording implementation fails.
- [x] Implement frame classification from the actor/support state actually used
  during that control step. State labels are metadata, not student inputs.
- [x] Implement 48D pre-action observations paired with that period's final
  applied common action, retaining all actual physics-substep actions and the
  final PD target. Capture before env.step can reset a terminal world.
- [x] Automate quad/tripod, rise/full-chain, and biped/down episodes. Use raw
  six-value operator commands only; no R input bit or reference command leakage.
- [x] Balance train/validation window anchors independently, preserving original
  episode data. Deficits choose the next scenario. Incomplete coverage must be
  reported and return nonzero unless explicitly allowing a partial recording.
- [x] Save shards and manifests after each batch; handle an interrupted current
  batch. Record checkpoint hashes, observation/action contracts, physics dt,
  label timing, category counts, failures, and scenario completion.
- [x] Run unit tests and a small real CPU recording covering all ten categories.
- [x] Review implementation and validate the actual saved dataset and counts.

Verification commands:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_fsm_teacher_recording.py tests/test_fsm_student.py`
and `python scripts/record_fsm_teacher.py --device cpu --num-envs 2 --windows-per-category 4 --sequence-length 8 --max-batches 12 --output-dir outputs/fsm_teacher_recording_smoke`.
