# Loco teacher unified return implementation plan

**Approved design:** Merge descent and recovery for `loco_pedipulation_t`. Keep
`blend=1` for tripod and `blend=0` for quadruped. Recover support by elapsed time,
keep FR reference tracking throughout the return, and exit only after both the
recovery duration and continuous ground-contact confirmation are satisfied.

**Architecture:** Enable the behavior through `unified_return` on the shared
command configuration, enabled by the loco teacher factory. Reuse phase slot 4
as RETURN and retain the six-channel critic phase layout; slot 5 remains for
the source task's legacy RECOVER. Actor and critic dimensions remain 54/95.

**Tech stack:** Python, PyTorch tensor operations, mjlab command/reward managers.

## Task 1: Command state and timing

Files: `src/tasks/loco_pedipulation/mdp/commands.py`,
`src/tasks/teacher_common/env_cfg.py`.

- [x] Add `unified_return: bool = False` to the shared configuration and enable
  it in the loco teacher factory.
- [x] Define `RETURN = LOWER` so existing phase indices and outcome counters
  stay consistent. The teacher never enters the legacy RECOVER phase.
- [x] In RETURN, retain the existing smooth descent over `reach_time` (0.7 s),
  and set `blend = _blend_start * (1-smoothstep(elapsed / (reach_time+recover_time)))`.
  Default support recovery therefore takes 1.0 s. Starting from partial REACH
  preserves the current blend and reference instead of jumping to full tripod.
- [x] Finish directly in QUAD only after the recovery duration and 0.06 s of
  continuous contact above 1 N. Keep the ground reference and blend zero if
  time completes without contact. Preserve the existing 2.7 s failure deadline.
- [x] Preserve target reactivation, per-environment reset and landing event
  counting. Canceling PREPARE, before blend rises, returns directly to QUAD
  without inventing a landing attempt.

## Task 2: Reward handover

Files: `src/tasks/loco_pedipulation/mdp/rewards.py`,
`src/tasks/teacher_common/env_cfg.py`.

- [x] Set FR tracking activation to one throughout unified RETURN; use the
  original blend activation in other phases. Keep weight 2.5 and tolerance 5 cm.
- [x] Use the existing `1-blend` support weight for contact slip and impact
  penalties, posture and standing regularization during support recovery.
- [x] Add a teacher-only `return_contact` reward with weight +0.5, active for
  both moving and stationary commands. Its activation is
  `(1-blend)*smoothstep((elapsed-0.5*reach_time)/(0.5*reach_time))` in RETURN.
  This encourages late-descent contact without requesting immediate impact.
- [x] Override the FR four-leg gait target to stance during unified RETURN;
  keep the other feet's existing interpolation and resume the normal FR gait
  target after QUAD. Do not reset the shared oscillator.
- [x] Exclude FR from the generic 10 cm clearance penalty throughout RETURN,
  since it conflicts with the descending reference as support weight recovers.

## Task 3: Documentation and static review

Files: `src/tasks/teacher_common/README.md`, `tests/test_teacher_tasks.py`.

- [x] Document timing, exit/timeout rules, reward changes and the source task
  compatibility flag. Explain that old policies have compatible tensor shapes
  but were not trained with the new return behavior.
- [x] Review the diff and all RETURN/LOWER/RECOVER consumers; parse changed
  Python files for syntax and run `git diff --check`.
- [x] Update the existing source-reward-list expectation to include the new
  teacher-only `return_contact` entry; retain the source term comparisons.
- [x] Do not add or run tests, MuJoCo rollouts or training: current developer
  instructions require an explicit user request for tests. Report this limit
  and avoid claiming that policy behavior has been validated.

The user's instruction to execute the selected design authorizes proceeding
without another design approval or execution-choice prompt. Work in the current
`wsl-validation` checkout and preserve unrelated untracked transition work.
