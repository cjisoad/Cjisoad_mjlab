# Native teacher action distillation

**Goal:** Distill the four frozen actors' native joint-position requests, without a target governor on either control path, and correct scenario/contact evaluation.

**Architecture:** Keep the FSM's actor selection, references, physical state and common-default-angle conversion. Return native common-coordinate requests as labels. Both teacher and student use the same 12-dimensional, 20 ms zero-order-hold position action, with the existing finite-value validation and numerical action clamp. Do not change the standalone teacher training configurations or running server job.

**Constraints:** Work in the existing `/home/e980/projects/Cjisoad_mjlab` checkout on `wsl-validation`; no new worktree. Preserve 48 observations, 12 actions, the GRU architecture and DAgger episode selection. Version the changed label/execution contract so old filtered-label checkpoints cannot silently resume.

## Execution plan

- [x] Add regression tests: labels equal native requests for all four owners; teacher/student target execution is identical from the first substep and contains no governor state.
- [x] Run the tests and record expected failures before implementation.
- [x] Remove teacher preview/governor and teacher-only action caches; remove limiter-state restoration from entry resets. Keep native teacher checkpoint validation separate from live execution.
- [x] Add evaluation tests: cycle does not require absent tripod_operation; quad still does. A 0.1 s per-hind-foot contact grace tolerates short contact flicker but rejects sustained/no support and invalid posture. Keep continuous 0.5 s posture/support holds.
- [x] Run the evaluation tests before implementing the criteria helper; wire it into the real evaluation loop.
- [x] Update contracts, obsolete timing CLI modes and documentation; reject incompatible old action semantics on resume/deployment.
- [x] Run relevant CPU regression tests, a CUDA training/save/load smoke test, and teacher evaluation on quad/biped/cycle. Record behavioral regressions honestly; removing the governor also changes the transition teachers' original training conditions.
- [x] Review the final diff, record verification and commit the authorized changes locally without pushing or touching the server.

This plan supersedes the earlier unimplemented 48-output substep proposal. The user's 2026-10-20 instruction explicitly selects native-output supervision and no target limiter.

## Verification results

- Initial regressions: 12 expected failures, 1 existing behavior passed. Native labels, removed governor state and cycle scoring failed as expected before implementation.
- Final regression suite: 138 passed in 35.57 s. An earlier parallel run was killed by WSL OOM; rerun serially passed.
- Added a separate failing regression for clamped action history, then synchronized manager.action to the applied clipped action; the final suite includes this fix.
- CUDA: one teacher-controlled update and one student-controlled update; GRU weights changed, checkpoint resumed, native-action student exported and deployed independently for 8 steps on 2 environments.
- Read-only independent code review: no additional critical or important findings.
- Teacher CUDA evaluation: 4 episodes per scenario, validation split, seed 123. Quad: 4 successes, 0 physical failures, mean operation foot error 0.00910 m. Biped: all 4 maintained a biped hold of about 7 s, then physically failed before completing the return. Cycle: all 4 physically failed at about 7.2 s. These are behavioral regressions of unfiltered transition execution, not passing validation of full-chain performance.
- Standalone teacher configurations and running server job were not changed. A long native-output training run was not started.
