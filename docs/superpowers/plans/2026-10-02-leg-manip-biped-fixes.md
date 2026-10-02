# leg_manip Biped Fixes Implementation Plan

> Execute inline in this session: the user authorized the two reviewed fixes.

**Goal:** Make biped reward gates independent across environments and allow bounded, safe biped exits.

**Architecture:** Retain the command latch and reward ramp. Add per-environment episode exit state,
which only relaxes pitch termination during a bounded descent; use per-environment height gates.

**Tech Stack:** Python, PyTorch, mjlab, unittest.

## Task 1: Reward gate

- [x] Add `test_height_gate_is_independent_in_mixed_batch` and compare the successful environment
  before/after changing neighboring heights. Assert `height_score.shape == (num_envs,)`.
- [x] Run `python -m unittest tests.test_leg_manip_rewards -v`; observe expected regression failure.
- [x] In `src/tasks/leg_manip/mdp/rewards.py`, replace `score.mean()` with `score` and update
  reward-order documentation in `env_cfg.py` and `doc/LEG_MANIP.md`.
- [x] Re-run reward tests and confirm success.

## Task 2: Exit protection

- [x] Add tests in `tests/test_leg_manip_rewards.py` and `tests/test_leg_manip_commands.py`
  for canceled high targets, low-target exits, expiry, lateral and excessive total tilt,
  recovered posture with front contact, re-entry and partial reset.
- [x] Run these two modules and inspect expected failures before implementation.
- [x] Add validated positive finite `biped_exit_timeout=2.0` and
  `biped_exit_recovery_angle=radians(50)` to `LegManipCommandCfg`; add active/timer buffers.
- [x] Start timer on a falling mode edge; advance only active exits, clear at expiry/re-entry/reset,
  or after weight zero + tilt below recovery angle + front contact.
- [x] In `terminations.fell_over`, apply lateral and 110° total tilt limits during active exits;
  retain existing LOCO and BIPED rules outside exits.
- [x] Re-run command/reward modules and correct any regression.

## Task 3: Integration and reporting

- [x] Run `python -m unittest discover -s tests -p 'test_leg_manip*.py' -v` including CPU integration.
- [x] Load the original rngzvn3h checkpoint in a two-environment CPU runner and verify inference,
  unchanged 48/131 dimensions and zero transient exit state after reset.
- [x] Run `git diff --check`; inspect the complete diff.
- [x] Explain tested behavior, finite grace parameters, and proposed velocity/noise/ABI changes.

## Validation result

Targeted regression tests: 33 passed. leg_manip suite including CPU integration: 66 passed.
Full headless repository suite: 181 tests, OK (3 X11 tests skipped because no reachable display).
Original checkpoint restored actor/critic/optimizer and curriculum stage 2; eight CPU inference steps
were finite with actor 48D / critic 131D. Independent review found no blocking issues.
`git diff --check` passed. These checks validate logic and compatibility, not post-training biped success.
