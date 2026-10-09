# Transition Target Limiter Implementation Plan

**Goal:** Bound transition joint-position target velocity and acceleration in
training and playback, preserve entry state and checkpoint contracts, then push.

**Architecture:** A batched Torch target governor is reused by the common
Pedipulation position action. Transition alone enables per-joint limits; keyboard
teacher selection and control delay feed the same target governor. Entry-bank
snapshots persist its state and checkpoint loading verifies its configuration.

**Tech Stack:** Python, PyTorch, mjlab/MuJoCo Warp, RSL-RL, pytest, Git/SSH.

- [x] Add failing tests in `tests/test_joint_target_limiter.py` for per-joint
  speed/acceleration bounds, reversal, settling and partial state restoration;
  run them before implementing `src/tasks/pedipulation/mdp/target_limiter.py`.
- [x] Add failing action integration tests; implement opt-in config and deferred
  reset in `src/tasks/pedipulation/mdp/actions.py`, call the common write-target
  helper from `src/tasks/transition_t/keyboard.py`, enable matching limits in
  `src/tasks/transition_t/env_cfg.py`, and check executed-action histories.
- [x] Add failing entry snapshot/restore/metadata tests; persist limiter state in
  `scripts/collect_transition_entries.py`, validate in `entry_bank.py` and restore
  in `entry_course.py`. Record and verify its recipe in `entry_rl.py`.
- [x] Add failing strict contract/migration tests; update `transition_t/rl.py`,
  expose explicit legacy-limiter testing in `scripts/keyboard_teacher.py` and
  `scripts/keyboard_transition_fsm.py`. Keep old-model ordinary loading strict.
- [x] Run focused regressions with
  `PYTHONPATH=.:/tmp/keyboard-transition-test-deps MPLCONFIGDIR=/tmp/transition_t_mpl
  XDG_CACHE_HOME=/tmp/transition_t_cache WARP_CACHE_PATH=/tmp/transition_t_review_warp
  /home/eden/miniconda3/envs/mjlab/bin/python -m pytest -q` on the limiter,
  transition_t, entry and related teacher action tests.
- [x] Collect a new bounded CPU bank and perform real legacy warm-start PPO,
  checkpoint reload and bounded playback checks; document bounds, observations,
  limitations and legacy package supersession in the transition README/report.
- [x] Review the complete selected diff, run `git diff --check`, stage only
  transition-related files and required keyboard dependencies, commit and push
  `HEAD:refs/heads/wsl-validation` to origin; verify the remote commit hash.

Review also fixed nonfinite action scale, mismatched bank action affine maps,
previous-request delay semantics and reset-distribution resume provenance.
Final defaults limit only FR to 2/30; support joints are unrestricted after the
64-world compatibility diagnostic. Local verification: 179 tests / 32 subtests,
real CPU collection, two PPO updates plus strict resume to three, and 2444
bounded physics substeps in legacy keyboard playback. Playback timed out and
does not establish improved success. New server port 42242 is authorized for
formal deployment and training after the commit is verified. Commit 66effff
was pushed and deployed with all 1016 content hashes verified. CUDA smoke
passed; official collection produced 99612 states / 4043 trajectories, then
PID 1509 started the 4096-world 5000-update warm start. Checkpoints through
model_600 were observed; level0 standing is 35/64 baseline, 20/64 after250,
17/64 after500, so the course remains0. The long run continues remotely.
