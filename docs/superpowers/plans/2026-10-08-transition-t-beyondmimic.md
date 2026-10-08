# transition_t BeyondMimic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add an independent Go2 transition tracking task using the reviewed seven-second reference, BeyondMimic rewards and sampling, and physically realized base/foot reset randomization.

**Architecture:** Build `src/tasks/transition_t` as a narrow extension of `src/tasks/tracking`: validate and map the reference by names, subclass its command/config to handle state-preserving starts and endpoint holding, add reset-time foot IK and full-orientation termination, and register a 12-action Go2 task with a compact PPO MLP. Separate smoke and deterministic evaluation scripts will exercise the task without changing existing teacher or bridge code.

**Tech Stack:** Python 3.11, PyTorch, MuJoCo/mjlab 1.2, RSL-RL, NumPy, pytest.

---

## File map

- Create `src/tasks/transition_t/motion.py`: strict loader for the accepted NPZ schema, name-based joint/body reordering, SHA256 and interpolation helpers.
- Create `src/tasks/transition_t/initialization.py`: per-reset base perturbation helpers and constrained per-foot IK based on the repository's Go2 MuJoCo model.
- Create `src/tasks/transition_t/commands.py`: `MotionCommandCfg` subclass; reference loader integration, reset initialization, state-preserving reference entry point, one-shot endpoint hold, initialization metrics.
- Create `src/tasks/transition_t/terminations.py`: full quaternion-angle termination plus a reset-step grace period for tracking-error terms.
- Create `src/tasks/transition_t/env_cfg.py`: Go2-specific scene, actor/critic observations, action, rewards, events and terminations based on `make_tracking_env_cfg()`.
- Create `src/tasks/transition_t/rl.py`: compact PPO configuration and checkpoint-contract runner for this task.
- Create `src/tasks/transition_t/__init__.py`: register `Transition-Go2-v0` with train and play configs.
- Create `src/tasks/transition_t/README.md`: task contract, training, playback and evaluation commands.
- Create `tests/test_transition_t_motion.py`: data validation, name mapping and reference endpoint behavior.
- Create `tests/test_transition_t_initialization.py`: base/foot perturbation ranges, IK reachability, support clearance, limits and subset isolation.
- Create `tests/test_transition_t_task.py`: task registration, config dimensions, reference reset, end hold and current-state start contract.
- Create `scripts/smoke_transition_t.py`: one CPU/GPU environment reset/step and a small PPO update.
- Create `scripts/evaluate_transition_t.py`: fixed-start nominal/base/feet/combined evaluation and grouped metrics.

Existing `src/tasks/tracking` code is read-only for this feature. Its generic train and play scripts accept `MotionCommandCfg` subclasses, so they should require no edits.

## Task 1: Lock down the data contract and endpoint behavior

**Files:**
- Create: `tests/test_transition_t_motion.py`
- Create: `src/tasks/transition_t/motion.py`

- [x] **Step 1: Write failing tests for reference loading**

Test the accepted asset's SHA256, 50 Hz sample rate, 351 frames, canonical 12-joint mapping, named body mapping and finite arrays. Write temporary NPZ variants with shuffled names, duplicate names, missing names, wrong dimensions, NaN values and invalid quaternions; shuffled valid arrays must restore canonical order and each malformed asset must raise a specific `ValueError`.

- [x] **Step 2: Run the new tests and verify the expected missing-module failures**

Run: `MPLCONFIGDIR=/tmp/transition_t_mpl /home/eden/miniconda3/envs/mjlab/bin/python -m pytest tests/test_transition_t_motion.py -q`

Expected: collection fails because `src.tasks.transition_t.motion` does not exist yet.

- [x] **Step 3: Implement the strict loader and tests for interpolation/hold**

Implement `TransitionMotion` that loads only the accepted fields (`fps`, `joint_names`, `body_names`, `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, `foot_pos_w`, `contact`), verifies the 50 Hz/finite/shape/quaternion contract, reorders joints and bodies by names, stores SHA256, and provides time sampling that clamps at the final frame while returning zero reference velocities after completion.

- [x] **Step 4: Run the motion tests**

Run: `MPLCONFIGDIR=/tmp/transition_t_mpl /home/eden/miniconda3/envs/mjlab/bin/python -m pytest tests/test_transition_t_motion.py -q`

Expected: all loader, mapping, interpolation and endpoint-hold tests pass.

## Task 2: Implement physically constrained initialization

**Files:**
- Create: `tests/test_transition_t_initialization.py`
- Create: `src/tasks/transition_t/initialization.py`

- [x] **Step 1: Write failing tests for base and foot targets**

Use `src.tasks.pedipulation_transition_t.motion.make_model()` to test zero perturbation reproduces nominal feet, each reset sample respects configured base/foot bounds, support feet retain sole-ground height, and randomized foot targets produce joint states whose FK residual is within 2 mm and whose joints remain inside the 0.98-soft-limit interval (the reference exceeds the narrower 0.9 interval). Include a batch where an unreachable target is forced and assert that the documented retry/fallback counter changes without modifying another environment's state.

- [x] **Step 2: Run initialization tests and confirm missing implementation**

Run: `MPLCONFIGDIR=/tmp/transition_t_mpl /home/eden/miniconda3/envs/mjlab/bin/python -m pytest tests/test_transition_t_initialization.py -q`

Expected: import or missing-function failures for `src.tasks.transition_t.initialization`.

- [x] **Step 3: Implement base perturbations and bounded batched leg IK**

Sample configured base deltas with a separately sampled 25% undisturbed mask. Transform body-frame foot perturbations through the sampled root pose; preserve support-foot ground height and apply vertical offsets only to feet marked non-contact in the sampled reference frame. Implement damped least-squares leg IK with joint-limit clipping, finite iteration/retry bounds, per-environment fallback to the unperturbed reference, FK residual/ground checks, and fallback metrics. Keep this helper independent of manager internals so it can be tested with real MuJoCo FK.

- [x] **Step 4: Run initialization tests and the existing workspace tests**

Run: `MPLCONFIGDIR=/tmp/transition_t_mpl /home/eden/miniconda3/envs/mjlab/bin/python -m pytest tests/test_transition_t_initialization.py tests/test_leg_manip_workspace.py -q`

Expected: transition IK tests pass and existing workspace tests remain unchanged.

## Task 3: Wire task command and manager configuration

**Files:**
- Create: `tests/test_transition_t_task.py`
- Create: `src/tasks/transition_t/commands.py`
- Create: `src/tasks/transition_t/terminations.py`
- Create: `src/tasks/transition_t/env_cfg.py`
- Create: `src/tasks/transition_t/__init__.py`

- [x] **Step 1: Write failing tests for configuration and command contracts**

Assert registry lookup finds `Transition-Go2-v0`; the command uses the selected reference and Go2 joint/body order; actor and critic dimensions are the reviewed 75/192; action dimension is 12; reset receives configured base/foot initialization; post-reference command advances to the terminal frame without another reset; and `start_from_current_state(env_ids)` does not call root/joint state writers or alter velocity, reference alignment uses the current root horizontal translation/yaw, and a frame angle error near 90 degrees is detected by termination. Use real model reset where installed simulator support permits and narrow manager doubles for isolated contracts.

- [x] **Step 2: Run task tests and verify they fail on absent registration/modules**

Run: `MPLCONFIGDIR=/tmp/transition_t_mpl /home/eden/miniconda3/envs/mjlab/bin/python -m pytest tests/test_transition_t_task.py -q`

Expected: task import or registry lookup fails because the task package does not exist.

- [x] **Step 3: Implement the command subclass and task configuration**

Subclass existing `MotionCommandCfg` so generic train/play motion-file options continue to work. Validate robot/reference joint and body names before use; bind the re-ordered reference loader; call parent reset sampling for adaptive frame selection; apply randomized root/joint/velocity state plus foot IK to each reset subset. Override command advance to hold the final reference frame without resampling. Add `start_from_current_state(env_ids)` that only records a reference transform/time and never writes simulation state. Build actor (75) and critic (192) terms around the existing tracking observations, use Go2's base/feet names and nominal action scale/zero, preserve the six tracking rewards, install custom full-quaternion/grace-period terminations, and register separate training/play configs.

- [x] **Step 4: Run task tests plus tracking and teacher regressions**

Run: `MPLCONFIGDIR=/tmp/transition_t_mpl /home/eden/miniconda3/envs/mjlab/bin/python -m pytest tests/test_transition_t_task.py tests/test_bridge_env.py tests/test_bridge_tracking.py tests/test_pedipulation.py tests/test_loco_pedipulation.py -q`

Expected: new config/command contracts pass and existing teacher/bridge tests pass.

## Task 4: Add PPO runner and user-facing run/evaluation paths

**Files:**
- Create: `src/tasks/transition_t/rl.py`
- Create: `src/tasks/transition_t/README.md`
- Create: `scripts/smoke_transition_t.py`
- Create: `scripts/evaluate_transition_t.py`

- [x] **Step 1: Write failing tests for runner contract and evaluation grouping**

Test compact actor/critic dimensions, 12-action output, reference hash and order in checkpoint contract, rejection of an incompatible checkpoint contract, and deterministic evaluation producing separate nominal/base/feet/combined result groups with complete-episode success distinguished from endpoint sampling.

- [x] **Step 2: Run runner/evaluation tests and verify missing task interfaces**

Run: `MPLCONFIGDIR=/tmp/transition_t_mpl /home/eden/miniconda3/envs/mjlab/bin/python -m pytest tests/test_transition_t_task.py -q`

Expected: runner contract and evaluation entry-point expectations fail until implemented.

- [x] **Step 3: Implement compact PPO, smoke and evaluation tools**

Use actor/critic ELU MLP `[256, 128, 128]`, normalization, init std `0.5`, learning rate `3e-4`, adaptive KL, and otherwise the reviewed BeyondMimic PPO values. Save/validate reference SHA, body/joint layout, observation/action dimensions, action scale/zero, physical config and adaptive sampling state. The smoke script constructs one CPU environment, resets/steps it, then performs one PPO update and checkpoint save/load. The evaluator fixes complete-trajectory start, sets deterministic actor mean and runs four separately seeded perturbation groups, reporting full-motion completion, held endpoint criteria, errors, termination causes, IK fallback rate and realized initial perturbation distributions. Document exact invocations in the README.

- [x] **Step 4: Run focused transition tests**

Run: `MPLCONFIGDIR=/tmp/transition_t_mpl /home/eden/miniconda3/envs/mjlab/bin/python -m pytest tests/test_transition_t_*.py -q`

Expected: all transition task, data, IK, runner and evaluation tests pass.

## Task 5: Verify runtime, PPO update and requirements

**Files:** all `src/tasks/transition_t/` modules, `scripts/smoke_transition_t.py`, `scripts/evaluate_transition_t.py`, and the tests above.

- [x] **Step 1: Run the CPU end-to-end smoke**

Run: `MPLCONFIGDIR=/tmp/transition_t_mpl MUJOCO_GL=egl /home/eden/miniconda3/envs/mjlab/bin/python scripts/smoke_transition_t.py --device cpu --num-envs 2 --ppo-updates 1`

Expected: finite 75D actor/192D critic observations, finite rewards, one completed PPO update, and successful contract-checked checkpoint reload.

- [x] **Step 2: Run the full focused test set**

Run: `MPLCONFIGDIR=/tmp/transition_t_mpl MUJOCO_GL=egl /home/eden/miniconda3/envs/mjlab/bin/python -m pytest tests/test_transition_t_*.py tests/test_bridge_env.py tests/test_bridge_tracking.py tests/test_pedipulation.py tests/test_loco_pedipulation.py -q`

Expected: all focused/new and neighboring teacher/bridge tests pass.

- [x] **Step 3: Run the deterministic evaluation entry-point at smoke scale**

Run: `MPLCONFIGDIR=/tmp/transition_t_mpl MUJOCO_GL=egl /home/eden/miniconda3/envs/mjlab/bin/python scripts/evaluate_transition_t.py --agent zero --episodes-per-group 1 --device cpu`

Expected: four groups are reported with episode counts, completion, termination, errors and initialization fallback statistics; the output is explicitly a runtime smoke result, not a trained-policy robustness claim.

- [x] **Step 4: Inspect scope and whitespace**

Run: `git diff --check`

Expected: no whitespace errors and no modifications outside new transition task files, task-specific tests/scripts, and this implementation plan.

## Plan self-review

- Reference schema, by-name mapping, trajectory hold, base and foot randomization, IK constraints, current-state start, full quaternion termination, compact PPO, task registration, train/play compatibility, endpoint evaluation, and verification each map to a task above.
- No new configuration knob is introduced without a stated range or test; no inherited tracking source files or existing teacher files are changed.
- Runtime claims distinguish smoke validation from actual training results; the audited reference's dynamic imbalance remains a limitation to report.

## Implementation and verification record (2026-10-09)

Implemented in the existing working tree on `wsl-validation`; no commit or edits to the existing tracking/teacher/bridge implementations. New evaluator/runner tests are split into `test_transition_t_evaluation.py` and `test_transition_t_rl.py` for readability.

The task uses the approved local tracking implementation. Its config additionally subclasses the installed mjlab MotionCommandCfg so generic train/play recognize `--motion-file`. IK and robot soft limits use 0.98 because parts of this asset exceed the original 0.9 interval. Nominal action zero and .01/.02 armatures follow the source locomotion teacher. Ideal PD effort limits are retained; a hard joint velocity limit/motor speed curve is not introduced. These physical choices are documented in the task README and checkpoint contract.

Review fixes included: static obstacles cannot inherit floor penetration tolerance; 50 Hz frame boundaries are rounded within floating-point tolerance; exact final-frame sampling is explicit; automatic reset does not advance reference time; explicit reset builds relative targets from the root pose just written, before derived MuJoCo state is refreshed; evaluator numerical failures serialize as null with an explicit flag; RSL-RL inference-created normalizer buffers are made writable before checkpoint restoration.

Validation runs use `/home/eden/miniconda3/envs/mjlab/bin/python`. This environment has no pytest installation, so `/tmp/run_transition_pytest.py` imports the system pytest with its legacy `py` module compatibility shim. No dependencies were installed or changed. The bounded CPU smoke exercises reset/step, actual 75/192/12 dimensions, state-preserving entry, fresh relative targets, PPO update, checkpoint roundtrip and incompatible action rejection. The four-group evaluator was exercised both with zero actions and the one-update smoke checkpoint. Neither is a trained robustness result.

A deterministic nominal-model reset audit requested combined perturbations at all 351 reference frames (seed 42, no intentionally undisturbed samples): 207 full-scale, 37 half-scale, 23 quarter-scale, 84 reference fallbacks (23.93%). This is a single initialization-distribution diagnostic, not policy success evidence. Detailed record: `outputs/transition_t_smoke/reset_distribution.json`.

Final verification output and artifacts are recorded below after the last review pass. GPU execution, converged tracking, learned robustness and closed-loop teacher handoff remain unverified.

Final results:

- Focused new-task tests plus teacher, bridge, pedipulation, loco and workspace regressions: **140 passed**, 2 pre-existing environment warnings (Matplotlib Axes3D and Torch JIT deprecation).
- Independent reviewer re-ran motion/evaluation/command tests: **39 passed**; no remaining important findings after fixes.
- CPU smoke: **passed**, 2 environments, 16 rollout steps, 1 PPO update; actor75/critic192/action12, finite rollout, fresh reset targets, state-preserving start, weight update, checkpoint roundtrip and incompatible action rejection. Artifact: `outputs/transition_t_smoke/verification.json`.
- Generic `scripts/train.py Transition-Go2-v0` entry with `--gpu-ids None`, 2 environments and 1 iteration: **passed**, saved `logs/rsl_rl/transition_t/2026-10-09_00-15-22_transition_t_entry_smoke/model_0.pt`. CLI help parsing also passed.
- Zero-policy and one-update-checkpoint four-group evaluations: **passed as runtime checks**, all four 1-episode groups reported failure to complete the motion, as expected for untrained policies. Artifacts: `outputs/transition_t_evaluation.json` and `outputs/transition_t_smoke/policy_evaluation.json`.
- Python compilation and `git diff --check`: **passed**. Existing `src/tasks/tracking/` has no changes. Files remain in the working tree; no commit/push/merge or long training run.
