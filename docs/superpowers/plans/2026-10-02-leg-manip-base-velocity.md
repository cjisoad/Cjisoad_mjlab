# Leg Manip Base Velocity Implementation Plan

> **For agentic workers:** Execute inline using executing-plans; use requesting-code-review for the completed change. Track each task below.

**Goal:** Add consistent noisy actor / exact critic anchor base velocity and train for measured stage 3 promotion.

**Architecture:** Append actor velocity at 48:51; reuse and scale critic's existing velocity slot. Start a separate experiment with random actor/critic initialization and fresh optimizer, iteration and stage-0 curriculum state. Checkpoint migration is retained only as an optional compatibility tool.

**Tech Stack:** PyTorch, mjlab, rsl_rl, unittest, TensorBoard, CUDA.

- [x] Add `tests/test_leg_manip_velocity.py`: rear-up/yaw transforms, bounded additive noise, shared first 48 channels, no mode gating, mirror involution, checkpoint output equivalence and optimizer reset. Update existing integration / PPO tests to expect actor 51 and critic 131. Run these tests before implementation and inspect expected failures.
- [x] Implement task-local observation functions in `src/tasks/leg_manip/mdp/observations.py`: `v = to_anchor(term.basis_w, root_link_lin_vel_w) * 2`; concatenate actor old48 with `v + U(-.04,.04)` when noisy; return shared sample `[:, :48]` to critic. Add explicit constants for dimensions, scale and physical noise amplitude.
- [x] Implement `src/tasks/leg_manip/rl/symmetry.py`: original mirror on `obs[..., :48]`, then append `obs[..., 48:51] * (1,-1,1)`. Add an overridable symmetry loss to `src/tasks/pedipulation/rl/ppo.py`; set LegManipPPO's loss to the task-local version and validate 51/131.
- [x] Add `src/tasks/leg_manip/rl/checkpoint.py` and `scripts/migrate_leg_manip_velocity_checkpoint.py`: append zero actor columns, divide critic velocity columns by 2, clear optimizer state, reset only curriculum statistics/age, retain iteration and stage. Validate old 48/131 widths and refuse output overwrite. Tag layout in migration and runner saves. Run the new tests green.
- [x] Update `doc/LEG_MANIP.md` and keyboard checkpoint help. Run all tests headless with `DISPLAY='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/eden/miniconda3/envs/mjlab/bin/python -m unittest discover -s tests`; inspect exit status. Review diff and request independent review.
- [x] Migrate `outputs/wandb_rngzvn3h/model_14999.pt` into `logs/rsl_rl/leg_manip/velocity_warmstart/model_14999.pt`. Verify a real PPO update/save/load/export and GPU smoke before full training. Keep source snapshot and launch parameters with run artifacts.
- [x] Launch a fresh 4096-env experiment on the 4090 server, seed 42, resume=False, 15000 updates, save_interval=100, W&B logger and upload_model=True. Verified random initialization, initial stage 0 and saved actor 51D / critic 131D checkpoint. Run: `rc8qgs8f`; full launch argv and paths in `outputs/leg_manip_velocity_20261002/launch.json`.
- [ ] After training, evaluate stage promotion and count-weighted tracking metrics; record whether and when stage 3 was reached. Training is currently running.

Verification: targeted 24 tests passed; full suite 189 tests, 3 skips, no failures. Real CPU PPO update/reload/51D ONNX export passed. GPU checks passed with 1024 environments (5 updates, 6.21 s) and 4096 environments (3 updates, 8.40 s). Independent review found no critical/important issues. Artifacts: `outputs/leg_manip_velocity_20261002/validation_report.json`. Formal fresh training is now running on the 4090 server under `/pedipulation/legmanip1/anchor_velocity_scratch_20261002_185344`. Remote suite: 162 tests, 4 skips, no failures; missing legacy stand reference source accounts for the reduced count. Remote fresh GPU PPO and first checkpoint verification passed. W&B credentials reused successfully; run `rc8qgs8f`. Every saved checkpoint is retained and uploaded.
