# Transition Down V2 adaptive training implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; the user has approved implementation, push and remote training.

**Goal:** Train the reviewed 3.7 s landing reference with measured knee protection and performance-driven curriculum, without fixed nominal/entry update counts.
**Architecture:** Add an independent transition_down_v2 package, preserving V1 data/checkpoints and unrelated dirty forward-task files. A single entry-capable PPO runner starts with reference resets and promotes into teacher-entry levels; its optimizer and learning clock remain continuous. Physical-substep safety metrics feed reward/termination, validation and final acceptance.
**Tech Stack:** Python, MuJoCo/mujoco-warp, PyTorch, mjlab, rsl-rl.

Approved relaxation: nominal rate .90, entry .85, two consecutive evaluations; final .90, nominal .95. Keep 40mm evaluation knee margin, train soft target50mm and hard termination25mm. Endpoint tolerances remain V1 (.25rad/.05m/.3m/s/.8rad/s); uninterrupted1s. FL contact must persist60ms by2.4s. Final zero observed nonfoot/FR contacts. Configurable evaluation interval250, budget40000 total updates only.

- [x] Test pure curriculum with good reports, failures, missing evidence, two windows, budget/resume recipe and sealed split isolation; observe failures before implementation.
- [x] Implement gates.py with JSON-serializable config and scheduler. A nominal gate requires standard100 and +/- .03m/s32 each; entry gate64 distinct validation trajectories plus nominal regression100.
- [x] Package exact approved V2 NPZ and regenerate a compatible entry bank from verified original raw states: recompute physical speed and FR error with stable teacher heading; keep raw execution histories intact; validate512 train/64 validation/64 final trajectory separation. Save source hashes and observed coverage.
- [x] Implement versioned commands/env/runner with actor75/critic192/action12, original PPO/action limiter/randomization, knee shaping, strict checkpoint contract and one continuous optimizer.
- [x] Implement all-leg mesh geometry audit and 5ms integration hook; verify native FK parity, latch unsafe intermediate samples through termination capture, clear only reset worlds.
- [x] Implement continuous one-second V2 evaluation with deadline, terminal force/pose, no reset credit and explicit failed records. Validation never reads final states.
- [x] Implement chunked training CLI: evaluate/advance/save at intervals, resumable scheduler and optimizer, explicit budget_exhausted status, candidate checkpoint chosen by pass evidence. Full acceptance tests use sealed final entries, four perturbation groups, augmentation and fresh teacher runs.
- [x] Replace remote launcher with hash-verified GPU smoke then adaptive V2 CLI; no fixed model filenames or 30001/10000 phases. Provide config and MuJoCo/local smoke commands in README.
- [ ] Run focused unit/integration tests plus real CPU PPO, evaluation and strict resume; review diff, merge only task files into current branch, verify then commit/push. Sync exact release/assets to remote42311, GPU smoke, start train and verify actual updates/checkpoints.

Verification:
```bash
PYTHONPATH=/tmp/transition_down_test_deps:. OMP_NUM_THREADS=1 MUJOCO_GL=egl /home/eden/miniconda3/envs/mjlab/bin/python -m pytest -q -p no:cacheprovider tests/test_transition_down_v2*.py
```
Critical assertions include no advancement on unsafe/duplicate/insufficient trials, no early stop on reward alone, no optimizer reset on stage advancement, V1 checkpoint rejection, physical-substep sampling, and final split isolation.
