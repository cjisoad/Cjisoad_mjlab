# FSM GRU student distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans (recommended) to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add runnable BC and DAgger training for the 48D GRU student and push the verified implementation to the current remote branch.

**Architecture:** `src/tasks/fsm_student/distillation.py` owns sequence datasets, anchor loss, checkpoint metadata, and deterministic training loops. `scripts/train_fsm_student.py` owns CLI, BC training, and online DAgger rollouts through the existing full-chain teacher. The teacher action term gets an explicit opt-in student override while its default endpoint-teacher behavior remains unchanged.

**Tech Stack:** PyTorch GRU, NumPy NPZ shards, existing mjlab MuJoCo FSM teacher, pytest.

---

### Task 1: Dataset and training primitives

**Files:**
- Create: `src/tasks/fsm_student/distillation.py`
- Test: `tests/test_fsm_student_distillation.py`

- [x] Write tests for lazy window loading, final-anchor loss, checkpoint round trip, and finite gradient updates.
- [x] Implement `WindowDataset` for raw dataset directories and `ArrayWindowDataset` for DAgger replay arrays; reject invalid shapes, nonfinite values, cross-reset windows, and contract mismatches.
- [x] Implement `anchor_smooth_l1_loss` and `train_epoch`, with zero initial hidden state per sampled window and gradient clipping.
- [x] Implement checkpoint save/load with stage, optimizer, metrics, seed, and observation/action contracts while remaining compatible with `GRUStudent.load`.
- [x] Run focused synthetic tests.

### Task 2: Student execution switch in full-chain teacher

**Files:**
- Modify: `src/tasks/transition_t/keyboard.py:KeyboardTransitionAction.process_actions`
- Modify: `scripts/keyboard_transition_fsm.py:FullChainAction`
- Test: `tests/test_keyboard_full_chain_handoff.py`

- [x] Add `use_teacher_policy=True` as the default action-term mode.
- [x] When false, bypass endpoint teacher replacement and pass supplied common actions through the existing phase-gated limiter/delay path.
- [x] Keep transition FSM state updates, owner/reference behavior, and default teacher playback unchanged.
- [x] Add a lightweight unit regression that default mode still replaces endpoint actions and override mode preserves supplied actions.

### Task 3: BC and DAgger CLI

**Files:**
- Create: `scripts/train_fsm_student.py`
- Modify: `src/tasks/fsm_student/__init__.py`
- Test: `tests/test_fsm_student_distillation.py`

- [x] Implement `bc`, `dagger`, and `all` subcommands with positive argument validation and no silent output overwrite.
- [x] Implement BC loading balanced windows, training only anchor labels, periodic checkpoint/JSON metrics, and deterministic seed setup.
- [x] Implement frozen teacher label routing for QUADRUPED/BIPED endpoint actors and RISE/DOWN actors without adding R to observations.
- [x] Implement student rollout with scripted commands, hidden reset on physical done, finite replay collection, atomic replay save, and combined BC/replay fine-tuning.
- [x] Keep DAgger rollout controls bounded and synchronize global rise/down references across worlds.

### Task 4: Documentation and end-to-end verification

**Files:**
- Modify: `docs/fsm_gru_student.md`
- Create: `docs/fsm_student_distillation.md`
- Modify: `docs/superpowers/plans/2026-10-14-fsm-student-distillation.md`

- [x] Document BC/DAgger commands, loss timing, replay semantics, checkpoint contents, and limitations.
- [x] Run focused tests, existing student/recording tests, BC CPU smoke, and DAgger CPU smoke.
- [x] Inspect generated metrics/replay and run `git diff --check`.
- [x] Request code review, fix important findings, commit related files, and push `wsl-validation` to `origin`.
