# FSM student warm-start and full-frame loss

**Goal:** Implement the user-approved teacher warm-start and full-frame GRU supervision, including short failed episodes.

**Design:** Keep the existing 48D input, recurrent policy, FR phase gate and action scale. Teacher controls the first configurable second after each physical reset while the student updates hidden. A per-environment teacher mask preserves the existing endpoint teacher action path. BC supplements its indexed windows with episode prefixes; replay builds prefixes even when no episode reaches 32 frames. Right padding is excluded by a boolean mask. All real frames receive Smooth L1 supervision; inverse per-frame category frequency replaces anchor-only balancing. Physical resets restart warm-up and memory; ordinary FSM handoffs do not.

**Execution:** Inline in the current workspace, preserving existing uncommitted work. No remote retraining is required by this implementation request.

- [x] Add failing regression tests in `tests/test_fsm_student_distillation.py` for short episodes, reset boundaries, prefix supervision, padding, category weighting and per-environment warm-start timing.
- [x] Implement padded sequence items, prefix windows, masked full-frame loss and frame weighting in `src/tasks/fsm_student/distillation.py`; keep the old anchor helper for legacy analysis.
- [x] Add per-environment endpoint teacher selection in `src/tasks/transition_t/keyboard.py`, and test mixed teacher/student control without changing the FR governor.
- [x] Add CLI warm-start duration, teacher-control replay metadata, short-replay checks and full-frame metrics in `scripts/train_fsm_student.py`.
- [x] Add optional keyboard warm-start in `scripts/keyboard_transition_fsm.py` with hidden continuity and reset handling.
- [x] Update `docs/fsm_student_distillation.md` and `docs/fsm_gru_student.md`; explain that old checkpoint weights are unchanged and need retraining for the new loss.
- [x] Run targeted regression tests, existing FSM tests and small real-data BC/DAgger smoke runs. Reindex the old biped replay to verify its formerly discarded valid frames now generate training windows.


## Verification results

- Related regression suite: **105 passed, 3 subtests passed**; one existing TorchScript deprecation warning.
- Real-data CPU BC smoke: 160 training / 32 validation windows, 2 epochs. Train loss 0.627943 → 0.593411; validation 0.551427 → 0.517648. Artifacts: `/tmp/fsm_full_frame_smoke/bc.pt` and `.metrics.json`.
- Real CPU DAgger smoke: 2 worlds × 80 steps, 0.2-second warm-start. Exactly 20 teacher-controlled frames and 140 student-controlled frames; 160 valid replay windows; one training epoch completed at loss 0.576202. Verified replay mask equals `episode_steps < 10`. Artifacts: `/tmp/fsm_full_frame_smoke/dagger.pt`, `.metrics.json`, `.replay/dagger_01.npz`.
- Old biped replay `outputs/fsm_student_analysis_20261015/results/student.replay/dagger_02.npz`: **22,268 windows**, history lengths 1–29 frames, all checked for valid/nonterminal frames and reset boundaries. Previously none reached the required 32 frames. Anchor categories: biped_walk 2,304; biped_manipulation 12,357; down 7,607.
- These smoke runs verify the training/control pipeline. They do not establish full-task student stability; formal retraining and longer closed-loop evaluation remain separate work.
