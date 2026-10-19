# legmanip_distil implementation

Approved design: autonomous 48D proprioception + canonical user intent, GRU(256)
and ELU MLP(256,128), deterministic 12D applied joint target action at 50 Hz.
R means six zero commands. Deployment contains no teacher or motion reference.

- [x] Register the independent task, direct PD action and per-environment commands.
- [x] Implement the frozen four-actor FSM with independent rise/down reference streams.
- [x] Extend RSL-RL 5.0.1 Distillation/DistillationRunner with continuous sequence L1,
  episode teacher/student mixing, persistence and standalone student export.
- [x] Add standalone playback, evaluation and the teacher timing comparison.
- [x] Verify observations/actions, teacher parity, asynchronous resets, learning,
  checkpoint/resume and standalone inference; run CPU smoke (GPU if available).
- [x] Review implementation and publish reproducible commands and measured limitations.

Validation results: 27 RL tests, 10 runtime physics tests, 6 source-parity tests,
and 3 deployment tests (46 collected tests) passed on CPU. A real two-environment run completed two
updates at about 55--62 steps/s and wrote `model_1.pt`, `model_2.pt`, and
`student.pt`; a resume run continued at completed update 2. Standalone playback
ran 100 steps without a teacher. A two-episode-per-scenario evaluator completed
without physical failures; the untrained smoke student correctly recorded zero
successes and zero script completions. No CUDA device was available.

Formal defaults: 256 environments, 32 steps, 60000 iterations, 8 epochs, 2
environment minibatches, AdamW 3e-4, gradient clipping 1, teacher-only first 1000
iterations then beta=(1-1/2000)^max(k-1000,0). TensorBoard, save every 500 updates.
Task sampling: 25% quadruped/tripod, 25% biped/down, 50% complete cycles. Use
training entry trajectories for resets, separate validation/final trajectories.
The first delivery is implementation plus short validation, not a full training run.

Teacher source is the current workspace's scripts/keyboard_transition_fsm.py and
mark/{quad,biped,up,down}_teacher.pt. Preserve existing uncommitted work. No changes
to installed RSL-RL/mjlab or the parkour reference checkout.
