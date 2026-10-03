# Stage0 stance implementation plan

Approved design: `outputs/actor_velocity_audit/report_and_proposal.md`, user approval “可以，执行吧”.

Goal: promote stage0 on independent stance-holding trials rather than strict stillness under exploration.

Architecture: task-local hold history and independent evaluation; runner logging hook schedules evaluations. Keep strict diagnostics and stage1/2 gates. Version checkpoints and offer explicit migration.

Tech stack: mjlab, PyTorch, rsl-rl5, WSL Ubuntu22.04 CUDA.

Global constraints: actor51/critic131 unchanged; physical initialization retained; no automatic cloud publishing; training branch wsl-validation.

- [x] Add tests in tests/test_leg_manip_stage0.py; run `python -m unittest tests.test_leg_manip_stage0 -v`, confirm nine failures for missing features/current gate.
- [x] Implement mdp/stance_hold.py (bounded contact history, mean speed magnitudes, pose and height), rl/stance_evaluation.py (fixed trial denominator, sticky failures, two-pass promotion, independent RNG). Add runner evaluation every500 iterations,32 trials per stance,1s settle,5s hold.
- [x] Fix mdp/events.py push write→forward; disable stage0 push initially, allow opt-in stable-only±.05 increments every10–15s, stage1 gradually restore full push. Preserve startup physics.
- [x] Disambiguate episode versus streaming Metrics/twist names; verify no collision.
- [x] Set init_std=.2, entropy=.001; implement real distribution bounds and stage1 gradual recovery; verify log_prob agrees with sampled distribution.
- [x] Version course checkpoint; persist last evaluation iteration; explicit migration clears optimizer/course and resets std while preserving network weights.
- [x] Replace automatic landing deadline with actual trajectory duration+timeout; regression checks for nominal1.8 versus manual.7.
- [x] Run targeted tests, full unittest discover, CUDA push regression, real old-model64-env evaluation, short PPO training with evaluation hook and save/load.
- [x] Review diff; deliver patch, commands, validation report. Do not launch15000-iteration training as part of verification.
