# V3 three-stage from-scratch implementation

Approved scope: three courses (50/50/0,40/20/40,20/20/60); scratch actor/critic/normalizers/Adam, no legacy checkpoint dependency. Level0 52/64 full-start successes twice; level1 48/64 independent holdout plus nominal52/64 twice; final level2 103/128 independent holdout acceptance. One second upright XY hold; natural stepping allowed. Nominal/small randomization,half,full physics. Preserve legacy training interface. Do not start formal training.

- [x] Write red tests for CLI scratch rejection/mutual exclusions, course evidence sources/counts/retention, actual randomization configuration, knee reward, terminal velocity relaxation.
- [x] Implement separate three-stage schedule, command/cfg factory and runner metadata.
- [x] Extend evaluator with explicit curriculum/source so full-start and real entries use proper stage physics; no phase-start promotion.
- [x] Extend trainer with scratch/three-stage/resume, stage rebuild preserving optimizer/lr/sampling and final acceptance; keep total update clock.
- [x] Add reviewed V3 training asset in repository and reproducible bank preparation/launch commands; no large checkpoint or state bank in git.
- [x] Verify CPU regression, scratch PPO and resume, all three stages and actual randomized model properties.
- [x] Prepare ClearML optional integration; verify offline training scalars, curriculum, evaluations, JSON artifacts and checkpoint archive.
- [ ] GPU smoke and online ClearML authentication/upload on server: deferred until user starts server. Do not connect/configure server yet.
- [x] Independent review and fixes: latest-policy acceptance, upload after reconstructed-stage save, PPO recipe checks, independent evaluation CLI, actual physics/Adam/normalizer/clock reconstruction and strict resume for all stages.
- [x] Prepare and verify focused commit via isolated clone (main git index read-only); push to wsl-validation. Server synchronization deferred until startup. No formal launch.
