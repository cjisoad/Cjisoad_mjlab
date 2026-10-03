# Dense actor MoE implementation plan

Goal: Train leg_manip from scratch with four learned dense actor experts and the unchanged MLP critic.

Architecture: Reuse the unmodified GatedMoENet and BaseMoENet from the local moe-rsl-rl repository at ebd4bcffc5a98b152611dbef401ae1a5ccd6d21f. Vendor only these network modules and their BSD license to keep the existing RSL-RL 5.0.1 PPO, action distribution, symmetry objective and exploration schedule. Register leg_manip_moe alongside the legacy task.

Constraints: use_explicit_expert=False, top_k=-1 (dense), num_experts=4, independent [512,256,128] experts, [128] gate. No Gaussian mixture, auxiliary routing losses, reward, observation, curriculum or optimizer changes. Actor only; critic MLP. Fresh server run: no checkpoint, stage0, 4096 environments, 15000 iterations, seed42, save100, diagnostics20/512.

- [x] Add failing tests for learned dense routing, all expert/router gradients, unchanged Gaussian/std, config/legacy compatibility, checkpoint playback, routing diagnostics and observational equivalence of PPO updates.
- [x] Add vendor network source with exact provenance and license; implement the MLPModel-compatible DenseMoEActor and actor-only task config.
- [x] Extend sampled diagnostics to log weights per command group and policy/total gradient cosines for router and each expert; restore router caches after diagnostics.
- [x] Auto-select MoE config for keyboard playback by checkpoint actor parameter keys, leaving legacy playback intact.
- [x] Run new tests, existing 152 regressions, and actual CUDA environment/PPO/save/reload smoke. Request independent code review and resolve material issues.
- [ ] Commit and push wsl-validation; verify remote commit. Sync server and test there before stopping the existing training process.
- [ ] Stop only the identified trainer, retain its artifacts, start an independent fresh leg_manip_moe run, verify finite updates/new W&B run/stage0/zero-start metadata, report observed runtime estimate.

Local evidence: 157 tests passed; two CUDA PPO updates using five epochs/four minibatches; all experts/router changed with finite parameters/losses; exact checkpoint roundtrip. Keyboard smoke passed 550 steps including command switching/reset. Independent read-only review found no material bug, verified exact vendor source/license and actual TorchScript compilation/inference equivalence. Enabled/disabled sampled diagnostics produced identical PPO updates. Actor parameters: 776770; original critic parameters: 231937.
