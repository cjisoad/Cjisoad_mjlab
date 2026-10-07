# Robust Rise Guidance Implementation Plan

> Execute inline with superpowers:executing-plans and superpowers:test-driven-development. User already authorized implementation; keep existing uncommitted work in place with backup at /tmp/bridge_guidance_before_20261007.

**Goal:** Train the bridge only for robust reference-guided rising and a measured frozen-biped handoff.

**Architecture:** Separate nominal reference sampling, bridge command ownership and teacher queries. Append guidance observations to the bridge while retaining original expert observation/action conversion. Keep complete-attempt PPO collection and version the objective/observation contract.

**Tech Stack:** Python, PyTorch, MuJoCo, mjlab, rsl_rl, unittest.

## Task 1: Reference and control

- [x] Add behavioral tests in tests/test_bridge_guidance.py using real existing motion and bridge fixtures: independent clock, full preparation, endpoint hold, actual-state blend, relaxed boundary handoff, partial reset, and no physical state/history overwrite.
- [x] Run `PYTHONPATH=.:tests MPLCONFIGDIR=/tmp/bridge-mpl OMP_NUM_THREADS=1 python -m unittest test_bridge_guidance -v` and confirm expected failures.
- [x] Create src/tasks/pedipulation_bridge_t/guidance.py for validated torch reference interpolation of height, normalized gravity, support joints, anchor-relative FL, contact and unloading. Add command reference speed and blend/capture buffers, boundary-only handoff, independently sampled targets, bounded initial velocity impulses and configurable handoff gates in commands.py/actions.py.
- [x] Re-run reference/control tests and retain original event attribution tests with explicitly revised entrance expectations.

## Task 2: Observations, reward, persistence

- [x] Add tests for reward invariance to velocity commands, reference feature visibility, FR tolerance schedule, FL/reference support tracking, motion continuity, expanded source actor equivalence, and measured terminal credit.
- [x] Implement observations.py: original actor51 plus `[progress, actual_height, reference_height, gravity3, support_joints9, FL3, unload, FR_error3]`; critic shares original48 plus added22 and retains existing exact velocity/privileged terms.
- [x] Replace rewards in env_cfg.py/rewards.py with height/orientation main tracking, weak support_pose/FL/FR, reference-aware support/front-contact, actual terminal-potential increment and normalized torque. Preserve owner masking.
- [x] Expand source mean MLP first-layer input from51 to73 with zero guidance weights in rl.py; enforce actor73/critic117. Add optional non-discounted measured outcome to rollout.finish while retaining explicit legacy discounted mode for callers.
- [x] Include new protocol, complete command controls, observation names, reward weights/parameters and measured feedback semantics in checkpoint contract v2. Test changed controls/rewards and legacy checkpoint rejection.

## Task 3: Integration and evidence

- [x] Update old task tests, scripts/smoke_bridge.py, scripts/evaluate_bridge_biped.py and README for v2 dimensions and protocols.
- [x] Run `PYTHONPATH=.:tests MPLCONFIGDIR=/tmp/bridge-mpl OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_bridge*.py' -v`.
- [x] Run bounded actual CPU smoke `PYTHONPATH=. python scripts/smoke_bridge.py --device cpu --num-envs 4 --cycles 1 --ppo --output outputs/bridge_rise_guidance_cpu` and verify reference observability, actor update, frozen experts and checkpoint roundtrip.
- [x] Run bounded GPU smoke with 32 worlds if GPU is accessible; use the sandbox escalation only for existing GPU/desktop access restrictions.
- [x] Request focused independent review via superpowers:requesting-code-review, resolve significant findings, and re-run affected tests.
- [x] Record measured results and limitations; do not claim robust success from a smoke update, and do not start an unattended long training run.

## Execution evidence

Completed 2026-10-07. Details and limitations: [verification record](../robust-rise-guidance-verification-2026-10-07.md).

- Bridge suite: 62 tests, OK.
- Existing teacher/transition regression: 47 tests, OK (1 skipped).
- CPU 4 worlds: 943 bridge samples, 1 actual PPO update; actor changed, experts unchanged, checkpoint roundtrip and contract rejection passed.
- GPU 32 worlds: 6,148 bridge samples, 1 actual PPO update; same checks passed.
- Static frozen-biped endpoint baseline: 4/4 held for 2.5 s at zero initial velocity.
- Independent review findings were verified, corrected, and covered by regression tests.
- Initialized bridge smoke success: 0 on CPU and GPU. No long training run was started; robust rising performance remains to be learned and evaluated.
