# Robust rise guidance v2: implementation evidence

Date: 2026-10-07. Task: `pedipulation_bridge_t`.

## Implemented behavior

- Preserve the live source teacher state, including initial momentum and action history, when the command reaches the start boundary. Healthy state deviations are accepted at entry.
- Sample the entire nominal preparation, weight shift and rise with an independent clock. Randomize speed from 0.85 to 1.15; hold the endpoint after completion. Blend actual state into reference targets over 0.3 s without writing physical poses or joints.
- Track height and gravity direction strongly, with weak support-joint, FL and FR guidance. Remove linear/angular velocity command tracking rewards.
- Retain source momentum and optionally add one bounded initial velocity increment (50% probability; horizontal world components ±0.10 m/s and angular components ±0.20 rad/s).
- Require actual handoff state checks followed by 2.5 s of control by the frozen biped teacher. Default validation velocity command is zero. Apply measured +20/−20 outcome to the final bridge step without discounting the validation duration; bridge GAE still uses gamma 0.99.
- Extend actor inputs from 51 to 73 and critic inputs from 95 to 117. Source actor initialization preserves its original mean output through zero weights on new guidance channels. Keep both expert policies frozen.
- Version the checkpoint contract to v2, including observation channels, assets, expert/model hashes, rewards, controls, contact sensors and terminal-credit semantics. Reject legacy v1 checkpoints.

## Commands and results

| Check | Result | Evidence |
|---|---|---|
| Bridge test suite | 62 tests passed | `outputs/bridge_rise_guidance_tests.log` |
| Existing teacher/transition regression | 47 tests, OK; 1 skipped | `outputs/bridge_rise_guidance_teacher_regression.log` |
| CPU smoke, 4 worlds | 943 bridge samples; 1 PPO update | `outputs/bridge_rise_guidance_cpu/verification.json` |
| GPU smoke, 32 worlds | 6,148 bridge samples; 1 PPO update | `outputs/bridge_rise_guidance_gpu/verification.json` |
| Frozen biped static endpoint baseline, 4 worlds | 4/4 held for 2.5 s; FR RMS 0.006707 m | `outputs/bridge_rise_guidance_biped_baseline.json` |

Both CPU/GPU smoke checks report `actor_updated=true`, `frozen_experts_unchanged=true`, `checkpoint_roundtrip=true`, and `incompatible_contract_rejected=true`. Actor/critic dimensions are 73/117. The maximum measured initial linear-speed norm in the GPU sample was 0.214633 m/s.

Commands executed:

```bash
PYTHONPATH=.:tests MPLCONFIGDIR=/tmp/bridge-mpl OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_bridge*.py' -v
PYTHONPATH=.:tests MPLCONFIGDIR=/tmp/bridge-mpl OMP_NUM_THREADS=1 \
  python -m unittest test_teacher_tasks test_teacher_tasks_runtime \
  test_teacher_anchor test_transition_task test_transition_runtime -v
PYTHONPATH=. python scripts/smoke_bridge.py --device cpu --num-envs 4 \
  --cycles 1 --ppo --output outputs/bridge_rise_guidance_cpu
PYTHONPATH=. python scripts/smoke_bridge.py --device cuda:0 --num-envs 32 \
  --cycles 1 --ppo --output outputs/bridge_rise_guidance_gpu
PYTHONPATH=. python scripts/evaluate_bridge_biped.py --device cpu --num-envs 4 \
  --output outputs/bridge_rise_guidance_biped_baseline.json
```

## Independent review

A read-only reviewer identified two material issues: incomplete self-collision sensor coverage and moving/stationary labels that overstated static endpoint evaluation coverage. Both findings were checked against the runtime and corrected. Added regression checks also cover non-finite root velocities and sensor configuration in the checkpoint contract. Follow-up review reported no remaining significant findings.

## Interpretation and remaining work

The implementation and training pipeline are operational. The CPU/GPU smoke starts from source-teacher initialization and performs only one update; **both report zero successful rises**. This is not a trained robust-rising result.

The frozen-biped baseline initializes directly at four distinct static endpoint witnesses with zero physical velocity. Its 4/4 result establishes only that the sampled endpoints and validation protocol can work. It does not measure dynamic bridge handoff or recovery from initial momentum.

No long training run was started. Train a new v2 run, then evaluate complete source→bridge→biped attempts across initial velocity, FR error, path and command classes. Current reference/path assets and initial disturbance bounds do not establish recovery from arbitrary states. Old bridge v1 models are incompatible with the new observations and objective contract.

Changes remain in the existing workspace; earlier unrelated edits were preserved. Pre-implementation bridge/test/script backups are in `/tmp/bridge_guidance_before_20261007/`.
