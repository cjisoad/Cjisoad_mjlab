# Sustained standing reward: verified 10,000-update continuation

The user approved adding the bounded 0.1-weight holding reward, extending
episodes to 10 s, pushing the changes, and warm starting a new 10,000-update run
from the original transition_t/model_30000.pt. The completed entry model_4999.pt
was not used as initialization.

Source: reward implementation `96356a1`; final deployed source
`e079bba28e846e58e08d9ba930a2355978284209` includes deployment symlink validation.
Both are pushed on `origin/wsl-validation`.

## Behavior and verification

After reference completion, a currently standing sample receives raw reward
`0.2 + 0.8 * min(credited_standing_seconds / 1, 1)`. Weight is 0.1; rewards scale
by dt, so each 20 ms step earns at most 0.002. The first two consecutive bad
samples earn zero and pause credited time; the third clears it. Physical,
tracking and nonfinite failures clear immediately. This uses the handoff gates,
without the separate 6 cm world-foot RMS diagnostic. Original FR target limits
remain 2 rad/s and 30 rad/s².

Reference duration is 7 s. Ten-second episodes allow about 3 s of terminal
practice from full-start entries. Entry recipe v6 fingerprints rewards, dt
scaling, episode duration and holding policy; strict resume rejects changed
objectives. Reward-manager resets are per-world and repeated same-step calls do
not add credit twice.

Validation: 115 tests and 15 subtests passed. Independent review found no
critical/important issues and reran all 7 reward tests. CPU 8-world rollout,
two PPO updates and checkpoint round trip passed. All 1,016 deployed Git blobs,
including symbolic link target text, matched their SHA256. GPU 8-world warm
start from the original seed completed two PPO updates; actor max weight change
was 0.0010482864 and networks were finite.

## Formal run

Server: `xj-member.bitahub.com:42242`.

Base directory:
`/pedipulation/legmanip1/transition_hold_10000_20261009_e079bba`

Training directory: `run/training`; log: `run/training.log`; courses:
`run/training/course.jsonl`; TensorBoard events and models share the training
directory. Final planned model is `model_9999.pt`; final four-level evaluation
is `run/final_evaluation.json`.

Original seed:
`/pedipulation/legmanip1/transition_entry_fr_20261009_66effff/repo/logs/rsl_rl/transition_t/2026-10-09_01-13-48_transition_t_direct_4090_20261009/model_30000.pt`

Seed SHA256:
`91b6323b8cae9eb1f105d62e68e540be7ae4dea56fbf782386438b95339df4bf`

Validated entry bank:
`/pedipulation/legmanip1/transition_entry_fr_20261009_66effff/run/entries.npz`

Bank SHA256:
`60fb91f2e92b5f099e0201977394c56aa9cf1a9542ddff2a2633332cb1730dd9`

The run retains actor/critic/normalizers and policy distribution, initializes a
fresh optimizer and course 0, and uses 4,096 worlds, cuda:0, 24 steps per world
per PPO update, seed 42. Evaluations run every 250 updates on 64 distinct held-out
trajectories; promotion needs >=80% in two consecutive windows. This is a new
course clock with 10,000 additional updates, not a strict optimizer resume.
Checkpoints use that new clock, so the final name is model_9999.pt.

Launched at 2026-10-09 23:36:13 CST in an independent session; orchestrator PID
11342, trainer PID 11354. SSH disconnection does not terminate it. Baseline and
final evaluations cover all four levels; the validated existing bank is reused.

## Startup snapshot at 2026-10-10 00:04 CST

Latest observed checkpoint: model_1300.pt. Full verification of model_1249.pt
confirmed 1,250 updates, original seed SHA, recipe v6, finite networks and finite
TensorBoard scalars. Actor max weight difference from seed: 0.04121187. At
TensorBoard step 1,265, Episode_Reward/standing_hold was 0.00961090 (episodic
contribution normalized by episode duration), mean reward 44.02317, mean episode
length 476.33 steps. GPU: 4,042 MiB and 59% utilization. Both processes alive.

| Updates | Full reference completions | Standing successes |
|---|---:|---:|
| Baseline | 50/64 | 37/64 |
| 250 | — | 17/64 |
| 500 | — | 23/64 |
| 750 | 53/64 | 36/64 |
| 1000 | 53/64 | 33/64 |
| 1250 | 52/64 | 39/64 |

Course remains 0. Latest 25 failures: 12 foot-height tracking failures and 13
standing-hold timeouts. Strict endpoint successes remain 0/64. These early
results do not establish a reliable improvement; holding reward is active and
formal training is progressing. Existing repeatability variation in simulation
standing outcomes remains unresolved. Evaluation measures handoff eligibility,
not execution of the complete biped-teacher continuation.

Local artifacts: `outputs/transition_standing_reward_20261009/` contains test,
CPU/GPU smoke, deployment, launch, recipe, checkpoint and curriculum verification
records. Previous training outputs are preserved.
