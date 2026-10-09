# Sustained standing reward and 10,000-update continuation

**Goal:** Implement the approved bounded 0.1-weight holding reward, extend episodes to 10 s, and start a new course run from the original model_30000.pt.

**Architecture:** Move the existing standing metrics/gates to `standing.py` for reuse without a keyboard/env-config import cycle. A stateful mjlab reward term owns per-world holding duration, bad-sample streaks, and a same-step guard; reward-manager resets clear selected worlds. Entry recipe v6 fingerprints the full reward configuration, dt scaling, and episode length.

**Tech stack:** PyTorch, mjlab 1.2.0, RSL-RL 5.0.1, unittest, SSH, RTX 4090.

- [ ] Add `tests/test_transition_t_standing_reward.py` testing pre-completion zero, ramp/cap, zero credit on bad samples, third-bad reset, physical/nonfinite/terminated failure, per-world reset, same-step idempotence, and no 6 cm foot gate. For example:

```python
for _ in range(25):
    env.common_step_counter += 1
    value = reward(env)
assert abs(value.item() - 0.6) < 1e-6  # raw; manager applies 0.1 * dt
```

- [ ] Add a real 2-world CPU reward-manager integration test confirming `standing_hold` is registered at weight 0.1, episode length is 10 s, reset is per-world, and the maximum weighted contribution is 0.002 at 50 Hz.
- [ ] Add recipe tests changing reward weight, episode length, and reward dt scaling; strict resume must reject these before loading saved weights.
- [ ] Run the new unittest tests and confirm missing-feature failures.
- [ ] Implement `SustainedStandingReward` in `rewards.py`, using the shared timer and current reference completion, with `raw = good * (0.2 + 0.8 * clamp(duration / 1, max=1))`. Register at 0.1 and change episode length to 10 s.
- [ ] Record objective in entry recipe v6; update approved design/README; run related suites, CPU PPO smoke, and independent review.
- [ ] Commit, integrate into wsl-validation, push, and deploy a hash-verified committed source snapshot in a fresh remote directory. Reuse the validated entries.npz and original checkpoint by explicit paths and SHA verification.
- [ ] Run 8-world GPU PPO/checkpoint smoke, then launch 4,096-world 10,000-update warm start with 250-update/64-trajectory evaluations and final four-level evaluation. Preserve previous training and outputs.
- [ ] Verify live processes, exact seed SHA, run recipe, checkpoint finite parameters, and a nonzero actor weight change before reporting training started. Record report and server paths.
