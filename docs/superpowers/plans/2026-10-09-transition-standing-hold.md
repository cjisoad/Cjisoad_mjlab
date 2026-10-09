# Transition standing hold implementation plan

> Execute inline with TDD; reward implementation and new training are outside the approved timer change.

**Goal:** Tolerate two consecutive bad standing samples, pausing credit, and reset on the third.

**Architecture:** A shared tensor timer in `standing.py` serves independent entry trials and keyboard handoff. Training recipes and evaluation JSON record the revised standing policy.

**Tech stack:** Python, PyTorch, unittest, mjlab CPU smoke, existing remote GPU evaluation.

## 1. Behavioral tests

- [x] Update the existing entry test to assert that one/two bad samples preserve 0.5 s and the third clears it.
- [x] Add tests for recovery, failing different conditions, physical failure, and finished-world freezing.
- [x] Add keyboard tests exercising the same policy, good-frame-only handoff, duplicate-step calls, and partial reset.
- [x] Add recipe validation that rejects saved promotion evidence from the old rule.
- [x] Run focused unittest suites and confirm failures describe the missing new behavior (6 failures and 3 missing-feature errors).

## 2. Shared timer and integration

- [x] Add `src/tasks/transition_t/standing.py` with constants, a policy contract, and a tensor update helper:

```python
next_bad = torch.where(good, 0, bad_samples + 1).clamp(max=3)
next_duration = torch.where(good, duration + dt,
                           torch.where(next_bad >= 3, 0., duration))
# Ineligible active worlds clear immediately; inactive worlds preserve state.
```

- [x] Integrate the helper and a per-world bad counter into `EntryTrials` and `KeyboardTransitionCommand`.
- [x] Add rule metadata to evaluation reports, status diagnostics, and entry recipe version 5.
- [x] Update the transition README to define pause versus reset and keep strict endpoint diagnostics explicit.
- [ ] Run the focused suites, inspect the complete diff, and commit only this feature.

## 3. Verification and report

- [ ] Integrate the tested commit into the user's current branch, preserving unrelated edits.
- [ ] Deploy a separate copy of the relevant code for an evaluation-only run using `model_4999.pt`, the original bank, 64 attempts, and seed 424242.
- [ ] Compare revised standing eligibility with the old result while explicitly stating there were no PPO updates.
- [ ] Present the bounded 0.1-weight hold reward proposal and proposed 10 s episodes for discussion.
