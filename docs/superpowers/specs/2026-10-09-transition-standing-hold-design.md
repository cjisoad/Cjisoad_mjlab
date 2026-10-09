# Transition standing hold: three-sample tolerance and shaping reward

The user approved changing standing eligibility so that its timer resets only
after three consecutive bad control samples. The reward and 10 s episodes were subsequently approved by the user, along with pushing the change and a fresh 10,000-update warm start from the original model_30000.pt.

## Approved timer behavior

At 50 Hz, each good sample adds 0.02 s to the standing timer and clears the
consecutive-bad counter. Bad samples one and two pause the timer without adding
credit. The third consecutive bad sample resets the timer to zero. A bad sample
means any standing condition fails, even if different conditions fail on
successive samples. Success requires a currently good sample and at least 1 s
of credited standing time.

Reference completion is required before any standing credit. Actual physical
failure and nonfinite state remain immediate failures; they receive no debounce.
Finished evaluation worlds retain their captured values. Partial world resets
clear both counters for only the selected worlds. Repeated keyboard updates in
one control step cannot advance either counter twice.

Held-out curriculum evaluation and keyboard teacher handoff share the timer
implementation. Strict endpoint accuracy remains a separate diagnostic with its
existing uninterrupted hold definition. The new standing definition is recorded
in evaluation JSON and in a new entry-training recipe version, so strict resume
cannot reuse promotion evidence collected under a different success definition.

## Approved reward

Recommended additional reward, weight 0.1:

    r_hold = reference_completed * currently_standing
             * (0.2 + 0.8 * min(credited_hold_seconds / 1.0, 1.0))

Only good samples earn this reward. The first two bad samples earn zero while
preserving previously credited duration; the third resets duration. Resetting a
world clears its reward state. The reward stays bounded at one before weighting
and remains at its maximum on good samples after one second. Physics failure
earns zero. It uses the existing handoff conditions, not the separate 6 cm
world-foot criterion, and leaves FR target velocity/acceleration limits intact.

Compared with a constant per-good-sample bonus, this favors sustained standing.
Compared with a one-time success bonus, it provides feedback before a full second
has been achieved. Intermittent stepping earns no reward during bad samples, so
stable standing earns more reward per real second.

Approved accompanying training change: extend episodes from 8 s to 10 s. The
reference has 351 frames at 50 Hz (7 s), leaving 1 s of terminal practice in an
8 s full-start episode and 3 s in a 10 s episode. Rewards are scaled by dt, so a
0.1-weight bounded reward adds at most 0.1 per simulated second (0.002 per good
control step), or 0.3 over three seconds before accounting for the ramp.
The new run explicitly forks the original model_30000.pt policy, retaining actor/critic/normalizers and starting fresh optimizer/course state. The completed entry-course model_4999.pt is excluded by the user. Entry recipe v6 records rewards and episode length and rejects strict resume with a changed objective.

## Verification

Test one/two bad samples, third-sample reset, recovery between bad streaks,
mixed failing conditions, immediate physical/nonfinite failure, no pre-reference
credit, no bad-frame handoff, frozen completed worlds, partial reset, and one
update per control step. Re-evaluate the same final checkpoint and held-out seed
under the new rule without PPO updates; report it as a criterion change, not as
an improvement in policy weights.
