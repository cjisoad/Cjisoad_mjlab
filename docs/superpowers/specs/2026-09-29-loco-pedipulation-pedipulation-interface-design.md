# Loco Pedipulation Pedipulation-Compatible Interface Design

## Goal

Make `loco_pedipulation` use the same actor observation and six-value command ABI as `pedipulation`, while retaining its locomotion, three-leg support, right-front transition machine, and reward design.

## Policy interface

The actor receives exactly 48 values in this order:

1. body-frame angular velocity, scaled by 0.25 (3),
2. body-frame projected gravity (3),
3. six-value command `[vx, vy, wz, dx, dy, dz]`, scaled by `[2, 2, .25, 1, 1, 1]` (6),
4. joint position relative to default (12),
5. joint velocity, scaled by 0.05 (12),
6. previous raw action (12).

Training applies the same actor-only uniform noise amplitudes as `pedipulation`: ±0.05 to angular velocity/gravity, ±0.01 to joint position, ±0.075 to scaled joint velocity, and none to commands or prior actions. The actor network remains 512/256/128 ELU with 12 action outputs, but must be trained from scratch because 65-dimensional checkpoint weights are incompatible.

## Command and manipulation activation

The command term exposes `[vx, vy, wz, dx, dy, dz]` in the gravity-aligned anchor frame. The velocity components retain Loco Pedipulation's sampling limits and current tripod speed limits. The FR target offset uses the task's existing reachable FK target bank. A target is active exactly when any of `dx`, `dy`, or `dz` is nonzero; no separate actor-visible enable flag exists.

The internal command state may retain `enabled`, `phase`, `blend`, target reference and contact history for transition logic, rewards, debug visualization, and metrics. `enabled` is derived from the final three command values each command update. An active target drives the existing `PREPARE`, `REACH`, and `HOLD` transitions; removing it drives `LOWER` and `RECOVER`.

## Critic and rewards

Rewards retain their existing locomotion/tripod, FR tracking, support, contact, posture, and landing objectives. The critic remains asymmetric and may retain privileged base velocity, foot state, contact, and internal manipulation state. It shares the exact noisy 48-value actor observation rather than regenerating a distinct corrupted policy observation.

## Playback and validation

The keyboard player writes the same six-value command. The target is active only for nonzero offset. Unit tests will verify 48-dimensional actor observation order/noise contract, six-dimensional command activation, target removal, and manual keyboard bridge behavior. The smoke path is updated for the retrained checkpoint interface; the existing 65-input model is deliberately rejected by strict loading.
