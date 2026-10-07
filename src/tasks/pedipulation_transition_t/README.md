# Tripod-to-biped transition teacher

`pedipulation_transition_t` learns the upward transition from FL/RL/RR tripod
support to RL/RR biped support while FR follows a smooth base-relative target.
It is a separate teacher with fresh rewards and PPO; it does not load either
existing teacher's incompatible checkpoints.

## Reference data and its limits

`src/assets/motions/go2/tripod_to_biped.npz` contains a 50 Hz, 6 s reference
(301 frames), canonical FL/FR/RL/RR joints, root state and velocities, foot
positions, contact phases, FL unloading, static forces/torques, and 25 reachable
FR trajectories. MuJoCo supplies kinematics and inverse dynamics; SciPy solves
support-constrained IK. The reference prepares until 0.5 s, shifts weight with
a short crouch through 0.95 s, rises from 0.9 to 2.5 s, then holds upright.

The final static audit found maximum nominal IK error 0.87 micrometers, maximum
nominal base equilibrium residual 0.00000244 times body weight, maximum residual
over all FR variants 0.00125 times body weight, maximum static torque utilization
64.0%, and minimum FR ground clearance 7.30 cm. Joint margins and collision
checks pass for the nominal reference and all FR variants.

**This is a kinematic/static training reference, not a validated dynamic
demonstration.** The supplied controller uses only the 12 joint actuators, with
no externally applied base forces or subsequent root teleportation. Its replay
fails at 0.94 s when the FR calf hits the ground; it does not complete the rise.
The audit explicitly records `dynamic_tracking_validated=false`. PPO must learn
dynamic balance, and only policy evaluation establishes whether it succeeds.

The output audit and separately labelled previews live in
`outputs/transition_motion/`: `audit.json`, `kinematic_reference.webm`,
`joint_only_replay.webm`, and `reference_curves.png`. VP8/WebM previews play on
Ubuntu without proprietary H.264 decoders. Failed candidate assets are
checked before atomic replacement, preserving any previously accepted asset.

## Training behavior

Control uses the standing Go2 model, nonzero nominal hip/thigh/calf armatures
0.01/0.01/0.02, PD gains 40/1, 50 Hz actions, and canonical position targets
`default_angles + 0.25 * clipped_action`. Actions clip at ±10. FR targets use
the existing teachers' common gravity-aligned anchor and follow base translation.
Reference heading is aligned at reset. Absolute horizontal base position is
not rewarded, and FR is excluded from joint imitation.

Actor84 contains the common actor51 plus height, FR error, progress, body-frame
reference orientation error (rotation vector, including heading), reference
height, nine support-joint positions and velocities, and six reference velocity
channels. Critic172 reuses the exact same actor noise sample and adds clean
privileged velocity, feet, contacts and physics. The reference orientation error
replaces reference gravity to make the full attitude reward observable.

Fresh exponential rewards weight orientation 3, height 1.5, FR tracking 3,
support-joint pose 0.5, linear/angular reference velocities 0.3 each, and FL
height 0.5. Soft penalties cover contact loss, unwanted front contact, sliding,
force, collision, normalized target changes (0.03), torque (0.02), and limits.
Reference-relative posture/height failures persist for 0.2 s; impacts, severe
limits and invalid state terminate immediately. Failure costs −5 once, with
the environment's reward-dt scaling accounted for. Timeouts are free.

Resets use 50% full tripod starts, 30% rise frames, and 20% endpoint states,
with matching root/joint velocities. Stages introduce, in order:

0. Nominal reference tracking.
1. FR trajectory-bank diversity.
2. Timing scale 0.85–1.15.
3. Initial velocity and post-stand velocity impulses.
4. Additional mid-rise impulses.
5. Physics randomization, observation noise and control delay.

At least 35% of worlds remain undisturbed. Initial velocity additions are
±0.12 m/s per axis and ±0.18 rad/s; subsequent impulses add horizontal velocity
up to ±0.2 m/s and angular velocity up to ±0.3 rad/s. Final-stage randomization
includes mass/inertia ±10%, COM ±1.5 cm, friction 0.6–1.2, armature ±20%,
PD ±10%, motor offsets ±0.015 rad, damping and joint friction.

Promotion requires at least 128 completed full-start episodes, at least 85%
success in the current window (up to 512), and 1,500 stage steps. Full-start
and reference-reset results are recorded separately. Course state survives
checkpoint loading; evaluation never promotes the course.

Success requires a continuous one-second endpoint hold: attitude within 15°,
height and FR error within 5 cm, front feet airborne, horizontal speed below
0.2 m/s and angular speed below 0.5 rad/s. Both rear feet must have contacted
within 0.15 s, with one currently supporting, permitting a brief recovery step.
A post-stand impulse clears previous held success and requires a fresh hold.
Fixed playback is used; adaptive phase slowdown is not implemented.

## Run locally

Use the repository's `mjlab` Python environment. All paths below are relative
to the repository root. The task carries its motion path internally.

```bash
python scripts/generate_transition_motion.py --dynamic-replay
python scripts/preview_transition_motion.py
python scripts/preview_transition_policy.py \
  --checkpoint logs/rsl_rl/pedipulation_transition_t/2026-10-06_01-41-07_transition_formal/model_14999.pt \
  --fr-bank 7
python scripts/smoke_transition.py --device cuda:0 --num-envs 32 --stage 5 --ppo
python scripts/train.py pedipulation_transition_t \
  --env.scene.num-envs 1024 --agent.run-name transition_local
```

The default run uses TensorBoard, local checkpoints every 100 iterations, and
15,000 PPO iterations. It does not upload models. A bounded pilot can override
`--agent.max-iterations 400`. Training checkpoints reject mismatched motion,
model, observation or action contracts rather than silently loading weights.

```bash
python scripts/evaluate_transition.py --checkpoint /absolute/path/model_400.pt \
  --stage 0 --episodes 128 --output outputs/transition_nominal_evaluation.json
python scripts/evaluate_transition.py --checkpoint /absolute/path/model_14999.pt \
  --stage 5 --disturbances --episodes 128 \
  --output outputs/transition_disturbed_evaluation.json
```

The policy preview runs one deterministic 10 s MuJoCo rollout from a full
tripod start, draws the FR reference marker, and overlays live tracking errors.
It writes VP8/WebM for Ubuntu playback.

Evaluation uses full tripod starts with fixed episode quotas per world, so
fast failures do not displace slow successes. Nominal stage-0 deterministic
episodes are repeated measurements of one target; they alone do not establish
robustness across FR goals or disturbances. Evaluate the final course with
diversity and disturbances separately. Old-teacher handoff remains untested;
future integration needs joint-target conversion and compatible dynamics.

Verification commands:

```bash
PYTHONPATH=.:tests python -m unittest \
  test_transition_motion test_transition_task test_transition_runtime \
  test_transition_evaluation test_teacher_anchor test_teacher_tasks -v
```

Run metadata, measured pilot results and the formal process status are recorded
in `outputs/transition_training/run_manifest.json`. A running job and passing
smoke tests do not imply convergence. The detached follow-up process in
`outputs/transition_training/evaluate_after_training.py` waits for formal
training to finish, then records nominal, FR-diverse, and disturbed full-start
evaluations. If training exits before the final checkpoint, it records that
failure instead of evaluating an earlier checkpoint as the final result.
