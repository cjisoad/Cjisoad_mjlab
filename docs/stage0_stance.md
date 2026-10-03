# Stage0 stance course, version2

Stage0 trains nominal quadruped and rear-leg biped posture holding. Strict stillness and switch certification remain diagnostics. Stage1/2 retain their stricter training-window gates.

Every500 iterations the runner evaluates64 independent environments (32 perstance), mean actions, no actor noise or interval push, with physical randomization and action delay retained. Seeds are independent of training; Python/NumPy/PyTorch random states are restored, and the training environment is untouched. On distributed runs rank0 evaluates and broadcasts one decision.

Protocol:2s quad warmup; request biped for the second32 trials; allow5s arrival and1s settling; then score5s. Any hard termination during arrival or scoring permanently fails that trial. Resetting never erases a failure. Failed trials contribute zero remaining planned frames. Each stance has8000 planned frames at50Hz.

Frame conditions:
- Quad tilt<20deg, nominal height±.06m, mean joint pose error<.25rad.
- Biped tilt<25deg, height.44–.62m, mean joint pose error<.35rad over all12 nominal joints.
- Each required support foot may lose contact for at most60ms. Quad needs at least3 current feet; biped at least1 current rear foot and no front contact.
- Over200ms, mean linear speed magnitude<.3m/s and angular speed magnitude<1rad/s. Opposite directions do not cancel.

Promotion: both stances qualified frame fraction>=.70 and nonfailed trial fraction>=.85, at least32 trials and1000 planned frames perstance, two consecutive independent passes. A failed evaluation resets the streak. `StanceEval/*` contains the authoritative promotion metrics. `Metrics/twist/*ready_fraction` remains strict stillness; `episode_*` records completed-episode summaries without overwriting streaming statistics.

Stage0 exploration: actual Gaussian std initial.2, effective bounds.02–.3; entropy.001. Sampling, log_prob, entropy and KL use the same distribution. On entering stage1 the std cap and entropy increase linearly to1.0/.01 over1500 iterations. No extra action noise scaling occurs after PPO sampling.

Stage0 interval pushes are disabled by default. Optional `--env.commands.twist.stage0-light-push True` enables±.05m/s horizontal and±.05rad/s yaw changes every10–15s, only in stable QUAD/HOLD after1s. Stage1 push amplitudes increase with velocity_level to the original±.15/±.2 limits. Push writes are followed by physics forward before actor/critic observations. Physical startup initialization remains enabled.

## Migrate an old model once

From the repository root with the configured conda environment:

```bash
export PYTHONPATH=.
python scripts/migrate_leg_manip_stance_checkpoint.py OLD_MODEL.pt logs/rsl_rl/leg_manip/stage0_restart/model_0.pt
```

Migration preserves actor/value network weights, resets per-joint std to.2, clears Adam moments and all course/event statistics, resets iteration/step to0. It refuses to overwrite its destination. Version1 checkpoints cannot silently resume under version2 gates. Preserve the original checkpoint.

The locally migrated model already exists at `logs/rsl_rl/leg_manip/stage0_restart/model_0.pt`.

## Train from migrated weights

```bash
cd /home/e980/projects/Cjisoad_mjlab
source /home/e980/miniforge3/etc/profile.d/conda.sh
conda activate cjisoad_mjlab
export PYTHONPATH=.
python scripts/train.py leg_manip --gpu-ids '[0]' --env.scene.num-envs 1024 --agent.resume True --agent.load-run '^stage0_restart$' --agent.load-checkpoint 'model_0.pt' --agent.run-name stage0_stance_v2 --agent.wandb-project pedipulation
```

## Evaluate without training

```bash
python scripts/evaluate_leg_manip_stance.py logs/rsl_rl/leg_manip/stage0_restart/model_0.pt --seed 100542 --output outputs/stance_eval_100542.json
```

One evaluation does not establish the required two consecutive passes. Repeat with an independent seed for diagnosis; training controls actual promotion.
