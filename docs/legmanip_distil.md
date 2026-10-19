# `legmanip_distil`

`legmanip_distil` trains a recurrent student to reproduce the existing four
FSM actors in `scripts/keyboard_transition_fsm.py`. The training environment
contains the physical transition command and frozen actors. The play and
evaluation environments contain only the canonical user intent and direct PD
action, so an exported student does not load an FSM, checkpoint, motion archive,
or entry-state bank.

The student contract is fixed and versioned in the checkpoint manifest:

* observation: 48 values in the order body angular velocity, projected gravity,
  six user command values, joint position relative to the common default, joint
  velocity, and the previous actually applied common action;
* scales: `(0.25, 1, 2, 2, .25, 1, 1, 1, 1, .05, 1)` by those fields;
* action: 12 common joint targets in FL/FR/RL/RR hip/thigh/calf order;
* execution: `q_target = q_default + .25 * action`, direct PD at 50 Hz,
  deterministic output, and observation/action clipping at 100;
* `R` is represented by six zero command values.

The model is one GRU layer with hidden size 256 followed by an ELU MLP with
widths 256 and 128 and a linear 12-output head. It has 335,244 parameters.
Teacher labels are the final physical substep's applied common action after the
original delay and joint governor. The student holds that final target across
the control step. Evaluation reports the difference between the original
substep trajectory and this final-target hold per owner state.

The default recipe follows the Parkour distillation scale and optimizer:
256 environments, 32 steps per environment, 60,000 completed updates, eight
epochs, two environment-axis minibatches, L1 action loss summed across 12
coordinates and averaged over frames, AdamW at `3e-4`, weight decay `.01`, and
gradient norm 1. The teacher controls all episodes for the first 1,000 updates;
then its episode-level control probability is
`(1 - 1/2000) ** (completed_updates - 1000)`. Each update writes TensorBoard
scalars and every 500 updates writes `model_*.pt` plus a standalone `student.pt`.

Automatic scenarios sample quadruped/tripod at 25%, biped/down at 25%, and full
cycles at 50%. Entry states use the training bank. Evaluation selects validation
or final bank splits and records the trajectory id. Scenario completion is not
called success: success also requires continuous physical holds, no failure,
and no deadline.

Train the configured task through the normal registry entry:

```bash
python scripts/train.py legmanip_distil
```

For a bounded CPU smoke, use an empty GPU list and a small environment count:

```bash
python scripts/train.py legmanip_distil \
  --gpu-ids None --env.scene.num-envs 2 \
  --agent.max-iterations 2 --agent.algorithm.num-learning-epochs 1 \
  --agent.save-interval 1 --agent.run-name cpu_smoke
```

Resume a completed-update checkpoint with the same teacher/reference hashes:

```bash
python scripts/train.py legmanip_distil \
  --gpu-ids None --agent.resume True \
  --agent.load-run 2026-01-01_00-00-00_name \
  --agent.load-checkpoint model_500.pt
```

Run the standalone student without loading a teacher:

```bash
python scripts/play_legmanip_distil.py \
  --checkpoint-file logs/rsl_rl/legmanip_distil/<run>/student.pt \
  --headless --steps 500
```

Use `--scenario quad|biped|cycle` for an automatic script, or omit it for
keyboard control. In keyboard mode W/S controls `vx`, A/D controls yaw, arrow
keys control the FR offset, Q/E controls its height, and R submits six zero
commands. The viewer resets the recurrent state only when the physical
environment resets.

Evaluate on the disjoint validation or final split:

```bash
python scripts/evaluate_legmanip_distil.py \
  --checkpoint-file logs/rsl_rl/legmanip_distil/<run>/student.pt \
  --policy student --split validation --episodes 16 --num-envs 16 \
  --output outputs/legmanip_distil/student_validation.json
```

For the teacher timing diagnostic, run `--policy compare-teacher-timing`; this
compares teacher substep execution with final-target direct PD and stores both
the per-owner timing gap and physical success fields.

Evaluation snapshots are taken by a nonterminating termination-manager hook
before mjlab automatically resets a completed world. The action and reset flags
are exact for that terminal frame; derived contact and kinematic fields are
reported with `capture_phase=termination_manager_pre_auto_reset`, so they are
not presented as an additional post-step forward pass.

Useful short checks are:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/legmanip_distil_mpl \
  XDG_CACHE_HOME=/tmp/legmanip_distil_cache \
  python -m pytest -q tests/test_legmanip_distil_rl.py \
    tests/test_legmanip_distil_runtime.py \
    tests/test_legmanip_distil_parity.py \
    tests/test_legmanip_distil_deployment.py
```

The implementation was validated with 46 focused CPU tests, a two-environment two-update
real training smoke, checkpoint resume, standalone headless playback, and a
short validation evaluator. CUDA was unavailable in the validation machine, so
the full 256-environment throughput and GPU behavior still need to be measured
on a CUDA host. The smoke runs do not establish convergence.
