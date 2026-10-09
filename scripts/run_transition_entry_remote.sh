#!/usr/bin/env bash
# Run from an unpacked, hash-verified deployment directory.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
task_python_bin="${TRANSITION_PYTHON_BIN:-/opt/conda/bin/python}"
task_run_dir="${TRANSITION_RUN_DIR:-outputs/transition_target_limiter_20261009/remote_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$task_run_dir")"
mkdir "$task_run_dir"
task_run_dir="$(realpath "$task_run_dir")"
printf '%s\n' "$$" > "$task_run_dir/runner.pid"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_run_dir/exit_code"' EXIT
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/transition_entry_mpl
export WARP_CACHE_PATH=/tmp/transition_entry_warp

"$task_python_bin" - <<'PY'
from importlib.metadata import version
import torch
expected = {'mjlab': '1.2.0', 'mujoco': '3.5.0', 'mujoco-warp': '3.5.0',
            'warp-lang': '1.12.0', 'rsl-rl-lib': '5.0.1'}
for package, wanted in expected.items():
    actual = version(package)
    print(package, actual, flush=True)
    if actual != wanted:
        raise RuntimeError(f'{package}: require {wanted}, found {actual}')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable; 4096-environment training requires a GPU')
print(torch.cuda.get_device_name(0), flush=True)
PY
task_free_kib="$(df -Pk "$task_run_dir" | awk 'NR == 2 {print $4}')"
if [[ -z "$task_free_kib" || "$task_free_kib" -lt 5242880 ]]; then
  printf 'Require at least 5 GiB free; found %s KiB\n' "${task_free_kib:-unknown}" >&2
  exit 1
fi
printf 'Free disk: %s KiB\n' "$task_free_kib"
sha256sum --check deployment_files.sha256 > "$task_run_dir/hash_check.log"

"$task_python_bin" scripts/collect_transition_entries.py \
  --device cuda:0 --num-envs 256 --batches 16 --seconds 8 \
  --sample-interval .2 --seed 42 --output "$task_run_dir/entries.npz" \
  > "$task_run_dir/collection.log" 2>&1

"$task_python_bin" - "$task_run_dir/entries.npz" > "$task_run_dir/bank_coverage.json" <<'PY'
import json
import sys
from src.tasks.transition_t.entry_bank import EntryStateBank
from src.tasks.transition_t.motion import DEFAULT_MOTION_PATH, TransitionMotion
from src.tasks.transition_t.env_cfg import transition_env_cfg
from src.tasks.pedipulation.mdp.target_limiter import target_limiter_contract
cfg = transition_env_cfg(play=True)
action = cfg.actions['joint_pos']
limiter = target_limiter_contract(action.target_velocity_limits, action.target_acceleration_limits,
    cfg.sim.mujoco.timestep)
bank = EntryStateBank(sys.argv[1], motion_sha256=TransitionMotion(DEFAULT_MOTION_PATH).sha256,
    target_limiter_contract=limiter)
coverage = bank.coverage()
print(json.dumps(dict(bank_sha256=bank.sha256, coverage=coverage,
    collection_coverage=bank.metadata['coverage']), indent=2), flush=True)
for split, levels in coverage.items():
    for level in levels:
        if level['trajectories'] < 64:
            raise RuntimeError(f'Insufficient independent {split} trajectories: {level}')
PY

"$task_python_bin" scripts/train_transition_entry.py \
  --entry-bank "$task_run_dir/entries.npz" \
  --device cuda:0 --num-envs 4096 --iterations 5000 \
  --eval-interval 250 --eval-attempts 64 --baseline all --seed 42 \
  --output "$task_run_dir/training" \
  > "$task_run_dir/training.log" 2>&1 &
task_training_pid=$!
printf '%s\n' "$task_training_pid" > "$task_run_dir/training.pid"
wait "$task_training_pid"

"$task_python_bin" scripts/evaluate_transition_entry.py \
  --entry-bank "$task_run_dir/entries.npz" \
  --checkpoint "$task_run_dir/training/model_4999.pt" \
  --device cuda:0 --level all --attempts 64 --seed 424242 \
  --output "$task_run_dir/final_evaluation.json" \
  > "$task_run_dir/final_evaluation.log" 2>&1
printf 'ENTRY_REMOTE_FINISHED %s\n' "$task_run_dir"
