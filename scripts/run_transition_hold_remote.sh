#!/usr/bin/env bash
# Warm start from an original transition checkpoint with an existing entry bank.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'Usage: bash %s SEED_CHECKPOINT ENTRY_BANK FRESH_RUN_DIRECTORY [UPDATES=10000]\n' "$0" >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
task_python_bin="${TRANSITION_PYTHON_BIN:-/opt/conda/bin/python}"
task_seed="$(realpath "$1")"
task_bank="$(realpath "$2")"
task_run_dir="$(realpath -m "$3")"
task_updates="${4:-10000}"
[[ "$task_updates" =~ ^[1-9][0-9]*$ ]] || { printf 'UPDATES must be a positive integer\n' >&2; exit 2; }
[[ -f "$task_seed" && -f "$task_bank" ]] || { printf 'Seed and entry bank must exist\n' >&2; exit 2; }
mkdir -p "$(dirname "$task_run_dir")"
mkdir "$task_run_dir"
printf '%s\n' "$$" > "$task_run_dir/runner.pid"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_run_dir/exit_code"' EXIT
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR=/tmp/transition_hold_mpl
export XDG_CACHE_HOME=/tmp/transition_hold_cache
export WARP_CACHE_PATH=/tmp/transition_entry_warp
if [[ -f deployment_files.sha256 ]]; then
  sha256sum --check deployment_files.sha256 > "$task_run_dir/hash_check.log"
fi
"$task_python_bin" - "$task_seed" "$task_bank" "$task_run_dir" "$task_updates" <<'PY'
from pathlib import Path
import hashlib,json,sys
import torch
seed,bank,output=map(Path,sys.argv[1:4])
saved=torch.load(seed,map_location='cpu',weights_only=False)
infos=saved.get('infos') or {}
if 'transition_entry_training' in infos:
    raise ValueError('Use the original transition seed, not an entry-course model')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA required for the 4096-world run')
record=dict(seed_checkpoint=str(seed),seed_checkpoint_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
    entry_bank=str(bank),entry_bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
    updates=int(sys.argv[4]),num_envs=4096,device='cuda:0',seed=42,
    initialization='original actor/critic/normalizers; fresh optimizer and course',
    episode_length_s=10.,standing_hold_weight=.1)
(output/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
print('TRANSITION_HOLD_LAUNCH '+json.dumps(record),flush=True)
PY
"$task_python_bin" scripts/train_transition_entry.py \
  --checkpoint "$task_seed" --entry-bank "$task_bank" \
  --device cuda:0 --num-envs 4096 --iterations "$task_updates" \
  --eval-interval 250 --eval-attempts 64 --baseline all --seed 42 \
  --output "$task_run_dir/training" \
  > "$task_run_dir/training.log" 2>&1 &
task_training_pid=$!
printf '%s\n' "$task_training_pid" > "$task_run_dir/training.pid"
wait "$task_training_pid"
task_final_index=$((task_updates-1))
"$task_python_bin" scripts/evaluate_transition_entry.py \
  --entry-bank "$task_bank" \
  --checkpoint "$task_run_dir/training/model_${task_final_index}.pt" \
  --device cuda:0 --level all --attempts 64 --seed 424242 \
  --output "$task_run_dir/final_evaluation.json" \
  > "$task_run_dir/final_evaluation.log" 2>&1
printf 'TRANSITION_HOLD_FINISHED %s\n' "$task_run_dir"
