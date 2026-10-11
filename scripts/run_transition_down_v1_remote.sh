#!/usr/bin/env bash
# Hash-verified GPU smoke → nominal from scratch → reverse entry course → evaluation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
task_python_bin="${TRANSITION_DOWN_PYTHON:-/opt/conda/bin/python}"
task_run_dir="${TRANSITION_DOWN_RUN_DIR:-outputs/transition_down_remote_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$task_run_dir")"
mkdir "$task_run_dir"
task_run_dir="$(realpath "$task_run_dir")"
printf '%s\n' "$$" > "$task_run_dir/runner.pid"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_run_dir/exit_code"' EXIT
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/transition_down_mpl XDG_CACHE_HOME=/tmp/transition_down_cache
export MUJOCO_GL=egl
sha256sum --check deployment_files.sha256 > "$task_run_dir/hash_check.log"
"$task_python_bin" - "$task_run_dir" <<'PY'
from importlib.metadata import version
from pathlib import Path
import json,sys,torch
from src.tasks.transition_down_t.data import verify_approved_assets,verify_teachers,DEFAULT_ENTRY_BANK
verify_approved_assets(bank_file=DEFAULT_ENTRY_BANK)
verify_teachers('mark/quad_teacher.pt','mark/biped_teacher.pt')
expected={'mjlab':'1.2.0','mujoco':'3.5.0','mujoco-warp':'3.5.0','warp-lang':'1.12.0','rsl-rl-lib':'5.0.1'}
for p,v in expected.items():
    if version(p)!=v:raise RuntimeError(f'{p}: require {v}, found {version(p)}')
if not torch.cuda.is_available():raise RuntimeError('CUDA required')
record=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),versions=expected,
    seed=42,num_envs=4096,nominal_updates=30001,entry_updates=10000,checkpoint_interval=100,
    entry_eval_interval=250,entry_eval_distinct_holdouts=64)
Path(sys.argv[1],'launch.json').write_text(json.dumps(record,indent=2)+'\n')
PY
task_motion=outputs/biped_to_tripod_preparation_20261010/biped_to_tripod_reference.npz
task_bank=outputs/biped_to_tripod_preparation_20261010/biped_entry_bank.npz
printf 'gpu_smoke\n' > "$task_run_dir/stage"
"$task_python_bin" scripts/smoke_transition_down.py --device cuda:0 --num-envs 8 --ppo-updates 2 \
  --motion-file "$task_motion" --output "$task_run_dir/gpu_smoke" > "$task_run_dir/gpu_smoke.log" 2>&1
"$task_python_bin" scripts/smoke_transition_down_entry.py --device cuda:0 \
  --motion-file "$task_motion" --entry-bank "$task_bank" \
  --checkpoint "$task_run_dir/gpu_smoke/roundtrip.pt" --output "$task_run_dir/gpu_entry_smoke" \
  > "$task_run_dir/gpu_entry_smoke.log" 2>&1
printf 'nominal\n' > "$task_run_dir/stage"
task_nominal_dir="logs/rsl_rl/transition_down_t/$(basename "$task_run_dir")_nominal"
"$task_python_bin" scripts/train_transition_down.py --device cuda:0 --num-envs 4096 --iterations 30001 \
  --seed 42 --motion-file "$task_motion" --output "$task_nominal_dir" > "$task_run_dir/nominal.log" 2>&1 &
task_training_pid=$!
printf '%s\n' "$task_training_pid" > "$task_run_dir/training.pid"
wait "$task_training_pid"
task_nominal_checkpoint="$task_nominal_dir/model_30000.pt"
printf 'nominal_evaluation\n' > "$task_run_dir/stage"
"$task_python_bin" scripts/evaluate_transition_down.py --agent trained --device cuda:0 \
  --checkpoint-file "$task_nominal_checkpoint" --motion-file "$task_motion" \
  --episodes-per-group 100 --output "$task_run_dir/nominal_full_start.json" > "$task_run_dir/nominal_evaluation.log" 2>&1
printf 'entry\n' > "$task_run_dir/stage"
task_entry_dir="logs/rsl_rl/transition_down_t_entry/$(basename "$task_run_dir")_entry"
"$task_python_bin" scripts/train_transition_down_entry.py --device cuda:0 --num-envs 4096 --iterations 10000 \
  --motion-file "$task_motion" --entry-bank "$task_bank" --checkpoint "$task_nominal_checkpoint" \
  --eval-interval 250 --eval-attempts 64 --baseline all --seed 42 --output "$task_entry_dir" \
  > "$task_run_dir/entry.log" 2>&1 &
task_training_pid=$!
printf '%s\n' "$task_training_pid" > "$task_run_dir/training.pid"
wait "$task_training_pid"
task_final_checkpoint="$task_entry_dir/model_9999.pt"
printf 'final_evaluation\n' > "$task_run_dir/stage"
"$task_python_bin" scripts/evaluate_transition_down.py --agent trained --device cuda:0 \
  --checkpoint-file "$task_final_checkpoint" --motion-file "$task_motion" --episodes-per-group 100 \
  --output "$task_run_dir/final_full_start.json" > "$task_run_dir/final_full_start.log" 2>&1
"$task_python_bin" scripts/evaluate_transition_down.py --agent trained --device cuda:0 \
  --checkpoint-file "$task_final_checkpoint" --motion-file "$task_motion" --episodes-per-group 100 \
  --startup-randomization --observation-noise --output "$task_run_dir/final_augmented.json" \
  > "$task_run_dir/final_augmented.log" 2>&1
"$task_python_bin" scripts/evaluate_transition_down_entry.py --device cuda:0 --motion-file "$task_motion" \
  --entry-bank "$task_bank" --checkpoint "$task_final_checkpoint" --level all --attempts 64 \
  --output "$task_run_dir/final_entries.json" > "$task_run_dir/final_entries.log" 2>&1
"$task_python_bin" scripts/evaluate_transition_down_chain.py --device cuda:0 --motion-file "$task_motion" \
  --entry-bank "$task_bank" --checkpoint "$task_final_checkpoint" --attempts 128 \
  --quad-teacher mark/quad_teacher.pt --biped-teacher mark/biped_teacher.pt \
  --output "$task_run_dir/final_chain.json" > "$task_run_dir/final_chain.log" 2>&1
for task_group in nominal combined; do
  "$task_python_bin" scripts/record_transition_down.py --device cuda:0 --checkpoint "$task_final_checkpoint" \
    --motion-file "$task_motion" --group "$task_group" --output "$task_run_dir/video_${task_group}" \
    > "$task_run_dir/video_${task_group}.log" 2>&1
done
sha256sum "$task_nominal_checkpoint" "$task_final_checkpoint" > "$task_run_dir/final_models.sha256"
printf 'complete\n' > "$task_run_dir/stage"
