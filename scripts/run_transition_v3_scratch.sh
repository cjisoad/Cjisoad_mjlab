#!/usr/bin/env bash
# Run after remote environment/ClearML setup; never loads old teacher weights.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
task_bank="${1:-src/tasks/transition_t/data/entries_v3_speed125_train.npz}"
task_output="${2:-logs/rsl_rl/transition_t_v3/$(date +%Y-%m-%d_%H-%M-%S)_scratch}"
task_python_bin="${TRANSITION_PYTHON_BIN:-python}"
exec "$task_python_bin" scripts/train_transition_entry.py \
  --entry-bank "$task_bank" --curriculum three-stage --from-scratch \
  --motion-file src/tasks/transition_t/data/tripod_to_biped_v3_speed125_train.npz \
  --device "${TRANSITION_DEVICE:-cuda:0}" --num-envs "${TRANSITION_NUM_ENVS:-4096}" \
  --iterations "${TRANSITION_ITERATIONS:-30000}" --eval-interval 250 \
  --baseline current --seed 42 --output "$task_output" \
  --clearml --clearml-project "${TRANSITION_CLEARML_PROJECT:-Go2/transition_t}" \
  --clearml-task-name "${TRANSITION_CLEARML_TASK:-V3 three-stage scratch}" \
  --clearml-checkpoint-interval 1000
