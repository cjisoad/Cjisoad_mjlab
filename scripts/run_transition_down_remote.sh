#!/usr/bin/env bash
# Reviewed V2 landing: GPU readiness then one continuous, evaluated PPO course.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
task_python_bin="${TRANSITION_DOWN_PYTHON:-/opt/conda/bin/python}"
task_run_dir="${TRANSITION_DOWN_RUN_DIR:-outputs/transition_down_v2_remote_$(date +%Y%m%d_%H%M%S)}"
task_worlds="${TRANSITION_DOWN_NUM_ENVS:-4096}"
task_budget="${TRANSITION_DOWN_MAX_UPDATES:-40000}"
task_eval_interval="${TRANSITION_DOWN_EVAL_INTERVAL:-250}"
mkdir -p "$(dirname "$task_run_dir")"
mkdir "$task_run_dir"
task_run_dir="$(realpath "$task_run_dir")"
printf '%s\n' "$$" > "$task_run_dir/runner.pid"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_run_dir/exit_code"' EXIT
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/transition_down_v2_mpl XDG_CACHE_HOME=/tmp/transition_down_v2_cache
printf 'release_verification\n' > "$task_run_dir/stage"
sha256sum --check deployment_files.sha256 > "$task_run_dir/hash_check.log"
"$task_python_bin" - "$task_run_dir" "$task_worlds" "$task_budget" "$task_eval_interval" <<'PY'
from importlib.metadata import version
from pathlib import Path
import json,sys,torch
from src.tasks.transition_down_v2.data import verify_assets
from src.tasks.transition_down_t.data import verify_teachers
expected={'mjlab':'1.2.0','mujoco':'3.5.0','mujoco-warp':'3.5.0','warp-lang':'1.12.0','rsl-rl-lib':'5.0.1'}
for p,v in expected.items():
    if version(p)!=v:raise RuntimeError(f'{p}: require {v}, found {version(p)}')
verify_assets();verify_teachers('mark/quad_teacher.pt','mark/biped_teacher.pt')
if not torch.cuda.is_available():raise RuntimeError('CUDA required')
Path(sys.argv[1],'launch.json').write_text(json.dumps(dict(
    task='Transition-Down-Go2-v2',source=json.loads(Path('source_release.json').read_text()),
    torch=torch.__version__,gpu=torch.cuda.get_device_name(0),versions=expected,
    seed=42,num_envs=int(sys.argv[2]),total_update_budget=int(sys.argv[3]),
    eval_interval=int(sys.argv[4]),fixed_stage_updates=False,optimizer_continuous=True),indent=2)+'\n')
PY
# Generate missing teacher geometry on this server; no local cache upload needed.
printf 'teacher_workspace\n' > "$task_run_dir/stage"
if [[ ! -f outputs/fr_reachability/go2_fr_v1.npz ]]; then
  "$task_python_bin" scripts/build_teacher_reachability.py > "$task_run_dir/workspace_build.log" 2>&1
elif [[ ! -f outputs/fr_reachability/go2_fr_v1_train.npz ]]; then
  "$task_python_bin" scripts/build_teacher_reachability.py --training-only > "$task_run_dir/workspace_build.log" 2>&1
fi
"$task_python_bin" - <<'PY'
from src.tasks.teacher_common.training_workspace import load_training_workspace
workspace=load_training_workspace()
print('Teacher workspace targets:',len(workspace.offsets),flush=True)
PY
printf 'gpu_smoke\n' > "$task_run_dir/stage"
"$task_python_bin" scripts/train_transition_down_v2.py --device cuda:0 --num-envs 8 --smoke \
  --max-updates 2 --eval-interval 2 --output "$task_run_dir/gpu_smoke" \
  --gate-config src/tasks/transition_down_v2/gates_default.json > "$task_run_dir/gpu_smoke.log" 2>&1
printf 'adaptive_training\n' > "$task_run_dir/stage"
task_resume_args=()
if [[ -n "${TRANSITION_DOWN_RESUME:-}" ]]; then task_resume_args=(--resume "$TRANSITION_DOWN_RESUME"); fi
"$task_python_bin" scripts/train_transition_down_v2.py --device cuda:0 --num-envs "$task_worlds" \
  --max-updates "$task_budget" --eval-interval "$task_eval_interval" --seed 42 \
  --gate-config src/tasks/transition_down_v2/gates_default.json \
  --output "$task_run_dir/training" "${task_resume_args[@]}" > "$task_run_dir/training.log" 2>&1 &
task_training_pid=$!
printf '%s\n' "$task_training_pid" > "$task_run_dir/training.pid"
set +e
wait "$task_training_pid"
task_training_exit=$?
set -e
if [[ -f "$task_run_dir/training/finished.json" ]]; then
  "$task_python_bin" - "$task_run_dir" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);result=json.loads((root/'training/finished.json').read_text())
(root/'stage').write_text(result['status']+'\n')
print('Training finished:',result['status'],'updates:',result['updates_completed'])
PY
else
  printf 'training_failed\n' > "$task_run_dir/stage"
fi
exit "$task_training_exit"
