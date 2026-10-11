"""Train the V3 rise teacher from scratch or use the legacy real-entry fine tune."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT/'logs/rsl_rl/transition_t/2026-10-09_01-13-48_transition_t_direct_4090_20261009/model_30000.pt'


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--entry-bank', type=Path, required=True)
  parser.add_argument('--checkpoint', type=Path)
  parser.add_argument('--resume', action='store_true')
  parser.add_argument('--from-scratch', action='store_true')
  parser.add_argument('--curriculum', choices=('legacy','three-stage'), default='legacy')
  parser.add_argument('--motion-file', type=Path,
    help='Explicit rise reference; omitted keeps the original reference')
  parser.add_argument('--allow-reference-migration', action='store_true',
    help='Warm start weights on a new reference; discard optimizer, course and phase history')
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--num-envs', type=int, default=4096)
  parser.add_argument('--iterations', type=int, help='Total updates; three-stage default 30000, legacy 5000')
  parser.add_argument('--eval-interval', type=int, default=250)
  parser.add_argument('--eval-attempts', type=int, default=64)
  parser.add_argument('--baseline', choices=('all', 'current', 'none'), default='all')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--output', type=Path)
  parser.add_argument('--clearml', action='store_true', help='Use existing clearml.conf or CLEARML_* environment credentials')
  parser.add_argument('--clearml-offline', action='store_true', help='Save ClearML session locally without network access')
  parser.add_argument('--clearml-project', default='Go2/transition_t')
  parser.add_argument('--clearml-task-name', default='V3 three-stage scratch')
  parser.add_argument('--clearml-output-uri', help='Optional artifact storage URI; defaults to configured ClearML files server')
  parser.add_argument('--clearml-checkpoint-interval', type=int, default=1000)
  args = parser.parse_args(argv)
  if args.from_scratch and (args.checkpoint is not None or args.resume or args.allow_reference_migration):
    parser.error('--from-scratch cannot load a checkpoint, resume or migrate weights')
  if args.curriculum == 'three-stage' and not (args.from_scratch or args.resume):
    parser.error('three-stage requires --from-scratch or --resume --checkpoint')
  if args.from_scratch and args.curriculum != 'three-stage':
    parser.error('--from-scratch requires --curriculum three-stage')
  if args.resume and args.checkpoint is None:
    parser.error('--resume requires --checkpoint')
  if not args.from_scratch and args.checkpoint is None:
    args.checkpoint = DEFAULT_CHECKPOINT
  if args.iterations is None:
    args.iterations = 30000 if args.curriculum == 'three-stage' else 5000
  if args.clearml_offline and not args.clearml:
    parser.error('--clearml-offline requires --clearml')
  if args.resume and args.allow_reference_migration:
    parser.error('--allow-reference-migration is weights-only and cannot be combined with --resume')
  for name in ('num_envs', 'iterations', 'eval_interval', 'eval_attempts', 'clearml_checkpoint_interval'):
    if getattr(args, name) <= 0:
      parser.error(f'--{name.replace("_", "-")} must be positive')
  return args


def write_json(path, data):
  path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def train(args):
  os.environ.setdefault('OMP_NUM_THREADS', '1')
  os.environ.setdefault('MPLCONFIGDIR', '/tmp/transition_t_mpl')
  os.environ.setdefault('XDG_CACHE_HOME', '/tmp/transition_t_cache')
  sys.path.insert(0, str(ROOT))
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.utils.os import dump_yaml
  from src.tasks.transition_t.entry_bank import LEVELS
  from src.tasks.transition_t.entry_course import entry_course_env_cfg
  from src.tasks.transition_t.entry_rl import EntryOnPolicyRunner
  from src.tasks.transition_t.entry_evaluation import evaluate_entries
  from src.tasks.transition_t.rl import transition_ppo_runner_cfg
  from src.tasks.transition_t.clearml_logging import ClearMLTracker
  from src.tasks.transition_t.three_stage import three_stage_env_cfg
  from src.tasks.transition_t.three_stage_rl import ThreeStageOnPolicyRunner

  for path in (p for p in (args.entry_bank, args.checkpoint) if p is not None):
    if not path.is_file():
      raise FileNotFoundError(path)
  run_kind = 'transition_t_v3' if args.curriculum == 'three-stage' else 'transition_t_entry'
  output = args.output or ROOT/'logs/rsl_rl'/run_kind/datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  if output.exists() and any(output.iterdir()):
    raise ValueError(f'Use a fresh output directory: {output}')
  output.mkdir(parents=True, exist_ok=True)
  configure_torch_backends(); torch.set_num_threads(1)
  three_stage = args.curriculum == 'three-stage'
  stage = 0
  if three_stage and args.resume:
    metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('infos') or {}
    state = metadata.get('transition_entry_training') or {}
    stage = (state.get('course') or {}).get('level')
  def make_cfg(stage):
    kwargs = dict(num_envs=args.num_envs,
      motion_file=args.motion_file.resolve() if args.motion_file is not None else None)
    result = (three_stage_env_cfg(args.entry_bank.resolve(), stage=stage, **kwargs) if three_stage
      else entry_course_env_cfg(args.entry_bank.resolve(), **kwargs))
    result.seed = args.seed
    return result
  cfg = make_cfg(stage)
  cfg.seed = args.seed
  agent = transition_ppo_runner_cfg()
  agent.seed = args.seed
  agent.experiment_name = run_kind
  agent.save_interval = min(100, args.eval_interval)
  env = ManagerBasedRlEnv(cfg, device=args.device)
  runner = None
  tracker = None
  try:
    tracker = ClearMLTracker(enabled=args.clearml, project=args.clearml_project,
      task_name=args.clearml_task_name, output_uri=args.clearml_output_uri,
      offline=args.clearml_offline, checkpoint_interval=args.clearml_checkpoint_interval)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner_type = ThreeStageOnPolicyRunner if three_stage else EntryOnPolicyRunner
    runner = runner_type(wrapped, asdict(agent), str(output), device=args.device)
    if args.resume:
      runner.resume(args.checkpoint)
    elif args.from_scratch:
      runner.start_from_scratch()
    else:
      runner.warm_start(args.checkpoint, allow_reference_migration=args.allow_reference_migration)
    completed = runner.current_learning_iteration
    if completed >= args.iterations:
      raise ValueError(f'Resumed checkpoint already has {completed} updates, target is {args.iterations}')
    term = runner.term
    dump_yaml(output/'params/env.yaml', asdict(cfg))
    dump_yaml(output/'params/agent.yaml', asdict(agent))
    write_json(output/'run.json', dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
      output=str(output), bank_sha256=term.bank.sha256, bank_coverage=term.bank.coverage(),
      seed_checkpoint_sha256=runner.seed_checkpoint_sha256,
      warm_start=('none; random actor/critic/normalizers and fresh Adam' if args.from_scratch
        else 'resume' if args.resume else 'actor_critic_normalizers_and_distribution; fresh_optimizer_and_course_clock'),
      motion_file=str(term.reference.path), motion_sha256=term.reference.sha256,
      recipe=runner._entry_recipe()))
    tracker.configuration(dict(curriculum=args.curriculum, seed=args.seed, num_envs=args.num_envs,
      iterations=args.iterations, bank_sha256=term.bank.sha256, motion_sha256=term.reference.sha256,
      recipe=runner._entry_recipe()))
    tracker.artifact(output/'run.json')
    def evaluate(level, *, step, phase, initialization='entries'):
      attempts = (128 if level == 2 and initialization == 'entries' else 64) if three_stage else args.eval_attempts
      report = evaluate_entries(runner.alg.actor, args.entry_bank.resolve(), device=args.device,
        level=level, attempts=attempts, seed=424242+level,
        motion_file=term.cfg.motion_file, curriculum=args.curriculum, initialization=initialization)
      report.update(updates_completed=step, phase=phase)
      path = output/f'evaluation_{step:06d}_{phase}_{initialization}_level{level}.json'
      write_json(path, report)
      tracker.evaluation(report, path, step)
      print('ENTRY_EVALUATION '+json.dumps({k: report[k] for k in
        ('updates_completed', 'phase', 'level', 'attempts', 'distinct_trajectories',
         'successes')}, allow_nan=False), flush=True)
      return report

    initial_levels = ([term.course.level] if args.baseline != 'none' else []) if three_stage else (
      range(len(LEVELS)) if args.baseline == 'all' else ([term.course.level] if args.baseline == 'current' else []))
    for level in initial_levels:
      evaluate(level, step=completed, phase='baseline', initialization='nominal' if three_stage and level == 0 else 'entries')

    while completed < args.iterations:
      count = min(args.eval_interval, args.iterations-completed)
      runner.current_learning_iteration = completed
      runner.learn(count)
      completed += count
      if runner.logger.writer is not None:
        runner.logger.writer.flush(); runner.logger.writer.close(); runner.logger.writer = None
      for model in (runner.alg.actor, runner.alg.critic):
        if not all(torch.isfinite(p).all() for p in model.parameters()):
          raise ValueError('Nonfinite network parameters after PPO update')
      level = term.course.level
      report = evaluate(level, step=completed, phase='heldout',
        initialization='nominal' if three_stage and level == 0 else 'entries')
      nominal = evaluate(level, step=completed, phase='retention', initialization='nominal') if three_stage and level > 0 else None
      promoted = (term.course.observe(report, nominal=nominal) if three_stage else
        term.course.observe(report['successes'], report['attempts'], distinct_trajectories=report['distinct_trajectories']))
      event = dict(updates_completed=completed, evaluated_level=level,
        standing_successes=report['successes'], attempts=report['attempts'],
        distinct_trajectories=report['distinct_trajectories'],
        nominal_successes=nominal['successes'] if nominal is not None else None,
        promoted=promoted, course=term.course.state_dict(), settings=term.course.settings)
      with (output/'course.jsonl').open('a') as stream:
        stream.write(json.dumps(event, allow_nan=False)+'\n')
      print('ENTRY_COURSE '+json.dumps(event), flush=True)
      tracker.course(event, output/'course.jsonl')
      runner.current_learning_iteration = completed-1
      checkpoint = output/f'model_{completed-1}.pt'
      boundary = output/f'stage{level}_promotion_{completed-1}.pt' if promoted and three_stage else checkpoint
      runner.save(str(boundary))
      if promoted and three_stage:
        # Startup events apply to the actual new model. Restore Adam, scheduler,
        # normalizers, reference sampling and update clock, never warm-start here.
        new_stage = term.course.level
        env.close()
        cfg = make_cfg(new_stage)
        env = ManagerBasedRlEnv(cfg, device=args.device)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = runner_type(wrapped, asdict(agent), str(output), device=args.device)
        runner.restore_stage_transition(boundary)
        term = runner.term
        dump_yaml(output/f'params/env_stage{new_stage}.yaml', asdict(cfg))
        write_json(output/f'stage_{new_stage}.json', runner._entry_recipe())
        # Save a checkpoint whose recipe matches the rebuilt stage for resume.
        runner.current_learning_iteration = completed-1
        runner.save(str(checkpoint))
        tracker.artifact(output/f'stage_{new_stage}.json')
      tracker.checkpoint(checkpoint, completed, force=promoted or completed == args.iterations)
      # Legacy training keeps worlds continuous; only future resets change level.
    result = dict(updates_completed=completed, course=term.course.state_dict(),
      final_checkpoint=str(output/f'model_{completed-1}.pt'), output=str(output))
    write_json(output/'finished.json', result)
    print('ENTRY_TRAINING_FINISHED '+json.dumps(result), flush=True)
    tracker.artifact(output/'course.jsonl')
    tracker.artifact(output/'finished.json')
    return result
  finally:
    writer = getattr(getattr(runner, 'logger', None), 'writer', None)
    if writer is not None:
      writer.flush(); writer.close()
    env.close()
    if tracker is not None:
      tracker.close()


if __name__ == '__main__':
  train(parse_args())
