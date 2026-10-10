"""Fine tune the actual transition actor on a graded physical entry bank."""
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
  parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument('--resume', action='store_true')
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--num-envs', type=int, default=4096)
  parser.add_argument('--iterations', type=int, default=5000, help='Total updates in this new course run')
  parser.add_argument('--eval-interval', type=int, default=250)
  parser.add_argument('--eval-attempts', type=int, default=64)
  parser.add_argument('--baseline', choices=('all', 'current', 'none'), default='all')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--output', type=Path)
  args = parser.parse_args(argv)
  for name in ('num_envs', 'iterations', 'eval_interval', 'eval_attempts'):
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

  for path in (args.entry_bank, args.checkpoint):
    if not path.is_file():
      raise FileNotFoundError(path)
  output = args.output or ROOT/'logs/rsl_rl/transition_t_entry'/datetime.now().strftime('%Y-%m-%d_%H-%M-%S_teacher14999')
  if output.exists() and any(output.iterdir()):
    raise ValueError(f'Use a fresh output directory: {output}')
  output.mkdir(parents=True, exist_ok=True)
  configure_torch_backends(); torch.set_num_threads(1)
  cfg = entry_course_env_cfg(args.entry_bank.resolve(), num_envs=args.num_envs)
  cfg.seed = args.seed
  agent = transition_ppo_runner_cfg()
  agent.seed = args.seed
  agent.experiment_name = 'transition_t_entry'
  agent.save_interval = min(100, args.eval_interval)
  env = ManagerBasedRlEnv(cfg, device=args.device)
  runner = None
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = EntryOnPolicyRunner(wrapped, asdict(agent), str(output), device=args.device)
    if args.resume:
      runner.resume(args.checkpoint)
    else:
      runner.warm_start(args.checkpoint)
    completed = runner.current_learning_iteration
    if completed >= args.iterations:
      raise ValueError(f'Resumed checkpoint already has {completed} updates, target is {args.iterations}')
    term = runner.term
    dump_yaml(output/'params/env.yaml', asdict(cfg))
    dump_yaml(output/'params/agent.yaml', asdict(agent))
    write_json(output/'run.json', dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
      output=str(output), bank_sha256=term.bank.sha256, bank_coverage=term.bank.coverage(),
      seed_checkpoint_sha256=runner.seed_checkpoint_sha256,
      warm_start='actor_critic_normalizers_and_distribution; fresh_optimizer_and_course_clock',
      recipe=runner._entry_recipe()))
    def evaluate(level, *, step, phase):
      report = evaluate_entries(runner.alg.actor, args.entry_bank.resolve(), device=args.device,
        level=level, attempts=args.eval_attempts, seed=424242+level)
      report.update(updates_completed=step, phase=phase)
      write_json(output/f'evaluation_{step:06d}_{phase}_level{level}.json', report)
      print('ENTRY_EVALUATION '+json.dumps({k: report[k] for k in
        ('updates_completed', 'phase', 'level', 'attempts', 'distinct_trajectories',
         'successes')}, allow_nan=False), flush=True)
      return report

    initial_levels = range(len(LEVELS)) if args.baseline == 'all' else ([term.course.level] if args.baseline == 'current' else [])
    for level in initial_levels:
      evaluate(level, step=completed, phase='baseline')

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
      report = evaluate(level, step=completed, phase='heldout')
      promoted = term.course.observe(report['successes'], report['attempts'],
        distinct_trajectories=report['distinct_trajectories'])
      event = dict(updates_completed=completed, evaluated_level=level,
        standing_successes=report['successes'], attempts=report['attempts'],
        distinct_trajectories=report['distinct_trajectories'],
        promoted=promoted, course=term.course.state_dict(), settings=term.course.settings)
      with (output/'course.jsonl').open('a') as stream:
        stream.write(json.dumps(event, allow_nan=False)+'\n')
      print('ENTRY_COURSE '+json.dumps(event), flush=True)
      runner.current_learning_iteration = completed-1
      runner.save(str(output/f'model_{completed-1}.pt'))
      # Keep trained worlds continuous; only future resets use the new level.
    result = dict(updates_completed=completed, course=term.course.state_dict(),
      final_checkpoint=str(output/f'model_{completed-1}.pt'), output=str(output))
    write_json(output/'finished.json', result)
    print('ENTRY_TRAINING_FINISHED '+json.dumps(result), flush=True)
    return result
  finally:
    writer = getattr(getattr(runner, 'logger', None), 'writer', None)
    if writer is not None:
      writer.flush(); writer.close()
    env.close()


if __name__ == '__main__':
  train(parse_args())
