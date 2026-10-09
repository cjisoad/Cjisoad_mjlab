"""Evaluate a transition checkpoint on held-out physical teacher entries."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--entry-bank', type=Path, required=True)
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--level', choices=('all', '0', '1', '2', '3'), default='all')
  parser.add_argument('--attempts', type=int, default=64)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--seed', type=int, default=424242)
  parser.add_argument('--allow-legacy-transition-limiter', action='store_true',
    help='Evaluate an old unlimited transition policy under new execution limits')
  parser.add_argument('--output', type=Path, default=ROOT/'outputs/transition_entries/evaluation.json')
  args = parser.parse_args(argv)
  if args.attempts < 1:
    parser.error('--attempts must be positive')
  return args


def evaluate(args):
  os.environ.setdefault('OMP_NUM_THREADS', '1')
  os.environ.setdefault('MPLCONFIGDIR', '/tmp/transition_t_mpl')
  os.environ.setdefault('XDG_CACHE_HOME', '/tmp/transition_t_cache')
  sys.path.insert(0, str(ROOT))
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from src.tasks.transition_t.entry_bank import LEVELS
  from src.tasks.transition_t.entry_evaluation import evaluate_entries, _copy_actor_for_eval
  from src.tasks.transition_t.env_cfg import transition_env_cfg
  from src.tasks.transition_t.rl import TransitionOnPolicyRunner, transition_ppo_runner_cfg

  for path in (args.entry_bank, args.checkpoint):
    if not path.is_file():
      raise FileNotFoundError(path)
  if args.output.exists():
    raise ValueError(f'Use a fresh evaluation output: {args.output}')
  configure_torch_backends(); torch.set_num_threads(1)
  cfg = transition_env_cfg(play=True)
  cfg.commands['motion'].debug_vis = False
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    agent = transition_ppo_runner_cfg()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = TransitionOnPolicyRunner(wrapped, asdict(agent), device=args.device)
    runner.load(str(args.checkpoint), load_cfg={'actor': True}, map_location=args.device,
      allow_legacy_target_limiter=args.allow_legacy_transition_limiter)
    policy = _copy_actor_for_eval(runner.alg.actor, args.device)
  finally:
    env.close()
  levels = range(len(LEVELS)) if args.level == 'all' else [int(args.level)]
  report = dict(checkpoint=str(args.checkpoint.resolve()),
    legacy_limiter_experiment=args.allow_legacy_transition_limiter,
    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), levels={})
  args.output.parent.mkdir(parents=True, exist_ok=True)
  for level in levels:
    result = evaluate_entries(policy, args.entry_bank, level=level, device=args.device,
      attempts=args.attempts, seed=args.seed+level)
    report['levels'][str(level)] = result
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('ENTRY_BASELINE '+json.dumps({key: result[key] for key in
      ('level', 'attempts', 'distinct_trajectories', 'full_motion_completions',
       'successes', 'strict_successes', 'failure_reasons')}), flush=True)
  return report


if __name__ == '__main__':
  evaluate(parse_args())
