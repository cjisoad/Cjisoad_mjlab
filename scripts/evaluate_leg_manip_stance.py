"""Evaluate a version2 checkpoint on32 fixed trials for each nominal stance."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.leg_manip.env_cfg import leg_manip_env_cfg
from src.tasks.leg_manip.rl import LegManipOnPolicyRunner, leg_manip_ppo_runner_cfg
from src.tasks.leg_manip.rl.stance_evaluation import evaluate_stances

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('checkpoint', type=Path)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--seed', type=int, default=100542)
  parser.add_argument('--output', type=Path)
  args = parser.parse_args()
  torch.set_num_threads(1)
  cfg = leg_manip_env_cfg(); cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    agent = leg_manip_ppo_runner_cfg(); agent.upload_model = False; agent.logger = 'tensorboard'
    runner = LegManipOnPolicyRunner(RslRlVecEnvWrapper(env, clip_actions=10.), asdict(agent), device=args.device)
    runner.load(str(args.checkpoint), map_location=args.device)
    results = evaluate_stances(runner.alg.actor, cfg, args.device, args.seed)
    payload = {'seed': args.seed, 'checkpoint': str(args.checkpoint), 'stances': results,
      'passed': all(row['hold_fraction'] >= .7 and row['survival'] >= .85 for row in results.values())}
    encoded = json.dumps(payload, indent=2)
    if args.output:
      args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(encoded + '\n')
    print(encoded)
  finally:
    env.close()

if __name__ == '__main__': main()
