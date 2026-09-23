"""Bounded rollout, transition, PPO, checkpoint and ONNX checks."""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
from dataclasses import asdict
from pathlib import Path
import tempfile

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.loco_pedipulation.env_cfg import loco_pedipulation_env_cfg
from src.tasks.loco_pedipulation.mdp.commands import HOLD, QUAD
from src.tasks.loco_pedipulation.rl import (
  LocoPedipulationOnPolicyRunner, loco_pedipulation_ppo_runner_cfg,
)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--ppo', action='store_true')
  args = parser.parse_args()
  torch.set_num_threads(1)
  cfg = loco_pedipulation_env_cfg()
  cfg.scene.num_envs = 2
  cfg.events.pop('push_robot', None)
  command_cfg = cfg.commands['twist']
  command_cfg.initial_stage = 3
  command_cfg.prepare_time = .04
  command_cfg.reach_time = .12
  command_cfg.recover_time = .04
  command_cfg.resampling_time_range = (100., 100.)
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    obs, _ = env.reset()
    assert obs['actor'].shape == (2, 65)
    assert obs['critic'].shape == (2, 92)
    term = env.command_manager.get_term('twist')
    term.set_command([0, 1], velocity=(0., 0., 0.), fr_enabled=False)
    term.set_command([0], fr_enabled=True, target_offset=(0., 0., .07))
    actions = torch.zeros(2, 12, device=args.device)
    for _ in range(14):
      obs, reward, terminated, _, _ = env.step(actions)
      assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all()
      assert torch.isfinite(reward).all()
      assert not terminated.any()
    assert term.phase[0] == HOLD and term.phase[1] == QUAD
    assert term.blend[0] == 1. and term.blend[1] == 0.
    term.set_command([0], fr_enabled=False)
    for _ in range(14):
      env.step(actions)
    assert term.phase[0] == QUAD, 'Landing contact did not release the manipulation state'
    assert term.stats[0, 9] == 1.
    other = (term.command[1].clone(), term.phase[1].clone(), term.reference[1].clone())
    env.reset(env_ids=torch.tensor([0], device=args.device))
    for actual, expected in zip((term.command[1], term.phase[1], term.reference[1]), other):
      torch.testing.assert_close(actual, expected)
    env.episode_length_buf[:] = env.max_episode_length
    _, _, _, truncated, _ = env.step(actions)
    assert truncated.all()
    assert (env.episode_length_buf == 0).all()
    print(f'PASS: {args.device} rollout, 65/92 observations, transitions, partial reset and timeout')

    if args.ppo:
      agent_cfg = loco_pedipulation_ppo_runner_cfg()
      agent_cfg.num_steps_per_env = 4
      with tempfile.TemporaryDirectory(prefix='loco_pedipulation_smoke_') as directory:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = LocoPedipulationOnPolicyRunner(wrapped, asdict(agent_cfg), directory, args.device)
        term.stage = 2
        runner.learn(1, init_at_random_ep_len=False)
        runner.logger.writer.flush()
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(directory).Reload()
        for tag in ('landing_attempts_total', 'landing_confirmed_total',
                    'tripod_stationary_samples', 'tripod_moving_samples'):
          assert f'Metrics/twist/{tag}' in events.Tags()['scalars'], tag
        checkpoint = Path(directory) / f'model_{runner.current_learning_iteration}.pt'
        assert checkpoint.exists()
        with torch.no_grad():
          sample = wrapped.get_observations()
          policy = runner.get_inference_policy()
          expected = policy(sample).clone()
        term.stage = 0
        # A fresh runner exercises the same restore path as train.py --resume.
        restored = LocoPedipulationOnPolicyRunner(wrapped, asdict(agent_cfg), device=args.device)
        restored.load(str(checkpoint), map_location=args.device)
        assert term.stage == 2, 'Curriculum stage was not restored'
        torch.testing.assert_close(restored.logger.landing_totals[0], runner.logger.landing_totals[0])
        with torch.no_grad():
          torch.testing.assert_close(restored.get_inference_policy()(sample), expected)
        import onnx
        onnx.checker.check_model(str(Path(directory) / 'policy.onnx'))
        print('PASS: PPO update, checkpoint/policy/curriculum restoration and ONNX export')
  finally:
    env.close()


if __name__ == '__main__':
  main()
