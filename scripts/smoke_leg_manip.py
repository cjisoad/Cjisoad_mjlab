"""Historic velocity-migration smoke for the former leg_manip curriculum.

The current runner rejects those former curriculum checkpoints. For a fresh
stance/walk/operation course use scripts/smoke_leg_manip_course.py instead.
This file is retained as a record of the earlier migration validation.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.leg_manip.env_cfg import leg_manip_env_cfg
from src.tasks.leg_manip.mdp.observations import base_velocity
from src.tasks.leg_manip.rl import LegManipOnPolicyRunner, leg_manip_ppo_runner_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--num-envs', type=int, default=2)
  parser.add_argument('--iterations', type=int, default=1)
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--result-file', type=Path)
  args = parser.parse_args()
  torch.set_num_threads(1)
  cfg = leg_manip_env_cfg()
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    agent = leg_manip_ppo_runner_cfg()
    agent.logger = 'tensorboard'
    agent.upload_model = False
    agent.save_interval = 1
    # CPU checks need only one minibatch; GPU benchmark uses training defaults.
    if args.device == 'cpu':
      agent.num_steps_per_env = 4
      agent.algorithm.num_learning_epochs = 1
      agent.algorithm.num_mini_batches = 1
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    with tempfile.TemporaryDirectory(prefix='leg_manip_velocity_smoke_', dir='/tmp') as directory:
      runner = LegManipOnPolicyRunner(wrapped, asdict(agent), directory, args.device)
      runner.load(str(args.checkpoint), map_location=args.device)
      term = env.command_manager.get_term('twist')
      initial_stage = term.stage
      assert initial_stage == 2
      assert not runner.alg.optimizer.state
      obs, _ = env.reset()
      assert obs['actor'].shape == (args.num_envs, 51)
      assert obs['critic'].shape == (args.num_envs, 131)
      torch.testing.assert_close(obs['critic'][:, :48], obs['actor'][:, :48])
      torch.testing.assert_close(obs['critic'][:, 48:51], base_velocity(env))
      noise_error = float((obs['actor'][:, 48:51] - obs['critic'][:, 48:51]).abs().max())
      assert noise_error <= .040001
      start = time.monotonic()
      runner.learn(args.iterations, init_at_random_ep_len=False)
      elapsed = time.monotonic() - start
      assert runner.alg.optimizer.state
      assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
      assert all(torch.isfinite(p).all() for p in runner.alg.critic.parameters())
      sample = wrapped.get_observations()
      with torch.inference_mode():
        expected = runner.get_inference_policy()(sample).clone()
      checkpoint = Path(directory) / f'model_{runner.current_learning_iteration}.pt'
      saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
      assert saved['infos']['leg_manip_observations']['actor_dim'] == 51
      restored = LegManipOnPolicyRunner(wrapped, asdict(agent), device=args.device)
      restored.load(str(checkpoint), map_location=args.device)
      with torch.inference_mode():
        torch.testing.assert_close(restored.get_inference_policy()(sample), expected)
      runner.export_policy_to_onnx(directory)
      import onnx
      model = onnx.load(str(Path(directory) / 'policy.onnx'))
      onnx.checker.check_model(model)
      assert model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value == 51
      result = {'device': args.device, 'num_envs': args.num_envs,
        'actor_dim': 51, 'critic_dim': 131, 'max_scaled_noise_error': noise_error,
        'precise_critic_velocity_verified': True, 'initial_stage': initial_stage,
        'ppo_updates': args.iterations, 'ppo_elapsed_seconds': elapsed,
        'optimizer_update_reload_verified': True, 'onnx_input_dim': 51,
        'stage_after_smoke': term.stage}
      if args.result_file:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(result, indent=2) + '\n')
      print(json.dumps(result, indent=2), flush=True)
  finally:
    env.close()


if __name__ == '__main__':
  main()
