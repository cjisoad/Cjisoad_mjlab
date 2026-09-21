"""Bounded CPU rollout and PPO validation for the Pedipulation task."""

import os

# Set before importing torch/Warp; this tool never selects a GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pedipulation_mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/pedipulation_cache')

import argparse
from dataclasses import asdict
from pathlib import Path
import tempfile

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.pedipulation.env_cfg import pedipulation_env_cfg
from src.tasks.pedipulation.rl import PedipulationOnPolicyRunner, pedipulation_ppo_runner_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--steps', type=int, default=16)
  parser.add_argument('--ppo', action='store_true')
  args = parser.parse_args()
  if not 1 <= args.steps <= 64:
    parser.error('--steps must be between 1 and 64')
  torch.set_num_threads(1)
  cfg = pedipulation_env_cfg()
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device='cpu')
  try:
    obs, _ = env.reset()
    assert obs['actor'].shape == (2, 45)
    assert obs['critic'].shape == (2, 86)
    torch.testing.assert_close(obs['critic'][:, 3:48], obs['actor'])
    state = env.action_manager.get_term('joint_pos')
    robot = env.scene['robot']
    dr = state.dr_observation.clone()
    assert (dr[:, 0] == 0).all()
    torch.testing.assert_close(dr[:, 32], dr[:, 33])
    torch.testing.assert_close(env.sim.model.dof_armature[:, robot.indexing.joint_v_adr], dr[:, 29:30].expand(-1, 12))
    torch.testing.assert_close(env.sim.model.dof_frictionloss[:, robot.indexing.joint_v_adr], dr[:, 30:31].expand(-1, 12))
    torch.testing.assert_close(env.sim.model.dof_damping[:, robot.indexing.joint_v_adr], dr[:, 31:32].expand(-1, 12))
    for index, actuator in enumerate(robot.actuators):
      torch.testing.assert_close(actuator.stiffness, 40 * dr[:, 5 + index:6 + index])
      torch.testing.assert_close(actuator.damping, dr[:, 17 + index:18 + index])
    for _ in range(args.steps):
      obs, reward, terminated, truncated, info = env.step(torch.zeros(2, 12))
      assert torch.isfinite(obs['actor']).all()
      assert torch.isfinite(obs['critic']).all()
      assert torch.isfinite(reward).all()
      torch.testing.assert_close(obs['critic'][:, 3:48], obs['actor'])
    env.episode_length_buf[:] = env.max_episode_length
    _, _, _, truncated, _ = env.step(torch.zeros(2, 12))
    assert truncated.all(), 'Source timeout boundary did not reset both worlds'
    assert (env.episode_length_buf == 0).all()
    torch.testing.assert_close(state.dr_observation, dr)
    # Reset one world without modifying the other world's command or frame.
    command = env.command_manager.get_term('stand')
    other_command = command.command[1].clone()
    other_frame = command.frame.basis_w[1].clone()
    env.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(command.command[1], other_command)
    torch.testing.assert_close(command.frame.basis_w[1], other_frame)
    torch.testing.assert_close(command.frame.basis_w.transpose(1, 2) @ command.frame.basis_w,
      torch.eye(3).repeat(2, 1, 1), atol=1e-6, rtol=1e-6)
    print(f'CPU rollout passed: 2 worlds, {args.steps} steps, 45/86 observations, timeout/reset')
    if args.ppo:
      runner_cfg = pedipulation_ppo_runner_cfg()
      runner_cfg.logger = 'tensorboard'
      runner_cfg.upload_model = False
      # Keep networks and all update hyperparameters; only shorten rollout.
      runner_cfg.num_steps_per_env = 4
      with tempfile.TemporaryDirectory(prefix='pedipulation_smoke_') as directory:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=100.)
        runner = PedipulationOnPolicyRunner(wrapped, asdict(runner_cfg), directory, 'cpu')
        runner.learn(1, init_at_random_ep_len=False)
        checkpoint = Path(directory) / f'model_{runner.current_learning_iteration}.pt'
        assert checkpoint.exists()
        with torch.no_grad():
          sample = wrapped.get_observations()
          expected_action = runner.get_inference_policy()(sample).clone()
          next(runner.alg.actor.parameters()).add_(.1)
        runner.load(str(checkpoint))
        with torch.no_grad():
          torch.testing.assert_close(runner.get_inference_policy()(sample), expected_action)
        runner.export_policy_to_onnx(directory)
        import onnx
        onnx.checker.check_model(str(Path(directory) / 'policy.onnx'))
        print('CPU PPO update, checkpoint restoration and ONNX export passed')
  finally:
    env.close()


if __name__ == '__main__':
  main()
