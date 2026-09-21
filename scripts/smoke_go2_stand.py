"""Bounded CPU validation that does not contend with an active GPU training job."""

import os

# Set before importing torch/Warp; this tool never selects a GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/go2_stand_mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/go2_stand_cache')

import argparse
from dataclasses import asdict
from pathlib import Path
import tempfile

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.stand.env_cfg import stand_env_cfg
from src.tasks.stand.rl import StandOnPolicyRunner, stand_ppo_runner_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--steps', type=int, default=16)
  parser.add_argument('--ppo', action='store_true')
  args = parser.parse_args()
  if not 1 <= args.steps <= 64:
    parser.error('--steps must be between 1 and 64')
  torch.set_num_threads(1)
  cfg = stand_env_cfg()
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
    print(f'CPU rollout passed: 2 worlds, {args.steps} steps, 45/86 observations, timeout/reset')
    if args.ppo:
      runner_cfg = stand_ppo_runner_cfg()
      # Keep networks and all update hyperparameters; only shorten rollout.
      runner_cfg.num_steps_per_env = 4
      with tempfile.TemporaryDirectory(prefix='go2_stand_smoke_') as directory:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=100.)
        runner = StandOnPolicyRunner(wrapped, asdict(runner_cfg), directory, 'cpu')
        runner.learn(1, init_at_random_ep_len=False)
        checkpoint = Path(directory) / f'model_{runner.current_learning_iteration}.pt'
        assert checkpoint.exists()
        runner.load(str(checkpoint))
        print('CPU PPO update, checkpoint and reload passed')
  finally:
    env.close()


if __name__ == '__main__':
  main()
