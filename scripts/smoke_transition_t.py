"""Bounded real-simulator and PPO/checkpoint validation for transition_t."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/transition_t_mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/transition_t_cache')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--num-envs', type=int, default=2)
  parser.add_argument('--steps', type=int, default=16)
  parser.add_argument('--ppo-updates', type=int, default=1)
  parser.add_argument('--output', type=Path, default=Path('outputs/transition_t_smoke'))
  args = parser.parse_args()
  if args.num_envs < 2 or args.steps < 1 or args.ppo_updates < 0:
    parser.error('Require >=2 environments, >=1 step and >=0 PPO updates')
  if args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from src.tasks.transition_t.env_cfg import transition_env_cfg
  from src.tasks.transition_t.rl import TransitionOnPolicyRunner, transition_ppo_runner_cfg
  torch.set_num_threads(1)
  args.output.mkdir(parents=True, exist_ok=True)
  cfg = transition_env_cfg()
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    obs, _ = env.reset()
    term = env.command_manager.get_term('motion')
    assert obs['actor'].shape == (args.num_envs, 75)
    assert obs['critic'].shape == (args.num_envs, 192)
    # A fresh reset must not retain relative targets from the prior world pose.
    robot = env.scene['robot']
    root = torch.cat((robot.data.root_link_pos_w, robot.data.root_link_quat_w,
                      robot.data.root_link_lin_vel_w, robot.data.root_link_ang_vel_w), -1).clone()
    root[:, :2] += root.new_tensor([1., 2.])
    robot.write_root_state_to_sim(root)
    env.sim.forward()
    obs, _ = env.reset()
    relative_before = term.body_pos_relative_w.clone()
    quat_before = term.body_quat_relative_w.clone()
    term._refresh_relative()
    torch.testing.assert_close(relative_before, term.body_pos_relative_w, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(quat_before, term.body_quat_relative_w, atol=1e-5, rtol=1e-5)
    action_term = env.action_manager.get_term('joint_pos')
    state = (env.sim.data.qpos.clone(), env.sim.data.qvel.clone(), action_term.raw_action.clone())
    term.start_from_current_state(torch.tensor([0], device=args.device))
    for before, after in zip(state, (env.sim.data.qpos.clone(), env.sim.data.qvel.clone(), action_term.raw_action)):
      torch.testing.assert_close(before, after, atol=0., rtol=0.)
    for _ in range(args.steps):
      obs, reward, *_ = env.step(torch.zeros(args.num_envs, 12, device=args.device))
      assert torch.isfinite(reward).all()
      assert all(torch.isfinite(obs[key]).all() for key in ('actor', 'critic'))
    record = dict(device=args.device, num_envs=args.num_envs, steps=args.steps,
                  actor_dim=75, critic_dim=192, action_dim=12,
                  motion_sha256=term.reference.sha256, finite_rollout=True,
                  fresh_reset_targets=True, state_preserving_entry=True)
    if args.ppo_updates:
      agent = transition_ppo_runner_cfg()
      agent.num_steps_per_env = 8
      agent.algorithm.num_learning_epochs = 1
      agent.algorithm.num_mini_batches = 2
      wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
      runner = TransitionOnPolicyRunner(wrapped, asdict(agent), str(args.output), args.device)
      before = [p.detach().clone() for p in runner.alg.actor.parameters()]
      runner.learn(args.ppo_updates)
      assert any(not torch.equal(a, b) for a, b in zip(before, runner.alg.actor.parameters()))
      checkpoint = args.output / 'roundtrip.pt'
      runner.save(str(checkpoint))
      sample = wrapped.get_observations()
      runner.alg.actor.eval()
      expected = runner.alg.actor(sample).detach().clone()
      with torch.no_grad():
        next(runner.alg.actor.parameters()).add_(.1)
      runner.load(str(checkpoint))
      torch.testing.assert_close(runner.alg.actor(sample), expected)
      wrapped.clip_actions = 5.
      try:
        runner.load(str(checkpoint))
      except ValueError:
        pass
      else:
        raise AssertionError('Changed action clipping must reject checkpoint')
      record.update(actor_updated=True, checkpoint_roundtrip=True,
                    incompatible_action_rejected=True, updates=args.ppo_updates,
                    checkpoint=str(checkpoint.resolve()))
    (args.output / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print('TRANSITION_T_SMOKE_PASSED', json.dumps(record), flush=True)
  finally:
    env.close()


if __name__ == '__main__':
  main()
