"""Compare every teacher reward at fixed states under old/new anchor bases.

Run from the repository with its Python environment and PYTHONPATH=.:
  MUJOCO_GL=disable python scripts/audit_teacher_anchor.py --device cuda:0 \
    --output /tmp/teacher_anchor_audit.json

This checks coordinate effects, not learned policy performance. Physical DR is
retained and each old/new comparison uses the same simulator state and command.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import PropertyMock, patch

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, euler_xyz_from_quat

import src.tasks  # noqa: F401
from src.tasks.leg_manip.mdp.anchor_frame import anchor_basis as old_basis
from src.tasks.teacher_common.anchor_frame import anchor_degenerate_fraction
from src.tasks.teacher_common.commands import PedipulationTeacherCommand, LocoPedipulationTeacherCommand
from src.tasks.pedipulation.constants import DESIRED_ANGLES
from src.tasks.pedipulation.env_cfg import pedipulation_env_cfg
from src.tasks.loco_pedipulation.env_cfg import loco_pedipulation_env_cfg
from src.tasks.teacher_common.rewards import source_reward_function


def reward_values(env):
  # The source biped's base_height mutates one batch-wide height gate.
  # Restore that history between comparisons, including legacy source terms.
  action = env.action_manager.get_term('joint_pos')
  previous_gate = action.height_score.clone()
  try:
    return {name: cfg.func(env, **cfg.params).detach().clone()
            for name, cfg in zip(env.reward_manager.active_terms, env.reward_manager._term_cfgs)}
  finally:
    action.height_score = previous_gate


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--num-envs', type=int, default=64)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(1)
  euler_check = {}
  for label, pitch in (('exact_upright', -torch.pi/2.), ('near_upright', -torch.pi/2.+.01)):
    q = quat_from_euler_xyz(torch.full((args.num_envs,), .25, device=args.device),
      torch.full((args.num_envs,), pitch, device=args.device),
      torch.full((args.num_envs,), .6, device=args.device))
    repeated = [euler_xyz_from_quat(q)[0].abs().mean().item() for _ in range(8)]
    euler_check[label] = {'repeated_roll_abs': repeated,
                         'max_repeat_difference': max(repeated)-min(repeated)}
  records = []
  for task, command_cls, source_factory in (
      ('pedipulation_t', PedipulationTeacherCommand, pedipulation_env_cfg),
      ('loco_pedipulation_t', LocoPedipulationTeacherCommand, loco_pedipulation_env_cfg)):
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = args.num_envs
    cfg.commands['twist'].debug_vis = False
    original = source_factory()
    removed_source_rewards = []
    if task == 'pedipulation_t':
      assert 'ang_xz' not in cfg.rewards
      original.rewards.pop('ang_xz')
      removed_source_rewards.append('ang_xz')
    assert tuple(cfg.rewards) == tuple(original.rewards)
    for name, before in original.rewards.items():
      after = cfg.rewards[name]
      assert after.weight == before.weight, name
      params = dict(after.params)
      if task == 'pedipulation_t':
        assert source_reward_function(params.pop('reward_name')) is before.func, name
      else:
        assert after.func is before.func, name
      assert params == before.params, name
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
      env.reset()
      robot = env.scene['robot']
      term = env.command_manager.get_term('twist')
      ids = torch.arange(env.num_envs, device=env.device)
      groups = ids % 4
      rolls = torch.tensor((0., .3, .3, .25), device=env.device)[groups]
      # Keep the same near-upright comparison poses as the original anchor
      # audit. The legacy ang_xz reward is now excluded from the teacher.
      pitches = torch.tensor((0., 0., -.5, -torch.pi/2.+.01), device=env.device)[groups]
      yaws = torch.full((env.num_envs,), .6, device=env.device)
      state = robot.data.default_root_state.clone()
      state[:, :3] = env.scene.env_origins
      state[:, 2] += .52 if task == 'pedipulation_t' else .4
      state[:, 3:7] = quat_from_euler_xyz(rolls, pitches, yaws)
      state[:, 7:10] = torch.tensor((.3, -.1, .04), device=env.device)
      state[:, 10:13] = torch.tensor((.2, -.3, .15), device=env.device)
      robot.write_root_state_to_sim(state)
      if task == 'pedipulation_t':
        q = torch.tensor(DESIRED_ANGLES, device=env.device).expand(env.num_envs, 12)
      else:
        q = robot.data.default_joint_pos.clone()
      robot.write_joint_state_to_sim(q, torch.zeros_like(q))
      env.scene.write_data_to_sim()
      env.sim.forward()
      env.sim.sense()
      env.scene.update(dt=env.physics_dt)
      term.set_command(ids, velocity=(.2, -.03, .15), target_offset=(.04, .01, .30))
      term._command[:, :3] = torch.tensor((.2, -.03, .15), device=env.device)
      env.action_manager.get_term('joint_pos').height_score.fill_(1.)
      if task == 'loco_pedipulation_t':
        from src.tasks.loco_pedipulation.mdp.commands import HOLD
        term.phase.fill_(HOLD)
        term.blend.fill_(1.)
        term.reference[:] = term.zero + term.command[:, 3:]
      new = reward_values(env)
      new_goal = term.foot_target_pos_w.detach().clone()
      old_frame = old_basis(robot.data.root_link_quat_w, robot.data.gravity_vec_w)
      with patch.object(command_cls, 'basis_w', new_callable=PropertyMock, return_value=old_frame):
        old = reward_values(env)
        old_goal = term.foot_target_pos_w.detach().clone()
      effects = {}
      for name in new:
        assert torch.isfinite(new[name]).all() and torch.isfinite(old[name]).all(), name
        difference = (new[name] - old[name]).abs()
        effects[name] = {'weight': cfg.rewards[name].weight,
          'max_abs_change': difference.max().item(),
          'mean_old': old[name].mean().item(), 'mean_new': new[name].mean().item()}
      allowed = ({'tracking_lin_vel', 'FR_pos_track'} if task == 'pedipulation_t'
                 else {'track_linear_velocity', 'fr_position_tracking'})
      unexpected = {name: value for name, value in effects.items()
                    if name not in allowed and value['max_abs_change'] > 2e-5}
      assert not unexpected, unexpected
      # Also exercise FR tracking near its optimum. Far-away nominal targets
      # can make both exponentials nearly zero and hide a coordinate effect.
      fr_id = robot.find_sites(('FR',), preserve_order=True)[0][0]
      actual_foot = robot.data.site_pos_w[:, fr_id]
      foot_anchor = (term.basis_w.transpose(-1, -2)
                     @ (actual_foot - robot.data.root_link_pos_w)[..., None]).squeeze(-1)
      near_offset = foot_anchor - term.zero + torch.tensor((.01, -.005, .008), device=env.device)
      term.set_command(ids, target_offset=near_offset)
      if task == 'loco_pedipulation_t':
        term.reference[:] = term.zero + term.command[:, 3:]
      near_new = reward_values(env)
      with patch.object(command_cls, 'basis_w', new_callable=PropertyMock, return_value=old_frame):
        near_old = reward_values(env)
      for name in near_new:
        if name not in allowed:
          torch.testing.assert_close(near_new[name], near_old[name], atol=2e-5, rtol=0.)
      fr_name = 'FR_pos_track' if task == 'pedipulation_t' else 'fr_position_tracking'
      near_foot_effect = {'max_abs_change': (near_new[fr_name]-near_old[fr_name]).abs().max().item(),
                         'mean_old': near_old[fr_name].mean().item(),
                         'mean_new': near_new[fr_name].mean().item()}
      assert near_foot_effect['max_abs_change'] > .01, near_foot_effect
      # Stopping costs depend only on horizontal norms and must remain invariant.
      term._command[:, :3].zero_()
      zero_new = reward_values(env)
      with patch.object(command_cls, 'basis_w', new_callable=PropertyMock, return_value=old_frame):
        zero_old = reward_values(env)
      linear_name = 'tracking_lin_vel' if task == 'pedipulation_t' else 'track_linear_velocity'
      torch.testing.assert_close(zero_new[linear_name], zero_old[linear_name], atol=2e-5, rtol=0.)
      heading_effect = None
      if task == 'pedipulation_t':
        term.manual.zero_()
        term.heading_target.zero_()
        new_wz = term.command[:, 2].detach().clone()
        with patch.object(command_cls, 'basis_w', new_callable=PropertyMock, return_value=old_frame):
          old_wz = term.command[:, 2].detach().clone()
        heading_effect = {'max_wz_change_rad_s': (new_wz-old_wz).abs().max().item(),
                          'new_wz_pure_quad_roll': new_wz[groups == 1].mean().item(),
                          'old_wz_pure_quad_roll': old_wz[groups == 1].mean().item()}
      record = {'task': task, 'device': args.device, 'num_envs': env.num_envs,
        'pose_groups': ['level', 'quad_roll', 'intermediate_roll', 'near_upright_roll'],
        'reward_count': len(effects), 'removed_source_rewards': removed_source_rewards,
        'remaining_weights_parameters_order_preserved': True,
        'state_and_command_fixed_for_reward_comparison': True,
        'allowed_direct_changes': sorted(allowed), 'unexpected_changed_rewards': unexpected,
        'zero_command_linear_reward_invariant': True,
        'foot_goal_max_change_m': (new_goal-old_goal).norm(dim=-1).max().item(),
        'foot_tracking_near_target': near_foot_effect,
        'heading_control_effect': heading_effect,
        'fallback_fraction': anchor_degenerate_fraction(env).mean().item(),
        'rewards': effects}
      records.append(record)
      print('REWARD_AUDIT_PASSED', task, len(effects), flush=True)
    finally:
      env.close()
      if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
  result = {'anchor': 'teacher_body_y_cross_up_v1', 'tasks': records,
            'existing_euler_upright_check_without_anchor': euler_check,
            'limitation': 'Fixed-state comparison verifies coordinate effects, not policy learning or global frame continuity.'}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps({r['task']: {'rewards': r['reward_count'], 'unexpected_changes': r['unexpected_changed_rewards']}
                    for r in records}, indent=2), flush=True)


if __name__ == '__main__':
  main()
