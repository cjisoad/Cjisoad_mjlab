"""BeyondMimic PPO with a compact Go2 policy and checked checkpoint metadata."""
from __future__ import annotations

import hashlib
from dataclasses import asdict

import torch
from mjlab.rl import MjlabOnPolicyRunner

from src.tasks.pedipulation.robot import SNAPSHOT_PATH
from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def transition_ppo_runner_cfg():
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.class_name = 'TransitionOnPolicyRunner'
  cfg.actor.hidden_dims = (256, 128, 128)
  cfg.critic.hidden_dims = (256, 128, 128)
  cfg.actor.distribution_cfg['init_std'] = .5
  cfg.algorithm.learning_rate = 3e-4
  cfg.experiment_name = 'transition_t'
  cfg.logger = 'tensorboard'
  cfg.upload_model = False
  cfg.clip_actions = 10.
  return cfg


class TransitionOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
    super().__init__(env, train_cfg, log_dir, device)
    self.term = env.unwrapped.command_manager.get_term('motion')
    if (self.alg.actor.obs_dim, self.alg.critic.obs_dim,
        self.alg.actor.distribution.output_dim) != (75, 192, 12):
      raise ValueError('transition_t requires actor75, critic192 and action12')

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    # Reference-state reset already chooses a physically matching phase.
    return super().learn(num_learning_iterations, init_at_random_ep_len=False)

  def _contract(self):
    env = self.env.unwrapped
    action = env.action_manager.get_term('joint_pos')
    robot = env.scene['robot']
    robot_cfg = env.cfg.scene.entities['robot']
    # Hash compiled nominal physical arrays as well as the source snapshot.
    model = env.sim.mj_model
    digest = hashlib.sha256()
    for name in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
                 'body_pos', 'body_quat', 'jnt_range', 'jnt_axis',
                 'geom_type', 'geom_size', 'geom_pos', 'geom_quat',
                 'geom_friction', 'dof_armature', 'dof_damping'):
      digest.update(name.encode())
      digest.update(getattr(model, name).tobytes())
    return {
      'version': 2, 'task': 'transition_t',
      'reset_method': 'beyondmimic_root_joint_noise_v1',
      'observation_schema': 'beyondmimic_actor75_critic192_v1',
      'actor_terms': list(env.cfg.observations['actor'].terms),
      'critic_terms': list(env.cfg.observations['critic'].terms),
      'motion_sha256': self.term.reference.sha256,
      'joint_names': list(robot.joint_names),
      'body_names': list(self.term.cfg.body_names),
      'anchor': self.term.cfg.anchor_body_name,
      'actor_dim': self.alg.actor.obs_dim, 'critic_dim': self.alg.critic.obs_dim,
      'action_dim': self.alg.actor.distribution.output_dim,
      'action_zero': action.default_angles[0].detach().cpu().tolist(),
      'action_scale': action.cfg.scale, 'action_delay': action.cfg.delay,
      'clip_actions': self.env.clip_actions,
      'source_physics_sha256': hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest(),
      'compiled_physics_sha256': digest.hexdigest(),
      'actuators': [asdict(act) for act in robot_cfg.articulation.actuators],
      'soft_joint_pos_limit_factor': robot_cfg.articulation.soft_joint_pos_limit_factor,
      'control_dt': env.step_dt, 'simulation_dt': env.cfg.sim.mujoco.timestep,
      'adaptive_bins': self.term.bin_count,
    }

  def save(self, path, infos=None):
    sampling = {name: getattr(self.term, name).detach().cpu().clone()
                for name in ('bin_failed_count', '_current_bin_failed')}
    super().save(path, {**(infos or {}), 'transition_t_contract': self._contract(),
                        'transition_t_sampling': sampling})

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    metadata = saved.get('infos') or {}
    if metadata.get('transition_t_contract') != self._contract():
      raise ValueError('Checkpoint transition_t contract differs in reference, observation, action or physics')
    sampling = metadata.get('transition_t_sampling', {})
    for name in ('bin_failed_count', '_current_bin_failed'):
      value = sampling.get(name)
      target = getattr(self.term, name)
      if (not isinstance(value, torch.Tensor) or value.shape != target.shape
          or not torch.isfinite(value).all() or (value < 0).any()):
        raise ValueError(f'Invalid transition_t sampling state: {name}')
    # RSL-RL assigns normalizer._std inside inference-mode rollouts. Convert
    # those buffers back before load_state_dict performs its in-place copy.
    with torch.inference_mode(False):
      for network in (self.alg.actor, self.alg.critic):
        for module in network.modules():
          for name, value in module.named_buffers(recurse=False):
            if value.is_inference():
              setattr(module, name, value.clone())
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    for name, value in sampling.items():
      if name in ('bin_failed_count', '_current_bin_failed'):
        getattr(self.term, name).copy_(value)
    return infos
