"""Plain PPO with motion/action compatibility checks and course persistence."""
import hashlib

import torch
from mjlab.rl import MjlabOnPolicyRunner
from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg
from src.tasks.pedipulation.robot import SNAPSHOT_PATH
from src.tasks.teacher_common.anchor_frame import ANCHOR_CONTRACT


def transition_ppo_runner_cfg():
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.class_name = 'TransitionOnPolicyRunner'
  cfg.actor.obs_normalization = False
  cfg.critic.obs_normalization = False
  cfg.actor.distribution_cfg['init_std'] = .5
  cfg.algorithm.learning_rate = 3e-4
  cfg.algorithm.entropy_coef = .005
  cfg.algorithm.gamma = .995
  cfg.experiment_name = 'pedipulation_transition_t'
  cfg.logger = 'tensorboard'
  cfg.upload_model = False
  cfg.clip_actions = 10.
  cfg.max_iterations = 15000
  cfg.save_interval = 100
  return cfg


class TransitionOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
    super().__init__(env, train_cfg, log_dir, device)
    self.term = env.unwrapped.command_manager.get_term('twist')
    if self.alg.actor.obs_dim != 84 or self.alg.actor.distribution.output_dim != 12:
      raise ValueError('Transition teacher requires actor84 and canonical action12')

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    # Reference-state resets already diversify phase and initialize matching
    # physical states. Randomizing only episode counters creates invalid pairs.
    return super().learn(num_learning_iterations, init_at_random_ep_len=False)

  def _contract(self):
    action = self.env.unwrapped.action_manager.get_term('joint_pos')
    robot_cfg = self.env.unwrapped.cfg.scene.entities['robot']
    return dict(version=1, task='pedipulation_transition_t',
      observation_schema='actor84_body_rotvec_v1_critic172_clean_feet_physics',
      motion_sha256=self.term.motion.sha256, anchor=ANCHOR_CONTRACT,
      source_physics_sha256=hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest(),
      actor_dim=self.alg.actor.obs_dim, critic_dim=self.alg.critic.obs_dim,
      command='vx_vy_wz_FR_offset_from_quadruped_zero',
      foot_zero=self.term.zero.detach().cpu().tolist(),
      joint_names=[action._entity.joint_names[i] for i in action.joint_ids.tolist()],
      default_angles=action.default_angles[0].detach().cpu().tolist(),
      action_scale=action.cfg.scale, clip_actions=self.env.clip_actions,
      nominal_armatures=[act.armature for act in robot_cfg.articulation.actuators],
      nominal_actuators=[dict(targets=list(act.target_names_expr),
        stiffness=act.stiffness, damping=act.damping, effort_limit=act.effort_limit,
        armature=act.armature, frictionloss=act.frictionloss)
        for act in robot_cfg.articulation.actuators],
      control_dt=self.env.unwrapped.step_dt,
      delay_schedule='stage5_uniform_substep_delay')

  def save(self, path, infos=None):
    super().save(path, {**(infos or {}), 'transition_contract':self._contract(),
      'transition_course':self.term.course_state_dict()})

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    contract = (saved.get('infos') or {}).get('transition_contract')
    if contract != self._contract():
      raise ValueError('Checkpoint transition contract does not match motion, model, observation or action')
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    if 'transition_course' in infos:
      self.term.load_course_state_dict(infos['transition_course'])
    return infos
