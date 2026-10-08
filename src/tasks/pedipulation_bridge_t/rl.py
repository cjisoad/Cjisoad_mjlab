"""PPO on completed bridge attempts, with delayed frozen-target feedback."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import torch
from rsl_rl.algorithms import PPO
from mjlab.rl import MjlabOnPolicyRunner, RslRlPpoAlgorithmCfg
from src.tasks.teacher_common.rl import teacher_ppo_runner_cfg
from .rollout import BridgeRolloutStorage
from .guidance import ACTOR_DIM, CRITIC_DIM, GUIDANCE_CHANNELS


def initialize_guided_actor(guided, source):
  """Preserve source mean exactly while adding initially unused guidance inputs."""
  source_state=source.state_dict()
  target_state=guided.state_dict()
  if source_state.keys()!=target_state.keys():
    raise ValueError('Guided actor must preserve the frozen source MLP architecture')
  mapped={}
  for name,value in source_state.items():
    if name=='0.weight':
      expected=(value.shape[0],ACTOR_DIM)
      if value.shape[1]!=51 or target_state[name].shape!=expected:
        raise ValueError(f'Guided actor first layer must extend source51 to guidance{ACTOR_DIM}')
      expanded=torch.zeros_like(target_state[name])
      expanded[:,:51]=value
      mapped[name]=expanded
    else:
      if value.shape!=target_state[name].shape:
        raise ValueError(f'Guided actor layer differs from source: {name}')
      mapped[name]=value
  guided.load_state_dict(mapped)


@dataclass
class BridgeAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = 'src.tasks.pedipulation_bridge_t.rl:BridgePPO'
  max_bridge_steps: int = 502


def bridge_ppo_runner_cfg():
  from dataclasses import asdict
  cfg = teacher_ppo_runner_cfg('loco_pedipulation_t')
  params = asdict(cfg.algorithm)
  params.pop('class_name')
  cfg.algorithm = BridgeAlgorithmCfg(**params)
  cfg.algorithm.max_bridge_steps=502
  cfg.algorithm.learning_rate=3e-4
  cfg.algorithm.entropy_coef=.005
  cfg.algorithm.gamma=.995
  cfg.algorithm.lam=.95
  cfg.actor.distribution_cfg['init_std']=.5
  cfg.class_name = 'BridgeOnPolicyRunner'
  cfg.experiment_name = 'pedipulation_bridge_t'
  cfg.max_iterations = 15000
  return cfg


class BridgePPO(PPO):
  def __init__(self, actor, critic, storage, *, max_bridge_steps=502, **kwargs):
    if actor.is_recurrent or critic.is_recurrent:
      raise ValueError('Complete bridge collection requires feed-forward models')
    packed = BridgeRolloutStorage('rl', storage.num_envs, max_bridge_steps,
      storage.observations[0], storage.actions_shape, storage.device)
    super().__init__(actor, critic, packed, **kwargs)

  def compute_returns(self, obs):
    # Every nonempty episode closes with a measured outcome. No value bootstrap
    # across expert-controlled steps or a PPO update with pending tail feedback.
    self.storage.compute_closed_returns(self.gamma, self.lam,
      normalize=not self.normalize_advantage_per_mini_batch)


class BridgeOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
    super().__init__(env, train_cfg, log_dir, device)
    if self.is_distributed:
      raise ValueError('Variable-length complete-attempt collector currently supports one GPU')
    if (self.alg.actor.obs_dim, self.alg.critic.obs_dim) != (ACTOR_DIM, CRITIC_DIM):
      raise ValueError('Tracking bridge requires actor84/critic172; frozen teachers remain actor51')
    self.term = env.unwrapped.command_manager.get_term('twist')
    self.action = env.unwrapped.action_manager.get_term('joint_pos')
    if self.alg.storage.num_transitions_per_env < int(self.term.cfg.bridge_deadline/env.unwrapped.step_dt)+2:
      raise ValueError('max_bridge_steps cannot hold a complete bridge deadline')
    self.experts = self.action._ensure_experts()
    self.action.experts = self.experts
    # Tracking actor/critic start fresh. Frozen experts only generate entries
    # or evaluate actual handoffs; they never initialize the student weights.
    self.updates = 0
    self.total_attempts = 0
    self.last_metrics = {}

  @torch.inference_mode()
  def collect_attempts(self):
    from .commands import BRIDGE, VERIFY
    obs, _ = self.env.reset()
    obs = obs.to(self.device)
    st = self.alg.storage
    st.clear()
    finished = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
    outcome = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)
    reason = torch.zeros_like(outcome)
    verified = torch.zeros_like(finished)
    moving = self.term.moving.clone().to(self.device)
    full_start=self.term.full_start.clone().to(self.device)
    live_start=self.term.live_start.clone().to(self.device)
    start_kind=torch.where(live_start,3,torch.where(full_start,0,
      torch.where(self.term.start_time.to(self.device)<self.term.guidance.rise_duration,1,2)))
    first_stand=torch.full((self.env.num_envs,),float('nan'),device=self.device)
    reward_phases={name:torch.zeros(self.env.num_envs,device=self.device) for name in ('preparation','rise','endpoint')}
    fr_error_squared = torch.zeros(self.env.num_envs, device=self.device)
    max_height = torch.zeros_like(fr_error_squared)
    upright_frames = torch.zeros_like(fr_error_squared)
    recorded_start=torch.zeros_like(finished)
    start_linear=torch.zeros(self.env.num_envs,3,device=self.device)
    start_angular=torch.zeros_like(start_linear)
    start_error=torch.zeros_like(fr_error_squared)
    limit = int(self.env.max_episode_length) + 5
    for ticks in range(limit):
      active = ~finished
      st.collection_mask.copy_((self.term.owner.to(self.device) == BRIDGE) & active)
      mask = st.collection_mask
      starting=mask & ~recorded_start
      reference_time=self.term.start_time.to(self.device)+self.term.bridge_elapsed.to(self.device)*self.term.reference_speed.to(self.device)
      phases={'preparation':reference_time<.5,'rise':(reference_time>=.5)&(reference_time<self.term.guidance.rise_duration),
              'endpoint':reference_time>=self.term.guidance.rise_duration}
      data = self.env.unwrapped.scene['robot'].data
      height = data.root_link_pos_w[:,2] - self.env.unwrapped.scene.env_origins[:,2]
      fr_error_squared += mask * self.term.fr_error.square().sum(-1)
      max_height = torch.where(mask, torch.maximum(max_height, height), max_height)
      upright_frames += mask * ((data.projected_gravity_b[:,0]<-.8) & (height>.44))
      actions = self.alg.act(obs)
      # Snapshot before physics/autoreset, including the impending one-off
      # velocity addition. Reference starts otherwise reused stale live buffers.
      pending=self.term.pending_initial_impulse.to(self.device)[:,None]
      start_linear[starting]=(data.root_link_lin_vel_w.to(self.device)+self.term.initial_impulse[:,:3].to(self.device)*pending)[starting]
      start_angular[starting]=(data.root_link_ang_vel_w.to(self.device)+self.term.initial_impulse[:,3:].to(self.device)*pending)[starting]
      start_error[starting]=self.term.fr_error.to(self.device).norm(dim=-1)[starting]
      obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
      obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
      recorded_start |= starting
      if not all(torch.isfinite(v).all() for v in obs.values()) or not torch.isfinite(rewards).all():
        raise FloatingPointError('Nonfinite bridge observations or rewards')
      if not torch.equal(st.collection_mask, (self.term.action_owner.to(self.device) == BRIDGE) & active):
        raise RuntimeError('Controller ownership changed before the physics step')
      # Bridge timeouts and target failures are actual terminal objectives;
      # suppress generic continuing-task timeout bootstrapping.
      self.alg.process_env_step(obs, rewards, dones, {})
      for name,phase in phases.items(): reward_phases[name]+=rewards*mask*phase
      verified |= (self.term.action_owner.to(self.device) == VERIFY) & active
      event = self.term.outcome_event.to(self.device)
      ended = active & ((event != 0) | (dones > 0))
      ids = ended.nonzero().flatten()
      if len(ids):
        result = event[ids].clone()
        # Also handle any external termination that bypasses the bridge term.
        missing = result == 0
        result[missing] = torch.where(st.lengths[ids][missing] > 0, -1, -2)
        outcome[ids] = result
        reason[ids] = self.term.event_reason.to(self.device)[ids]
        feedback = torch.where(result == -1, -5., 0.)
        first_stand[ids]=self.term.completed_first_stand_time.to(self.device)[ids]
        tail_steps = self.term.tail_steps.to(self.device)[ids].clone()
        st.finish(ids, feedback, tail_steps, gamma=self.alg.gamma, discount_tail=False)
        finished[ids] = True
      if finished.all():
        break
    if not finished.all():
      # A bounded guard is a failed attempt, never a successful unresolved tail.
      ids = (~finished).nonzero().flatten()
      outcome[ids] = torch.where(st.lengths[ids] > 0, -1, -2)
      st.finish(ids, torch.where(st.lengths[ids] > 0, -5., 0.),
        self.term.tail_steps.to(self.device)[ids], gamma=self.alg.gamma, discount_tail=False)
      reason[ids] = 99
    lengths = st.lengths.clone()
    metrics = {'attempts':self.env.num_envs, 'physics_steps':ticks+1,
      'transition_samples':int(st.valid.sum()), 'source_rejected':int((outcome == -2).sum()),
      'source_yield':float((lengths > 0).float().mean()),
      'target_attempts':int(verified.sum()), 'successes':int((outcome == 1).sum()),
      'transition_or_target_failed':int((outcome == -1).sum()),
      'mean_transition_steps':float(lengths[lengths > 0].float().mean()) if (lengths > 0).any() else 0.,
      'fr_rms_error_m':float((fr_error_squared.sum()/lengths.sum().clamp_min(1)).sqrt()),
      'mean_max_base_height_m':float(max_height[lengths > 0].mean()) if (lengths > 0).any() else 0.,
      'upright_frame_fraction':float(upright_frames.sum()/lengths.sum().clamp_min(1)),
      'mean_start_linear_speed_m_s':float(start_linear[recorded_start].norm(dim=-1).mean()) if recorded_start.any() else 0.,
      'max_start_linear_speed_m_s':float(start_linear.norm(dim=-1).max()),
      'mean_start_angular_speed_rad_s':float(start_angular[recorded_start].norm(dim=-1).mean()) if recorded_start.any() else 0.,
      'mean_start_fr_error_m':float(start_error[recorded_start].mean()) if recorded_start.any() else 0.,
      'reason_counts':{str(int(r)):int((reason == r).sum()) for r in reason.unique()}}
    metrics['tracking_successes']=int((outcome==1).sum()) if not self.term.cfg.verify_handoff else 0
    metrics['course_stage']=self.term.course_stage
    metrics['course_level']=self.term.course.level
    metrics['live_probability']=self.term.course.live_probability
    metrics['randomization_fraction']=self.term.course.randomization_fraction
    metrics['reward_integrals_by_phase']={name:float(values.sum()) for name,values in reward_phases.items()}
    metrics['by_start']={}
    for index,label in enumerate(('full_reference','rise_reference','endpoint_reference','live_source')):
      subset=(start_kind==index)&(lengths>0)
      finite=subset & torch.isfinite(first_stand)
      metrics['by_start'][label]=dict(attempts=int(subset.sum()),successes=int(((outcome==1)&subset).sum()),
        stood_within5s=int((finite & (first_stand<=5.)).sum()),
        mean_first_stand_seconds=float(first_stand[finite].mean()) if finite.any() else None)
    # A whole packed batch uses one fixed course. Promote only after collection
    # so auto-reset worlds cannot change the training regime of active attempts.
    self.term.course.record((outcome==1).cpu().tolist(),full_start.cpu().tolist(),
      live_start.cpu().tolist(),step=self.env.unwrapped.common_step_counter)
    for label, subset in (('stationary', ~moving), ('moving', moving)):
      metrics[label] = {'attempts':int(subset.sum()),
        'source_yield':float((lengths[subset] > 0).float().mean()) if subset.any() else 0.,
        'target_attempts':int((verified & subset).sum()),
        'successes':int(((outcome == 1) & subset).sum())}
    speed=start_linear.norm(dim=-1)
    metrics['by_start_speed']={}
    for label,subset in (('below_0.1',recorded_start & (speed<.1)),
                        ('0.1_to_0.25',recorded_start & (speed>=.1) & (speed<.25)),
                        ('at_least_0.25',recorded_start & (speed>=.25))):
      metrics['by_start_speed'][label]=dict(attempts=int(subset.sum()),
        target_attempts=int((verified & subset).sum()),successes=int(((outcome==1) & subset).sum()))
    self.last_metrics = metrics
    self.total_attempts += self.env.num_envs
    return obs, metrics

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    self.alg.train_mode()
    self.logger.init_logging_writer()
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    metrics_path = Path(self.logger.log_dir)/'bridge_metrics.jsonl' if self.logger.log_dir else None
    try:
      for it in range(start_it, total_it):
        start = time.time()
        obs, metrics = self.collect_attempts()
        collected = time.time()
        if metrics['transition_samples'] >= self.alg.num_mini_batches:
          with torch.inference_mode(): self.alg.compute_returns(obs)
          losses = self.alg.update()
          self.updates += 1
        else:
          self.alg.storage.clear()
          losses = {'value':0., 'surrogate':0., 'entropy':0.}
        elapsed = time.time()
        self.current_learning_iteration = it
        metrics.update(iteration=it, updates=self.updates,
          collection_seconds=collected-start, update_seconds=elapsed-collected,
          losses=losses)
        std=self.alg.actor.output_std.detach().reshape(-1,12)[0]
        metrics['action_std_by_joint']=std.cpu().tolist()
        metrics['pd_target_std_rad_by_joint']=(std*self.action.cfg.scale).cpu().tolist()
        print('BRIDGE_PROGRESS '+json.dumps(metrics), flush=True)
        if metrics_path:
          with metrics_path.open('a') as stream: stream.write(json.dumps(metrics)+'\n')
        if self.logger.writer:
          # Honest counts for variable collection, without claiming 24 physics
          # steps were collected. Source failures cannot dilute learning reward.
          self.logger.tot_timesteps += metrics['physics_steps']*self.env.num_envs
          for key in ('source_yield','target_attempts','successes','transition_samples'):
            self.logger.writer.add_scalar('Bridge/'+key, metrics[key], it)
          for key, value in losses.items(): self.logger.writer.add_scalar('Loss/'+key, value, it)
          self.logger.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, it)
          self.logger.writer.add_scalar('Policy/mean_std', self.alg.actor.output_std.mean().item(), it)
          from .experts import JOINT_NAMES
          for joint,value,target in zip(JOINT_NAMES,metrics['action_std_by_joint'],metrics['pd_target_std_rad_by_joint']):
            self.logger.writer.add_scalar('Policy/std/'+joint,value,it)
            self.logger.writer.add_scalar('Policy/target_std_rad/'+joint,target,it)
        if metrics_path and it % self.cfg['save_interval'] == 0:
          self.save(str(metrics_path.parent/f'model_{it}.pt'))
      if metrics_path:
        self.save(str(metrics_path.parent/f'model_{self.current_learning_iteration}.pt'))
    finally:
      if self.logger.writer: self.logger.stop_logging_writer()

  def _contract(self):
    from .commands import BRIDGE_CONTRACT
    from .paths import PATH_ASSET
    command = {key:getattr(self.term.cfg, key) for key in (
      'warm_seconds','low_hold_seconds','reach_time','velocity_ramp_time',
      'prefix_rise_seconds','bridge_rise_seconds',
      'start_height_range','moving_probability','lin_vel_x','lin_vel_y','ang_vel_z',
      'source_deadline','bridge_deadline','attempt_deadline','verify_seconds',
      'candidate_seconds','recent_rear_seconds','invalid_seconds',
      'reference_speed_range','reference_blend_seconds','fr_tracking_scale',
      'handoff_fr_error','handoff_linear_speed','verification_velocity',
      'initial_impulse_probability','initial_linear_impulse','initial_angular_impulse')}
    # Static interface and controls (not mutable phase/counters).
    from src.tasks.pedipulation.robot import SNAPSHOT_PATH
    robot_cfg = self.env.unwrapped.cfg.scene.entities['robot']
    command.update({key:getattr(self.term.cfg,key) for key in (
      'min_stage_steps','min_curriculum_episodes','success_window','promotion_success',
      'undisturbed_probability')})
    return dict(version=3, task='pedipulation_bridge_t', handoff=BRIDGE_CONTRACT,
      actor_dim=ACTOR_DIM, critic_dim=CRITIC_DIM, guidance_channels=GUIDANCE_CHANNELS,
      source='loco_pedipulation_t',
      checkpoint_hashes={name:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name,path in (('quadruped',self.action.cfg.quadruped_checkpoint),
                          ('biped',self.action.cfg.biped_checkpoint))},
      expert_contracts=self.experts.contracts,
      motion_sha256=hashlib.sha256(Path(self.term.cfg.motion_path).read_bytes()).hexdigest(),
      paths_sha256=hashlib.sha256(PATH_ASSET.read_bytes()).hexdigest(),
      default_angles=self.action.default_angles[0].cpu().tolist(),
      clip_actions=self.env.clip_actions, action_scale=self.action.cfg.scale,
      control_delay='course2_disturbed_worlds_only',
      control_dt=self.env.unwrapped.step_dt,
      decimation=self.env.unwrapped.cfg.decimation,
      physical_snapshot_sha256=hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest(),
      nominal_actuators=[dict(targets=list(act.target_names_expr),
        stiffness=act.stiffness, damping=act.damping, effort_limit=act.effort_limit,
        armature=act.armature, frictionloss=act.frictionloss)
        for act in robot_cfg.articulation.actuators],
      command_parameters=command, terminal_feedback=[-5.,0.],
      terminal_credit='tracking_failure_once_last_student_step',
      actor_initialization='fresh_tracking_actor84',
      courses=['reference_and_fr','timing_and_recovery','live_source_and_randomization'],
      reward_terms={name:dict(weight=term.weight,params=term.params,
        function=term.func.__module__+'.'+term.func.__name__)
        for name,term in self.env.unwrapped.cfg.rewards.items()},
      contact_sensors=[asdict(sensor) for sensor in self.env.unwrapped.cfg.scene.sensors],
      rollout='packed_complete_attempts_expert_steps_excluded_v1')

  def save(self, path, infos=None):
    course=self.term.course.state_dict()
    course['age_steps']=self.env.unwrapped.common_step_counter-self.term.course.started
    super().save(path, {**(infos or {}), 'bridge_contract':self._contract(),
      'bridge_training':{'updates':self.updates, 'total_attempts':self.total_attempts},
      'bridge_course':course})

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('infos') or {}).get('bridge_contract') != self._contract():
      raise ValueError('Bridge checkpoint contract differs in teachers, reference, interface or handoff controls')
    info = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    state = (info or {}).get('bridge_training', {})
    self.updates = state.get('updates', 0)
    self.total_attempts = state.get('total_attempts', 0)
    if 'bridge_course' in (info or {}):
      enabled=self.term.course.enabled
      self.term.course.load_state_dict(info['bridge_course'])
      self.term.course.enabled=enabled
      self.term.course.started=self.env.unwrapped.common_step_counter-info['bridge_course'].get('age_steps',0)
    return info
