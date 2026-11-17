"""DAgger plus differentiable Go2 imagination with isolated FSM supervision."""
from collections import deque
from dataclasses import asdict
from copy import deepcopy
import torch
from torch import nn
from .physics import WorldModel, Go2Kinematics, DifferentiableTaskReward, state_from_env, state_observation
from .teacher import ImaginedTeacher, teacher_context
from .augmentation import mirror_model_batch, sequence_symmetry, mirror_observation
from .config import SuperMimicCfg
from .diagnostics import GradientMonitor, task_balanced_mean
from .learning import validate_dynamics, train_real_critic, imagined_termination
from .clearml_logging import SuperMimicTracker
from ..rl.algorithm import LegManipDistillation
from ..rl.runner import LegManipDistillationRunner


class TransitionReplay:
    """Time-major ring. The transition ending an episode remains supervised."""
    def __init__(self, capacity):
        self.steps = deque(maxlen=capacity)

    def add(self, values):
        self.steps.append({key: value.detach().clone() for key, value in values.items()})

    def sample(self, batch_size, horizon, allowed_envs=None):
        if len(self.steps) < horizon:
            raise ValueError('Not enough contiguous steps for world-model training')
        data = {key: torch.stack([step[key] for step in self.steps]) for key in self.steps[0]}
        starts = len(self.steps) - horizon + 1
        labels = data.get('task', data['owner'])[:starts]
        # Stratify physical support patterns within each behavior.
        bits = (data['contacts'][:starts] > .5).long()
        patterns = (bits * bits.new_tensor((1,2,4,8))).sum(-1)
        owners = (labels * 16 + patterns).flatten()
        candidates_mask = torch.ones_like(labels, dtype=torch.bool)
        if allowed_envs is not None:
            candidates_mask &= torch.isin(torch.arange(labels.shape[1], device=owners.device), allowed_envs)[None]
        flat_candidates = candidates_mask.flatten().nonzero().flatten()
        if not len(flat_candidates):
            raise ValueError('No eligible replay environments')
        groups = owners[flat_candidates].unique()
        selected = []
        for group in groups:
            candidates = ((owners == group) & candidates_mask.flatten()).nonzero().flatten()
            selected.append(candidates[torch.randint(len(candidates), (batch_size,), device=owners.device)])
        table = torch.stack(selected)
        flat = table[torch.randint(len(groups), (batch_size,), device=owners.device),
                     torch.arange(batch_size, device=owners.device)]
        env_ids = flat % data['state'].shape[1]
        time_ids = flat // data['state'].shape[1]
        offsets = torch.arange(horizon, device=owners.device)[:, None] + time_ids
        result = {key: value[offsets, env_ids] for key, value in data.items()}
        alive = torch.ones(batch_size, device=owners.device, dtype=torch.bool)
        valid = []
        finite = torch.ones_like(result['done'], dtype=torch.bool)
        for key in ('state', 'next_state', 'targets', 'contacts', 'collision', 'real_features', 'next_features', 'real_reward'):
            if key not in result:
                continue
            finite &= torch.isfinite(result[key]).reshape(*result['done'].shape, -1).all(-1)
            result[key] = torch.nan_to_num(result[key], nan=0., posinf=0., neginf=0.)
        for t, done in enumerate(result['done']):
            alive = alive & finite[t]
            valid.append(alive)
            alive = alive & ~done.bool()
            if 'boundary' in result:
                alive = alive & ~result['boundary'][t].bool()
        result['valid'] = torch.stack(valid)
        result['env_ids'] = env_ids
        return result


def critic_features(state, command, context, contacts):
    # World translation should not change the value of identical tasks.
    canonical = torch.cat((torch.zeros_like(state[:, :2]), state[:, 2:]), -1)
    return torch.cat((canonical, command, context['desired_upright'][:, None],
                      context['op_weight'][:, None], contacts), -1)


class SuperMimicAlgorithm(LegManipDistillation):
    @property
    def beta(self):
        probability = super().beta
        if self.teacher_probability is not None:
            return probability
        return max(getattr(self, 'teacher_probability_floor', .1), probability)

    def configure(self, env, cfg):
        if self.is_multi_gpu:
            raise ValueError('SuperMimic currently trains on one GPU per run')
        self.world_env = env.unwrapped
        self.sm_cfg = SuperMimicCfg(**cfg)
        c = self.sm_cfg
        if abs(self.world_env.step_dt - .02) > 1e-9 or self.world_env.cfg.decimation != 4:
            raise ValueError('Go2 world model requires 50 Hz control and four 5 ms substeps')
        if min(c.horizon, c.model_horizon, c.model_updates, c.imagination_updates,
               c.model_batch_size, c.hidden_dim) < 1 or c.replay_steps < c.model_horizon:
            raise ValueError('Positive horizons/batches/updates and sufficient replay capacity required')
        if c.imagination_start_update < 0 or c.learning_rate <= 0:
            raise ValueError('Invalid world-model warmup or learning rate')
        if not 0 < c.gamma <= 1 or not 0 <= c.gae_lambda <= 1:
            raise ValueError('Invalid return discount or lambda')
        self.world_model = WorldModel(hidden_dim=c.hidden_dim).to(self.device)
        self.critic = nn.Sequential(nn.Linear(49, 256), nn.ELU(), nn.Linear(256, 128),
                                    nn.ELU(), nn.Linear(128, 1)).to(self.device)
        self.critic_target = deepcopy(self.critic).requires_grad_(False)
        self.model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=c.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c.learning_rate)
        self.reward = DifferentiableTaskReward(Go2Kinematics().to(self.device)).to(self.device)
        self.replay = TransitionReplay(c.replay_steps)
        self.shadow = ImaginedTeacher(self.teacher)
        self.teacher_probability_floor = c.teacher_probability_floor
        if not 0 <= c.teacher_probability_floor <= 1 or min(c.joint_updates, c.gradient_interval,
            c.detailed_gradient_interval, c.validation_interval, c.validation_batch_size,
            c.reliable_validations) < 1 or c.max_horizon < c.horizon or c.validation_error_scale <= 0:
            raise ValueError('Invalid joint update/diagnostic/validation configuration')
        if env.num_envs < 2:
            raise ValueError('At least two environments required for independent dynamics validation')
        validation_count = max(1, env.num_envs // 8)
        self.training_envs = torch.arange(env.num_envs-validation_count, device=self.device)
        self.validation_envs = torch.arange(env.num_envs-validation_count, env.num_envs, device=self.device)
        self.validation_scores = torch.full((6,), float('inf'), device=self.device)
        self.contact_scores = torch.ones(6, device=self.device)
        self.validation_counts = torch.zeros(6, dtype=torch.long, device=self.device)
        self.active_horizon = c.horizon
        self.gradient_monitor = GradientMonitor()
        self.mirrored_hidden = torch.zeros(1,env.num_envs,256,device=self.device)
        self.symmetry_valid_history = torch.ones(env.num_envs,dtype=torch.bool,device=self.device)
        self.real_metrics = {}


    @torch.no_grad()
    def act(self, obs):
        intent = self.world_env.command_manager.get_term('intent')
        if hasattr(intent, 'set_training_progress'):
            intent.set_training_progress(self.num_updates)
        task = getattr(intent, 'task_id', self.teacher.term.owner.clamp(0,5))
        if self.storage.step == 0:
            self.imag_state = state_from_env(self.world_env).clone()
            self.imag_previous = self.teacher.action.raw_action.clone()
            self.imag_task = task.clone()
            self.symmetry_initial_hidden = self.mirrored_hidden.clone()
            self.shadow.seed()
            self.symmetry_gates, self.rollout_tasks = [], []
            self.real_metrics = {}
        self.physical_before = state_from_env(self.world_env).clone()
        self.owner_before = self.teacher.term.owner.clone()
        self.task_before = task.clone()
        self.command_before = intent.command.clone()
        self.context_before = {k:v.detach().clone() for k,v in teacher_context(self.teacher).items()}
        self.previous_before = self.teacher.action.raw_action.clone()
        self.contacts_before = self._contacts()
        walking = ((self.teacher.term.owner == 0) & (self.teacher.term.blend == 0)
                   & (intent.command[:,3:] == 0).all(-1))
        self.symmetry_valid_history = self.symmetry_valid_history & walking
        self.symmetry_gates.append((walking & self.symmetry_valid_history).clone())
        self.rollout_tasks.append(task.clone())
        _, self.mirrored_hidden = self.student.forward_sequence(
            mirror_observation(obs['student'],self.teacher.action.default_angles)[None],
            torch.zeros(1,self.world_env.num_envs,dtype=torch.bool,device=self.device),self.mirrored_hidden)
        return super().act(obs)

    def _contacts(self):
        force = self.world_env.scene['feet_ground_contact'].data.force
        return (force.reshape(self.world_env.num_envs,4,-1,3).sum(2).norm(dim=-1)>5.).float()

    @torch.no_grad()
    def process_env_step(self, obs, rewards, dones, extras):
        terminal = self.world_env.supermimic_terminal
        # Recover applied final-substep common action before reset destroys
        # action buffers. This also handles teacher substep control correctly.
        applied = (terminal['targets'][:,-1] - self.teacher.action.default_angles
                   - self.teacher.action.motor_offsets) / self.teacher.action.cfg.scale
        terms = self.reward(self.physical_before, terminal['next_state'], applied,
                            self.previous_before, self.command_before, self.context_before,
                            terminal['contacts'], terminal['collision'])
        before = critic_features(self.physical_before,self.command_before,self.context_before,self.contacts_before)
        intent = self.world_env.command_manager.get_term('intent')
        after_context = teacher_context(self.teacher)
        after = critic_features(terminal['next_state'],intent.command,after_context,terminal['contacts'])
        values = {'state': self.physical_before, 'owner': self.owner_before, 'task': self.task_before,
                  'done': dones.reshape(-1).bool(), 'boundary':torch.zeros_like(dones.reshape(-1),dtype=torch.bool), 'real_features':before,
                  'next_features':after,'real_reward':terms['total'], **terminal}
        self.replay.add(values)
        for task in self.task_before.unique().tolist():
            gate = self.task_before == task
            for name in ('linear_error','angular_error','velocity_objective','foot_objective','posture_objective','regularization_objective','total'):
                key = f'supermimic/real/task_{task}/{name}'
                total,count = self.real_metrics.get(key,(0.,0))
                self.real_metrics[key] = (total+terms[name][gate].sum().item(),count+gate.sum().item())
            operation = gate & (terms['operation'] > .01)
            if operation.any():
                key = f'supermimic/real/task_{task}/foot_error'
                total,count = self.real_metrics.get(key,(0.,0))
                self.real_metrics[key] = (total+terms['foot_error'][operation].sum().item(),
                                         count+terms['operation'][operation].sum().item())
        self.mirrored_hidden[:,dones.reshape(-1).bool()] = 0.
        self.symmetry_valid_history[dones.reshape(-1).bool()] = True
        super().process_env_step(obs,rewards,dones,extras)

    def _train_model(self):
        c = self.sm_cfg
        stats = {}
        for _ in range(c.model_updates):
            batch = self.replay.sample(c.model_batch_size, c.model_horizon, allowed_envs=self.training_envs)
            if c.mirror_augmentation:
                batch = mirror_model_batch(batch)
            predicted = batch['state'][0]
            losses = []
            for t in range(c.model_horizon):
                output = self.world_model(predicted, batch['targets'][t])
                # The same public loss supplies manifold/scaled state and contact errors.
                loss = self.world_model.loss(predicted, batch['targets'][t], batch['next_state'][t],
                                             batch['contacts'][t], batch['collision'][t],
                                             valid_mask=batch['valid'][t])
                losses.append(loss['total'])
                predicted = output['state']
            total = torch.stack(losses).mean()
            self.model_optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.)
            self.model_optimizer.step()
            stats['supermimic/model_loss'] = total.detach().item()
        return stats

    def _imagine(self, initial_hidden):
        """Build differentiable objectives; parameter update belongs to update()."""
        c = self.sm_cfg
        self.shadow.reset()
        state, previous = self.imag_state.detach().clone(), self.imag_previous.detach().clone()
        hidden = initial_hidden.detach().clone()
        task = self.imag_task.detach().clone()
        horizon = self.active_horizon
        alive = state.new_ones(len(state))
        total_return = state.new_zeros(len(state))
        anchor = state.new_zeros(len(state))
        components = {k:state.new_zeros(len(state)) for k in
                      ('velocity_objective','foot_objective','posture_objective','regularization_objective')}
        contacts = state.new_zeros(len(state),4)
        relabel = ((torch.rand(len(state),device=self.device)<c.command_relabel_fraction)
                   & (self.shadow.teacher.term.owner == 0) & (self.shadow.teacher.term.blend == 0)
                   & (self.shadow.command[:,3:]==0).all(-1))
        velocity = self.shadow.command[:,:3].clone()
        velocity += (torch.rand_like(velocity)*2-1)*state.new_tensor((.03,0.,.03))
        for module in (self.world_model,self.critic):
            module.requires_grad_(False)
        try:
            for t in range(horizon):
                with torch.no_grad():
                    ids = relabel.nonzero().flatten()
                    if len(ids):
                        self.shadow.set_walking_velocity(ids,velocity[ids])
                command = self.shadow.command.detach().clone()
                context = {k:v.detach().clone() for k,v in self.shadow.context().items()}
                obs = state_observation(state,command,previous,self.teacher.action.default_angles)
                prediction,hidden = self.student.forward_sequence(obs[None],
                    torch.zeros(1,len(state),dtype=torch.bool,device=self.device),hidden)
                action = prediction[0].clamp(-100.,100.)
                target = self.shadow.query(state.detach()).clone()
                anchor = anchor+alive.detach()*(action-target).abs().mean(-1)/horizon
                targets = (self.teacher.action.default_angles+.25*action
                           +self.teacher.action.motor_offsets)[:,None].expand(-1,4,-1)
                output = self.world_model(state,targets)
                next_state = output['state']
                contact = output['contact_logits'].sigmoid()
                collision = output['collision_logits'].sigmoid().reshape(-1)
                terms = self.reward(state,next_state,action,previous,command,context,contact,collision)
                weight = (c.gamma**t)*alive/horizon
                total_return = total_return+weight*terms['total']
                for k in components:
                    components[k] = components[k]+weight*terms[k]
                alive = alive*(1-collision)
                intent = self.shadow.env.command_manager.get_term('intent')
                scenario_done_before = (intent.finished | intent.timed_out).clone()
                self.shadow.advance(next_state.detach(),action.detach())
                alive = alive*(~imagined_termination(self.shadow.env,scenario_done_before)).float()
                previous,state,contacts = action,next_state,contact
            with torch.no_grad():
                ids = relabel.nonzero().flatten()
                if len(ids):
                    self.shadow.set_walking_velocity(ids,velocity[ids])
            context = {k:v.detach().clone() for k,v in self.shadow.context().items()}
            endpoint = critic_features(state,self.shadow.command.detach().clone(),context,contacts)
            bootstrap = (c.gamma**horizon)*alive*self.critic(endpoint).squeeze(-1)/horizon
            total_return = total_return+bootstrap
            # Independent validation controls only reward-derived gradients.
            score = self.validation_scores[task]
            confidence = torch.where(torch.isfinite(score),
                torch.exp(-score/c.validation_error_scale),torch.full_like(score,.1)).clamp(.05,1.)
            confidence *= (1-self.contact_scores[task]).clamp(.1,1.)
            per_env = c.teacher_anchor*anchor-c.reward_weight*confidence*total_return
            objectives = {'imagined_teacher':c.teacher_anchor*task_balanced_mean(anchor,task),
                          'imagined_reward':-c.reward_weight*task_balanced_mean(confidence*total_return,task)}
            detail = {k:-c.reward_weight*task_balanced_mean(confidence*v,task) for k,v in components.items()}
            detail['value_bootstrap'] = -c.reward_weight*task_balanced_mean(confidence*bootstrap,task)
            return task_balanced_mean(per_env,task),per_env,objectives,detail,{
                'supermimic/teacher_anchor':anchor.detach().mean().item(),
                'supermimic/imagined_return':total_return.detach().mean().item(),
                'supermimic/imagination_confidence':confidence.mean().item()}
        finally:
            for module in (self.world_model,self.critic):
                module.requires_grad_(True)

    def update(self):
        c = self.sm_cfg
        initial_hidden = self.rollout_initial_hidden.detach().clone()
        observations = self.storage.observations['student'].detach().clone()
        targets = self.storage.privileged_actions.detach().clone()
        dones = self.storage.dones[...,0].detach().clone().bool()
        task = torch.stack(self.rollout_tasks).detach().clone()
        gates = torch.stack(self.symmetry_gates).detach().clone()
        mirrored_initial = self.symmetry_initial_hidden.detach().clone()
        stats = {}
        if len(self.replay.steps) >= c.model_horizon:
            stats.update(self._train_model())
            stats.update(train_real_critic(self))
            if (self.num_updates+1)%c.validation_interval == 0:
                stats.update(validate_dynamics(self))
        imagine = self.num_updates+1 >= c.imagination_start_update and len(self.replay.steps)>=c.model_horizon
        for joint in range(c.joint_updates):
            prediction,_ = self.student.forward_sequence(observations,dones,initial_hidden)
            error = (prediction-targets).abs().mean(-1)
            behavior = task_balanced_mean(error,task)
            objectives = {'real_teacher':behavior}
            total = behavior
            symmetry_parts = None
            if c.symmetry_weight > 0 and gates.any():
                sym_errors,sym_gates = sequence_symmetry(self.student,observations,dones,gates,
                    self.teacher.action.default_angles,initial_hidden,mirrored_initial,return_errors=True)
                symmetry_parts = c.symmetry_weight*sym_errors*sym_gates/sym_gates.sum().clamp_min(1)
                symmetry = symmetry_parts.sum()
                objectives['symmetry'] = symmetry
                total = total+symmetry
                stats['supermimic/symmetry_loss'] = symmetry.detach().item()
            per_env,details = None,{}
            if imagine:
                imagined,per_env,imag_objectives,details,imag_stats = self._imagine(initial_hidden)
                total = total+imagined
                objectives.update(imag_objectives)
                stats.update(imag_stats)
            if joint == 0:
                task_losses = {}
                for k in task.unique().tolist():
                    loss = error[task==k].mean()/len(task.unique())
                    if per_env is not None and (self.imag_task==k).any():
                        loss = loss+per_env[self.imag_task==k].mean()/len(self.imag_task.unique())
                    if symmetry_parts is not None:
                        loss = loss+symmetry_parts[task==k].sum()
                    task_losses[f'task_{k}'] = loss
                if (self.num_updates+1)%c.gradient_interval == 0:
                    measured = self.gradient_monitor.measure(task_losses,list(self.student.named_parameters()))
                    stats.update({'supermimic/grad_tasks/'+k:v for k,v in measured.items()})
                if (self.num_updates+1)%c.detailed_gradient_interval == 0:
                    # Coarse and detailed objectives overlap, so report them in
                    # separate families rather than treating them as a sum.
                    for family,losses in (('objectives',objectives),('rewards',details)):
                        measured = self.gradient_monitor.measure(losses,list(self.student.named_parameters()))
                        stats.update({'supermimic/grad_'+family+'/'+k:v for k,v in measured.items()})
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.student.parameters(),self.max_grad_norm)
            if not torch.isfinite(grad):
                raise FloatingPointError('Nonfinite joint student gradient')
            self.optimizer.step()
            stats.update(behavior=behavior.detach().item(),**{
                'supermimic/actor_loss':total.detach().item(),
                'supermimic/student_grad_norm':float(grad)})
        self.storage.clear()
        self.rollout_initial_hidden = None
        self.num_updates += 1
        stats.update(teacher_probability=self.beta,
                     teacher_control_fraction=self.teacher_control_mask.float().mean().item())
        stats.update({k:total/max(count,1) for k,(total,count) in self.real_metrics.items()})
        intent = self.world_env.command_manager.get_term('intent')
        if hasattr(intent,'coverage_metrics'):
            stats.update({'supermimic/coverage/'+k:v for k,v in intent.coverage_metrics().items()})
        return stats

    def reset_rollout_state(self):
        super().reset_rollout_state()
        if hasattr(self,'mirrored_hidden'):
            self.mirrored_hidden = torch.zeros_like(self.mirrored_hidden)
            self.symmetry_valid_history = torch.ones_like(self.symmetry_valid_history)

    def save(self):
        checkpoint = super().save()
        checkpoint['supermimic'] = {
            'version': 2, 'config': asdict(self.sm_cfg),
            'world_model': self.world_model.state_dict(), 'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'model_optimizer': self.model_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay': list(self.replay.steps),
            'gradient_monitor':self.gradient_monitor.state_dict(),
            'validation_scores':self.validation_scores,
            'validation_counts':self.validation_counts,
            'contact_scores':self.contact_scores,
            'active_horizon':self.active_horizon,
            'sampler':self.world_env.command_manager.get_term('intent').state_dict()
                      if hasattr(self.world_env.command_manager.get_term('intent'),'state_dict') else None,
        }
        return checkpoint

    def load(self, loaded_dict, load_cfg=None, strict=True):
        saved = loaded_dict.get('supermimic')
        if saved is None or saved.get('version') != 2 or saved['config'] != asdict(self.sm_cfg):
            raise ValueError('Resume requires a compatible SuperMimic checkpoint from this recipe')
        restored = super().load(loaded_dict, load_cfg, strict)
        for name in ('world_model', 'critic', 'critic_target', 'model_optimizer', 'critic_optimizer'):
            getattr(self, name).load_state_dict(saved[name])
        self.gradient_monitor.load_state_dict(saved['gradient_monitor'])
        for name in ('validation_scores','validation_counts','contact_scores'):
            setattr(self,name,saved[name].to(self.device))
        self.active_horizon = saved['active_horizon']
        intent = self.world_env.command_manager.get_term('intent')
        if saved.get('sampler') is not None:
            intent.load_state_dict(saved['sampler'])
        self.replay.steps.clear()
        for step in saved['replay']:
            self.replay.add({key: value.to(self.device) for key, value in step.items()})
        if self.replay.steps:
            # Resume resets physics. The old replay remains usable, but no window
            # may connect its last transition to a newly initialized episode.
            self.replay.steps[-1]['boundary'].fill_(True)
        return restored


class SuperMimicRunner(LegManipDistillationRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        cfg = asdict(train_cfg) if hasattr(train_cfg, '__dataclass_fields__') else deepcopy(train_cfg)
        self.tracker = SuperMimicTracker(cfg.get('clearml', {}), cfg.get('run_name', ''))
        self.last_checkpoint = None
        try:
            super().__init__(env, cfg, log_dir, device)
            self.alg.configure(env, self.cfg['supermimic'])
            self.tracker.configuration(self._metadata())
        except BaseException:
            self.tracker.close(failed=True)
            raise

    def save(self, path, infos=None):
        super().save(path, infos)
        self.last_checkpoint = path
        self.tracker.checkpoint(path, self.alg.num_updates)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        failed = True
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len)
            if self.last_checkpoint is not None:
                self.tracker.checkpoint(self.last_checkpoint, self.alg.num_updates, force=True)
            failed = False
        finally:
            try:
                self.alg.shadow.close()
            finally:
                self.tracker.close(failed=failed)
