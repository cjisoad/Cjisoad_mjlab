"""Uninterrupted landing trials captured before any environment reset."""
import math
import torch
from src.tasks.transition_down_t.standing import tripod_conditions
from .gates import GateConfig

class LandingTrials:
    def __init__(self,count,dt,*,device='cpu',cfg=None):
        self.cfg=cfg or GateConfig();self.dt=dt
        self.elapsed=torch.zeros(count,dtype=torch.float64,device=device)
        self.hold=self.elapsed.clone();self.finished=torch.zeros(count,dtype=torch.bool,device=device)
        self.standing_held=self.finished.clone();self.unsafe=self.finished.clone()
        self.minimum=torch.full((count,),float('inf'),device=device)
        self.landing=torch.full_like(self.minimum,float('inf'))
        self.reasons=[None]*count;self.last_step=None
    def record(self,completed,metrics,terminated,truncated,*,safety,step):
        if self.last_step is not None and step<=self.last_step:return
        self.last_step=step;active=~self.finished;self.elapsed.add_(active*self.dt)
        self.minimum[active]=torch.minimum(self.minimum[active],safety['knee_min'][active])
        self.landing[active]=safety['landing_time'][active]
        finite=torch.stack([torch.isfinite(v) for v in metrics.values()],-1).all(-1)
        unsafe=(safety['knee_min']<self.cfg.knee_margin_m)|safety['nonfoot']|safety['fr_contact']|~finite
        self.unsafe.logical_or_(active&unsafe)
        late=(self.landing>self.cfg.landing_deadline_s)&(self.elapsed>=self.cfg.landing_deadline_s)
        late |= torch.isfinite(self.landing)&(self.landing>self.cfg.landing_deadline_s)
        failed=active&(terminated|unsafe|late)
        good=active&~failed&completed&torch.stack(tuple(tripod_conditions(metrics).values()),-1).all(-1)&(metrics['feet_rms']<.08)&torch.isfinite(self.landing)
        self.hold.copy_(torch.where(active,torch.where(good,self.hold+self.dt,0.),self.hold))
        self.standing_held.logical_or_(active&good&(self.hold>=self.cfg.hold_s-1e-9))
        timed=active&(truncated|(self.elapsed>=self.cfg.timeout_s-1e-9))
        end=failed|timed|self.standing_held
        for i in (end&active).nonzero().flatten().tolist():
            self.reasons[i]='unsafe' if bool(unsafe[i]) else 'termination' if bool(terminated[i]) else 'late_landing' if bool(late[i]) else 'success' if bool(self.standing_held[i]) else 'timeout'
        self.finished.logical_or_(end)
    def records(self):
        return [dict(success=bool(self.standing_held[i]),unsafe=bool(self.unsafe[i]),
            reason=self.reasons[i],elapsed_s=float(self.elapsed[i]),continuous_hold_s=float(self.hold[i]),
            knee_min_m=float(self.minimum[i]) if torch.isfinite(self.minimum[i]) else None,
            landing_time_s=float(self.landing[i]) if torch.isfinite(self.landing[i]) else None)
            for i in range(len(self.finished))]
    def summary(self):
        return dict(attempts=len(self.finished),successes=int(self.standing_held.sum()),
            unsafe_attempts=int(self.unsafe.sum()),records=self.records())

def evaluate_policy(actor,*,gate=None,device='cpu',group='nominal',count=100,level=0,split='validation',seed=424242,augmented=False):
    """Separate evaluation worlds and RNG; never update the training normalizer."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import TerminationTermCfg
    from tensordict import TensorDict
    from src.tasks.transition_down_t.entry_evaluation import isolated_rng,_copy_actor_for_eval
    from src.tasks.transition_down_t.standing import tripod_endpoint_metrics
    from .env_cfg import landing_env_cfg
    from .safety import install_monitor
    if group not in ('nominal','push_forward','push_backward','base','joints','combined','entry'):raise ValueError(group)
    if group=='entry' and split not in ('validation','final'):raise ValueError('Evaluation cannot sample training entries')
    gate=gate or GateConfig()
    with isolated_rng(device):
        cfg=landing_env_cfg(play=True,split=split);cfg.scene.num_envs=count;cfg.seed=seed
        cmd=cfg.commands['motion'];cmd.debug_vis=False;cmd.sampling_mode='start'
        cmd.entry_probability=1. if group=='entry' else 0.;cmd.nominal_probability=1.-cmd.entry_probability
        training=landing_env_cfg()
        if group in ('base','combined'):
            cmd.pose_range=training.commands['motion'].pose_range
            cmd.velocity_range=training.commands['motion'].velocity_range
        if group in ('joints','combined'):cmd.joint_position_range=training.commands['motion'].joint_position_range
        if augmented:
            cfg.events=training.events;cfg.observations['actor'].enable_corruption=True
        trials=None
        def capture(env):
            if trials is not None:
                term=env.command_manager.get_term('motion');manager=env.termination_manager
                trials.record(term.time_steps>=term.motion.time_step_total-1,tripod_endpoint_metrics(env),
                    manager.terminated,manager.time_outs,safety=env.landing_monitor.snapshot(),step=env.common_step_counter)
            return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        cfg.terminations['landing_capture']=TerminationTermCfg(func=capture)
        env=ManagerBasedRlEnv(cfg,device=device)
        try:
            monitor=install_monitor(env,gate);term=env.command_manager.get_term('motion')
            selected=None
            if group=='entry':
                term.course.phase='entry';term.course.level=level
                bounds=term.course.settings
                selected=term.bank.sample_trials(count,split=split,max_speed=bounds['max_speed'],max_fr_error=bounds['max_fr_error'])
                distinct=len(term.bank.data['trajectory_id'][selected].unique())
                if distinct<count:raise ValueError('Not enough distinct evaluation trajectories')
                term.forced_entry_indices=selected.clone()
            with torch.inference_mode():
                obs,_=env.reset(seed=seed)
                if group in ('push_forward','push_backward'):
                    robot=env.scene['robot'];v=torch.cat((robot.data.root_link_lin_vel_w,robot.data.root_link_ang_vel_w),-1).clone()
                    v[:,0]+=.03 if group=='push_forward' else -.03
                    robot.write_root_link_velocity_to_sim(v);env.sim.forward()
                    obs=env.observation_manager.compute(update_history=False)
                if not bool((term.time_steps==0).all()):raise AssertionError('Evaluation must start at frame0')
                monitor.sample(0.)
                trials=LandingTrials(count,env.step_dt,device=device,cfg=gate)
                policy=_copy_actor_for_eval(actor,device) if actor is not None else None
                if policy is not None:policy.eval()
                for _ in range(math.ceil(gate.timeout_s/env.step_dt)+2):
                    actions=policy(TensorDict(obs,batch_size=[count]),stochastic_output=False) if policy is not None else torch.zeros(count,12,device=device)
                    obs,_,_,_,_=env.step(actions.clamp(-10.,10.))
                    if bool(trials.finished.all()):break
            if not bool(trials.finished.all()):raise RuntimeError('Uncaptured landing attempts')
            result=trials.summary()
            result.update(group=group,split=split if selected is not None else None,level=level if selected is not None else None,
                distinct_trajectories=distinct if selected is not None else 0,seed=seed,augmented=augmented,physics_dt=env.physics_dt)
            if selected is not None:
                ids=term.bank.data['trajectory_id'][selected].tolist()
                for i,record in enumerate(result['records']):
                    record.update(trajectory_id=ids[i],entry_index=int(selected[i]),
                        actual_speed=float(term.bank.data['actual_speed'][selected[i]]),fr_error=float(term.bank.data['fr_error'][selected[i]]))
            return result
        finally:env.close()
