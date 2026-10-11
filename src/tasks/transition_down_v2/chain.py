"""Fresh live teacher rollouts, with continuous state at both ownership switches."""
from dataclasses import dataclass,fields,asdict
import math
import torch
from src.tasks.transition_down_t.chain import ChainTwistCfg,chain_failure,switch_owner,AlignedKeyboardTeacher,BIPED,TRANSITION,TRIPOD
from src.tasks.transition_t.keyboard import KeyboardTransitionActionCfg
from src.tasks.teacher_common.env_cfg import loco_pedipulation_teacher_env_cfg
from src.tasks.transition_down_t.standing import tripod_endpoint_metrics,tripod_conditions
from src.tasks.transition_down_t.entry_evaluation import isolated_rng,_copy_actor_for_eval
from .commands import LandingCommand,LandingCommandCfg
from .env_cfg import landing_env_cfg
from .evaluation import LandingTrials
from .safety import install_monitor

@dataclass(kw_only=True)
class LiveMotionCfg(LandingCommandCfg):
    def build(self,env):return LiveMotion(self,env)

class LiveMotion(LandingCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.active=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
    def _resample_command(self,ids):
        super()._resample_command(ids);self.active[ids]=False
    def _update_command(self):
        self._last_update_step[~self.active]=self._env.common_step_counter
        super()._update_command()

def sample_biped_commands(count,device):
    """Original source collection: half nominal, quarter offsets, quarter moving.

    Fresh commands, not stored final-entry states. Bounds match the approved
    source collection; realized handoff speed/error are recorded separately.
    """
    c=torch.zeros(count,6,device=device);c[:,3:]=c.new_tensor([.12,0.,.36])
    nominal=count//2;moving=3*count//4
    lo=c.new_tensor([.07,-.04,.30]);hi=c.new_tensor([.17,.04,.40])
    c[nominal:,3:]=lo+(hi-lo)*torch.rand(count-nominal,3,device=device)
    lo=c.new_tensor([-.25,-.15,-.6]);hi=c.new_tensor([1.,.15,.6])
    c[moving:,:3]=lo+(hi-lo)*torch.rand(count-moving,3,device=device)
    return c

def live_chain_cfg(gate,count,seed):
    from mjlab.managers import TerminationTermCfg
    cfg=landing_env_cfg(play=True,split='final');cfg.scene.num_envs=count;cfg.seed=seed
    old=cfg.commands['motion'];cfg.commands['motion']=LiveMotionCfg(**{f.name:getattr(old,f.name) for f in fields(old)})
    cfg.commands['motion'].entry_probability=0.;cfg.commands['motion'].nominal_probability=1.;cfg.commands['motion'].debug_vis=False
    source=loco_pedipulation_teacher_env_cfg(play=True).commands['twist']
    cfg.commands['twist']=ChainTwistCfg(**asdict(source))
    cfg.commands['twist'].resampling_time_range=(1e9,1e9)
    cfg.actions['joint_pos']=KeyboardTransitionActionCfg(**asdict(cfg.actions['joint_pos']))
    # Only DOWN follows a motion reference. All owners retain physical safety.
    for name in ('anchor_pos','anchor_ori','ee_body_pos','physical_failure'):
        cfg.terminations[name]=TerminationTermCfg(func=chain_failure,params={'kind':name})
    cfg.terminations.pop('time_out',None)
    cfg.episode_length_s=2.+gate.timeout_s+3.+1.
    return cfg

def evaluate_chain(actor,*,gate,device,count,seed,quad_teacher,biped_teacher):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import TerminationTermCfg
    from src.tasks.loco_pedipulation.mdp.commands import HOLD
    from tensordict import TensorDict
    with isolated_rng(device):
        cfg=live_chain_cfg(gate,count,seed)
        trials=None
        failed=torch.zeros(count,dtype=torch.bool,device=device);unsafe=failed.clone();success=failed.clone()
        source_elapsed=torch.zeros(count,device=device);teacher_elapsed=source_elapsed.clone()
        minima=torch.full((count,),float('inf'),device=device);reasons=[None]*count;handoffs=[];handoff_metrics={}
        def capture(env):
            if trials is None:return torch.zeros_like(failed)
            owner=env.command_manager.get_term('twist').owner
            m=env.landing_monitor;active=~failed&~success
            minima[active]=torch.minimum(minima[active],m.knee_min[active])
            bad=m.unsafe&active;unsafe.logical_or_(bad)
            stop=active&(env.termination_manager.terminated|env.termination_manager.time_outs|bad)
            for i in stop.nonzero().flatten().tolist():reasons[i]='unsafe' if bool(bad[i]) else 'termination'
            failed.logical_or_(stop)
            down=(owner==TRANSITION)&~failed
            trials.finished.logical_or_(~down)
            term=env.command_manager.get_term('motion')
            metric=tripod_endpoint_metrics(env)
            trials.record(term.time_steps>=term.motion.time_step_total-1,metric,
                env.termination_manager.terminated,torch.zeros_like(failed),safety=m.snapshot(),step=env.common_step_counter)
            lost=down&trials.finished&~trials.standing_held
            for i in lost.nonzero().flatten().tolist():reasons[i]=trials.reasons[i]
            failed.logical_or_(lost)
            tripod=(owner==TRIPOD)&~failed&~success
            teacher_elapsed.add_(tripod.to(torch.float32)*env.step_dt)
            final_good=torch.stack(tuple(tripod_conditions(metric).values()),-1).all(-1)
            done=tripod&(teacher_elapsed>=3.-1e-6)
            success.logical_or_(done&final_good)
            failed.logical_or_(done&~final_good)
            for i in (done&~final_good).nonzero().flatten().tolist():reasons[i]='tripod_teacher_unstable'
            return torch.zeros_like(failed)
        cfg.terminations['live_chain_capture']=TerminationTermCfg(func=capture)
        env=ManagerBasedRlEnv(cfg,device=device)
        try:
            monitor=install_monitor(env,gate);motion=env.command_manager.get_term('motion');twist=env.command_manager.get_term('twist')
            action=env.action_manager.get_term('joint_pos')
            action.source_teacher=AlignedKeyboardTeacher(quad_teacher,'loco_pedipulation_t',device)
            action.biped_teacher=AlignedKeyboardTeacher(biped_teacher,'pedipulation_t',device)
            with torch.inference_mode():
                obs,_=env.reset(seed=seed)
                commands=sample_biped_commands(count,device)
                twist._command.copy_(commands);twist._command[:,:3]=0.;twist.phase[:]=HOLD;twist.blend[:]=1.
                obs=env.observation_manager.compute(update_history=False);monitor.sample(0.)
                trials=LandingTrials(count,env.step_dt,device=device,cfg=gate);trials.finished[:]=True
                policy=_copy_actor_for_eval(actor,device) if actor is not None else None
                if policy is not None:policy.eval()
                for _ in range(math.ceil((2.+gate.timeout_s+3.+.2)/env.step_dt)):
                    source=(twist.owner==BIPED)&~failed&~success
                    twist._command[source,:3]=commands[source,:3]*(source_elapsed[source]/1.).clamp(0.,1.)[:,None]
                    actions=policy(TensorDict(obs,batch_size=[count]),stochastic_output=False) if policy is not None else torch.zeros(count,12,device=device)
                    obs,_,_,_,_=env.step(actions.clamp(-10.,10.))
                    source=(twist.owner==BIPED)&~failed&~success
                    source_elapsed.add_(source.to(torch.float32)*env.step_dt)
                    ids=(source&(source_elapsed>=2.-1e-6)).nonzero().flatten()
                    if len(ids):
                        differences=switch_owner(env,ids,TRANSITION)
                        for i in ids.tolist():
                            handoff_metrics[i]=dict(actual_speed_m_s=float(env.scene['robot'].data.root_link_lin_vel_w[i,:2].norm()),
                                fr_error_m=float((motion.robot_body_pos_w[i,motion.cfg.body_names.index('FR_foot')] - motion.body_pos_relative_w[i,motion.cfg.body_names.index('FR_foot')]).norm()))
                        monitor.reset(ids)
                        handoffs.append(dict(kind='BIPED_DOWN',worlds=ids.tolist(),max_abs_differences=differences))
                        trials.finished[ids]=False
                    ids=((twist.owner==TRANSITION)&trials.standing_held&~failed&~success).nonzero().flatten()
                    if len(ids):
                        differences=switch_owner(env,ids,TRIPOD)
                        handoffs.append(dict(kind='DOWN_TRIPOD',worlds=ids.tolist(),max_abs_differences=differences))
                    if bool((failed|success).all()):break
                    obs=env.observation_manager.compute(update_history=False)
            return dict(attempts=count,successes=int(success.sum()),unsafe_attempts=int(unsafe.sum()),
                fresh_live_teacher_rollouts=True,seed=seed,source_teacher_seconds=2.,tripod_teacher_seconds=3.,
                distinct_commands=len(torch.unique(commands,dim=0)),handoffs=handoffs,
                command_distribution='source_v1_half_nominal_quarter_offsets_quarter_moving_ramp1s',
                records=[dict(success=bool(success[i]),unsafe=bool(unsafe[i]),reason='success' if bool(success[i]) else reasons[i] or 'timeout',
                    knee_min_m=float(minima[i]) if torch.isfinite(minima[i]) else None,command=commands[i].tolist(),
                    handoff=handoff_metrics.get(i)) for i in range(count)])
        finally:env.close()
