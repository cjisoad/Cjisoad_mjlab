"""Evaluate reference tracking or a real source→student→biped rollout."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bridge-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/bridge-cache')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import torch


def configure_case(term, case):
  term.course.enabled=False
  term.forced_start='full'
  term.forced_bank=0 if case=='nominal' else None
  term.forced_live_probability=1. if case.startswith('live_') else 0.
  term.course.stage=2 if case.startswith('live_') else (1 if case=='reference_disturbed' else 0)
  term.course.level=4
  term.forced_disturbed=False if case in ('nominal','fr_diversity') else None
  term.cfg.verify_handoff=case=='live_handoff'
  if case.startswith('live_'):
    # Each world keeps one class for all its quota episodes.
    term.forced_moving=(torch.arange(term.num_envs,device=term.device)%2)==1


def physical_snapshot(raw, term, world):
  robot=raw.scene['robot']; data=robot.data
  action=raw.action_manager.get_term('joint_pos')
  def serial(tensor):
    def clean(value):
      if isinstance(value,list): return [clean(v) for v in value]
      return None if isinstance(value,float) and not math.isfinite(value) else value
    return clean(tensor.detach().cpu().tolist())
  state=dict(root_state=serial(torch.cat((data.root_link_pos_w[world],data.root_link_quat_w[world],
    data.root_link_lin_vel_w[world],data.root_link_ang_vel_w[world]))),
    joint_pos=serial(data.joint_pos[world,term.joint_ids]),
    joint_vel=serial(data.joint_vel[world,term.joint_ids]),
    raw_action=serial(action.raw_action[world]),previous_action=serial(action.previous_action[world]),
    motor_offsets=serial(action.motor_offsets[world]),delay_steps=serial(action.delay_steps[world]),
    kp_multipliers=serial(action.kp_multipliers[world]),kd_multipliers=serial(action.kd_multipliers[world]),
    fr_error=serial(term.fr_error[world]),command=serial(term.command[world]),
    pending_initial_impulse=serial(term.initial_impulse[world]*term.pending_initial_impulse[world]),
    physics={})
  for name,index in (('body_mass',robot.indexing.body_ids),('body_inertia',robot.indexing.body_ids),
                     ('body_ipos',robot.indexing.body_ids),('geom_friction',robot.indexing.geom_ids),
                     ('geom_solref',robot.indexing.geom_ids),('dof_armature',robot.indexing.joint_v_adr),
                     ('dof_damping',robot.indexing.joint_v_adr),('dof_frictionloss',robot.indexing.joint_v_adr)):
    state['physics'][name]=serial(getattr(raw.sim.model,name)[world,index])
  return state


@torch.inference_mode()
def evaluate(env, policy, episodes_per_world):
  term=env.unwrapped.command_manager.get_term('twist')
  original=term.advance
  try: return _evaluate_impl(env,policy,episodes_per_world)
  finally: term.advance=original


def _evaluate_impl(env, policy, episodes_per_world):
  term=env.unwrapped.command_manager.get_term('twist')
  obs,_=env.reset()
  count=env.num_envs
  done_count=torch.zeros(count,dtype=torch.long,device=env.device)
  records=[]
  samples=torch.zeros(count,device=env.device)
  fr_square=torch.zeros_like(samples)
  entered=torch.zeros(count,dtype=torch.bool,device=env.device)
  tracking_held=torch.zeros_like(entered)
  start_velocity=torch.zeros(count,3,device=env.device)
  elapsed=torch.zeros_like(samples)
  entry_states=[None]*count; handoff_states=[None]*count; final_states=[None]*count
  classes=term.moving.clone()
  original_advance=term.advance
  def capture_advance():
    before=term.owner.clone(); previous=term._last_advance_step
    original_advance()
    if previous==term._last_advance_step: return
    active=done_count<episodes_per_world
    handoff=(before==BRIDGE)&(term.owner==VERIFY)&active
    ended=(term.outcome_event!=0)&active
    for world in handoff.nonzero().flatten().tolist():
      handoff_states[world]=physical_snapshot(env.unwrapped,term,world)
    for world in ended.nonzero().flatten().tolist():
      final_states[world]=physical_snapshot(env.unwrapped,term,world)
  from src.tasks.pedipulation_bridge_t.commands import BRIDGE,VERIFY
  term.advance=capture_advance
  limit=(env.unwrapped.max_episode_length+5)*episodes_per_world
  for _ in range(limit):
    active=done_count<episodes_per_world
    bridge=active & (term.owner==BRIDGE)
    starting=bridge & ~entered
    for world in starting.nonzero().flatten().tolist():
      entry_states[world]=physical_snapshot(env.unwrapped,term,world)
    data=env.unwrapped.scene['robot'].data
    velocity=data.root_link_lin_vel_w+term.initial_impulse[:,:3]*term.pending_initial_impulse[:,None]
    start_velocity[starting]=velocity[starting]
    entered|=bridge
    tracking_held|=active & (term.owner==VERIFY)
    samples+=bridge
    fr_square+=bridge*term.fr_error.square().sum(-1)
    elapsed+=active*env.unwrapped.step_dt
    obs,_,dones,_=env.step(policy(obs))
    events=term.outcome_event
    ended=active & (dones.bool() | (events!=0))
    for world in ended.nonzero().flatten().tolist():
      event=int(events[world])
      records.append(dict(world=world,episode=int(done_count[world]),outcome=event,
        reason=int(term.event_reason[world]),source_entered=bool(entered[world]),
        tracking_held_before_handoff=bool(tracking_held[world]),
        first_stand_seconds=float(term.completed_first_stand_time[world]) if torch.isfinite(term.completed_first_stand_time[world]) else None,
        duration_seconds=float(elapsed[world]),
        initial_linear_speed_m_s=float(start_velocity[world].norm()),
        command_class='moving' if bool(classes[world]) else 'stationary',
        entry_state=entry_states[world],handoff_state=handoff_states[world],terminal_state=final_states[world],
        transition_samples=int(samples[world]),
        fr_square_sum=float(fr_square[world])))
    done_count[ended]+=1
    samples[ended]=0.; fr_square[ended]=0.; elapsed[ended]=0.
    entered[ended]=False; tracking_held[ended]=False
    for world in ended.nonzero().flatten().tolist():
      entry_states[world]=None; handoff_states[world]=None; final_states[world]=None
    classes[ended]=term.moving[ended]
    if (done_count>=episodes_per_world).all(): break
  else: raise RuntimeError('Evaluation quota was not completed; no partial success report')
  term.advance=original_advance
  entered_count=sum(r['source_entered'] for r in records)
  successes=sum(r['outcome']==1 for r in records)
  total_samples=sum(r['transition_samples'] for r in records)
  def summarize(rows):
    return dict(attempts=len(rows),successes=sum(r['outcome']==1 for r in rows),
      success_rate=sum(r['outcome']==1 for r in rows)/len(rows) if rows else None)
  groups={label:summarize([r for r in records if r['command_class']==label]) for label in ('stationary','moving')}
  speeds={label:summarize([r for r in records if r['source_entered'] and low<=r['initial_linear_speed_m_s']<high])
    for label,low,high in (('below_0.1',0.,.1),('0.1_to_0.25',.1,.25),('at_least_0.25',.25,float('inf')))}
  return dict(episodes=len(records),sampling='fixed_quota_per_world',policy='deterministic_mean',
    successes=successes,success_rate=successes/len(records),
    source_yield=entered_count/len(records),
    success_rate_given_entry=successes/entered_count if entered_count else 0.,
    fr_rms_error_m=(sum(r['fr_square_sum'] for r in records)/max(1,total_samples))**.5,
    stood_within5s=sum(r['first_stand_seconds'] is not None and r['first_stand_seconds']<=5. for r in records),
    by_command_class=groups,by_start_speed=speeds,
    tracking_held_but_handoff_failed=sum(r['handoff_state'] is not None and r['outcome']!=1 for r in records),
    records=records)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--case',choices=('nominal','fr_diversity','reference_disturbed','live_source','live_handoff'),default='nominal')
  parser.add_argument('--device',default='cuda:0')
  parser.add_argument('--num-envs',type=int,default=32)
  parser.add_argument('--episodes-per-world',type=int,default=4)
  parser.add_argument('--seed',type=int,default=123)
  parser.add_argument('--viewer',choices=('none','native'),default='none')
  parser.add_argument('--output',type=Path,default=Path('outputs/bridge_tracking_evaluation.json'))
  args=parser.parse_args()
  if args.num_envs<1 or args.episodes_per_world<1: parser.error('Positive world count and episode quota required')
  if args.case.startswith('live_') and args.num_envs%2:
    parser.error('Live evaluation requires an even world count for exact50/50 stationary/moving quotas')
  if args.viewer=='none': os.environ.setdefault('MUJOCO_GL','disable')
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from src.tasks.pedipulation_bridge_t.env_cfg import bridge_env_cfg
  from src.tasks.pedipulation_bridge_t.rl import bridge_ppo_runner_cfg,BridgeOnPolicyRunner
  torch.set_num_threads(1)
  cfg=bridge_env_cfg(); cfg.scene.num_envs=args.num_envs; cfg.seed=args.seed
  cfg.commands['twist'].curriculum_enabled=False
  cfg.commands['twist'].debug_vis=args.viewer=='native'
  raw=ManagerBasedRlEnv(cfg,device=args.device)
  try:
    env=RslRlVecEnvWrapper(raw,clip_actions=10.)
    runner=BridgeOnPolicyRunner(env,asdict(bridge_ppo_runner_cfg()),device=args.device)
    runner.load(str(args.checkpoint),map_location=args.device)
    configure_case(raw.command_manager.get_term('twist'),args.case)
    policy=runner.get_inference_policy(device=args.device)
    if args.viewer=='native':
      env.reset()
      from mjlab.viewer import NativeMujocoViewer
      NativeMujocoViewer(env,policy).run()
      return
    result=evaluate(env,policy,args.episodes_per_world)
    result.update(case=args.case,seed=args.seed,checkpoint=str(args.checkpoint.resolve()),
      teacher_handoff_tested=args.case=='live_handoff',
      success_definition='actual_biped_handoff' if args.case=='live_handoff' else 'reference_rise_and_current_one_second_hold')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('BRIDGE_TRACKING_EVALUATION '+json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
  finally: raw.close()


if __name__=='__main__': main()
