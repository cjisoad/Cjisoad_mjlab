"""Measure live biped-teacher → descent → tripod-teacher chains."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]


def parse_args(argv=None):
  p=argparse.ArgumentParser(description=__doc__)
  for name in ('motion-file','entry-bank','quad-teacher','biped-teacher','output'):
    p.add_argument('--'+name,type=Path,required=True)
  p.add_argument('--checkpoint',type=Path)
  p.add_argument('--agent',choices=('trained','zero'),default='trained')
  p.add_argument('--attempts',type=int,default=128)
  p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=424242)
  args=p.parse_args(argv)
  if args.attempts<1 or args.attempts>128:p.error('Require 1..128 independent holdout trajectories')
  if args.agent=='trained' and args.checkpoint is None:p.error('trained requires --checkpoint')
  return args


def evaluate(args):
  os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MPLCONFIGDIR','/tmp/transition_down_mpl')
  os.environ.setdefault('XDG_CACHE_HOME','/tmp/transition_down_cache');sys.path.insert(0,str(ROOT))
  import torch
  from tensordict import TensorDict
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import TerminationTermCfg
  from mjlab.rl import RslRlVecEnvWrapper
  from src.tasks.transition_down_t.chain import chain_env_cfg,switch_owner,AlignedKeyboardTeacher,BIPED,TRANSITION,TRIPOD
  from src.tasks.transition_down_t.data import verify_teachers,file_sha256
  from src.tasks.transition_down_t.entry_evaluation import EntryTrials,json_safe
  from src.tasks.transition_down_t.standing import tripod_endpoint_metrics
  from src.tasks.transition_down_t.rl import DownOnPolicyRunner,down_ppo_runner_cfg
  verify_teachers(args.quad_teacher,args.biped_teacher)
  if args.output.exists():raise ValueError('Use a fresh chain output')
  torch.set_num_threads(1)
  cfg=chain_env_cfg(args.entry_bank,motion_file=args.motion_file,num_envs=args.attempts);cfg.seed=args.seed
  failed=torch.zeros(args.attempts,dtype=torch.bool,device=args.device)
  reasons=[None]*args.attempts;handoffs=[];trials=None;teacher_elapsed=None
  source_elapsed=torch.zeros(args.attempts,dtype=torch.float64,device=args.device)
  success=torch.zeros_like(failed)
  def capture(env):
    nonlocal teacher_elapsed
    twist=env.command_manager.get_term('twist');motion=env.command_manager.get_term('motion')
    manager=env.termination_manager
    causes={name:manager.get_term(name) for name in manager.active_terms if name!='chain_capture'}
    newly=manager.terminated & ~failed & ~success
    for i in newly.nonzero().flatten().tolist():
      reasons[i]='+'.join(name for name,flag in causes.items() if bool(flag[i])) or 'physical_failure'
    failed.logical_or_(newly)
    if trials is not None:
      down=twist.owner==TRANSITION
      trials.finished.logical_or_(~down)
      trials.record(motion.time_steps>=motion.motion.time_step_total-1,tripod_endpoint_metrics(env),
        manager.terminated,torch.zeros_like(failed),termination_reasons=causes,step=env.common_step_counter)
      timed=down & trials.finished & ~trials.standing_held & ~failed
      for i in timed.nonzero().flatten().tolist():reasons[i]=trials.reasons[i]
      failed.logical_or_(timed)
    if teacher_elapsed is not None:
      tripod=(twist.owner==TRIPOD)&~failed&~success
      m=tripod_endpoint_metrics(env)
      lost=tripod & ~m['fr_clear']
      for i in lost.nonzero().flatten().tolist():reasons[i]='tripod_teacher_FR_contact_or_low_clearance'
      failed.logical_or_(lost)
      teacher_elapsed.add_((tripod & ~failed).to(torch.float64)*env.step_dt)
      success.logical_or_(tripod & ~failed & (teacher_elapsed>=3.-1e-9))
    return torch.zeros_like(failed)
  cfg.terminations['chain_capture']=TerminationTermCfg(func=capture)
  env=ManagerBasedRlEnv(cfg,device=args.device)
  try:
    action=env.action_manager.get_term('joint_pos')
    action.source_teacher=AlignedKeyboardTeacher(args.quad_teacher,'loco_pedipulation_t',args.device)
    action.biped_teacher=AlignedKeyboardTeacher(args.biped_teacher,'pedipulation_t',args.device)
    motion=env.command_manager.get_term('motion');twist=env.command_manager.get_term('twist')
    actor=None
    if args.agent=='trained':
      agent=down_ppo_runner_cfg()
      runner=DownOnPolicyRunner(RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions),asdict(agent),device=args.device)
      runner.load(str(args.checkpoint),load_cfg={'actor':True},map_location=args.device)
      actor=runner.alg.actor;actor.eval()
    selected=motion.bank.sample_trials(args.attempts,split='holdout',max_speed=1.2,max_fr_error=.45)
    motion.forced_entry_indices=selected.clone()
    obs,_=env.reset(seed=args.seed)
    twist._command.copy_(motion.bank.data['command'][selected]);twist.phase[:]=3;twist.blend[:]=1.
    obs=env.observation_manager.compute(update_history=False)
    teacher_elapsed=torch.zeros_like(source_elapsed)
    trials=EntryTrials(args.attempts,env.step_dt,device=args.device)
    trials.finished[:]=True
    with torch.inference_mode():
      for _ in range(math.ceil(15.1/env.step_dt)):
        actions=actor(TensorDict(obs,batch_size=[args.attempts]),stochastic_output=False) if actor else torch.zeros(args.attempts,12,device=args.device)
        obs,_,_,_,_=env.step(actions.clamp(-10.,10.))
        source=(twist.owner==BIPED)&~failed&~success
        source_elapsed.add_(source.to(torch.float64)*env.step_dt)
        ids=(source & (source_elapsed>=2.-1e-9)).nonzero().flatten()
        if len(ids):
          differences=switch_owner(env,ids,TRANSITION)
          handoffs.append(dict(kind='BIPED_DOWN',worlds=ids.tolist(),step=env.common_step_counter,max_abs_differences=differences))
          trials.finished[ids]=False
        ids=((twist.owner==TRANSITION)&trials.standing_held&~failed&~success).nonzero().flatten()
        if len(ids):
          differences=switch_owner(env,ids,TRIPOD)
          handoffs.append(dict(kind='DOWN_TRIPOD',worlds=ids.tolist(),step=env.common_step_counter,max_abs_differences=differences))
        if bool((failed|success).all()):break
        obs=env.observation_manager.compute(update_history=False)
    for i in (~failed&~success).nonzero().flatten().tolist():reasons[i]='chain_timeout'
    trajectories=motion.bank.data['trajectory_id'][selected].cpu().tolist()
    records=trials.records()
    for i,record in enumerate(records):
      record.update(trajectory_id=trajectories[i],entry_index=int(selected[i]),chain_success=bool(success[i]),
        chain_reason='success' if bool(success[i]) else reasons[i],tripod_teacher_seconds=float(teacher_elapsed[i]),
        actual_speed=float(motion.bank.data['actual_speed'][selected[i]]),fr_error=float(motion.bank.data['fr_error'][selected[i]]))
    report=json_safe(dict(agent=args.agent,runtime_smoke_only=args.agent=='zero',attempts=args.attempts,
      distinct_trajectories=len(set(trajectories)),successes=int(success.sum()),success_rate=float(success.float().mean()),
      motion_sha256=motion.reference.sha256,entry_bank_sha256=motion.bank.sha256,
      checkpoint_sha256=file_sha256(args.checkpoint) if args.checkpoint else None,
      source_teacher_seconds=2.,tripod_teacher_required_seconds=3.,handoffs=handoffs,records=records))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('DOWN_CHAIN '+json.dumps({k:v for k,v in report.items() if k not in ('records','handoffs')}),flush=True)
    return report
  finally:env.close()

if __name__=='__main__':evaluate(parse_args())
