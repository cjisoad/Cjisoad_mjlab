"""Evaluate the frozen biped from audited static endpoints under shared DR.

This is an evaluation-only initialization baseline. It does not test a trained
transition policy, and no simulation state is transplanted during training.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bridge-biped-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/bridge-cache')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


class FirstOutcomeStats:
  """Accumulate post-physics samples through the first outcome per world."""
  def __init__(self,num_envs,device='cpu',horizon_seconds=2.5):
    if num_envs<1 or not math.isfinite(horizon_seconds) or horizon_seconds<=0:
      raise ValueError('positive world count and finite positive horizon required')
    self.horizon_seconds=horizon_seconds
    self.active=torch.ones(num_envs,device=device,dtype=torch.bool)
    self.samples=torch.zeros(num_envs,device=device,dtype=torch.long)
    self.outcomes=torch.zeros_like(self.samples); self.reasons=torch.zeros_like(self.samples)
    self.duration=torch.zeros(num_envs,device=device,dtype=torch.float64)
    self.fr_square=torch.zeros(num_envs,3,device=device,dtype=torch.float64)
    self.contacts=torch.zeros(num_envs,4,device=device,dtype=torch.float64)
    self.gravity=torch.zeros(num_envs,3,device=device,dtype=torch.float64)
    self.height=torch.zeros(num_envs,device=device,dtype=torch.float64)
    self.height_min=torch.full((num_envs,),float('inf'),device=device,dtype=torch.float64)
    self.height_max=torch.full((num_envs,),-float('inf'),device=device,dtype=torch.float64)
    self.physical_failure=torch.zeros(num_envs,device=device,dtype=torch.bool)
    self.gate_failures={}

  def observe(self,*,fr_error,contacts,gravity,height,gate_conditions,
              physical_failure,events,reasons,dt):
    live=self.active.clone()
    self.samples+=live.long(); self.duration+=live*dt
    self.fr_square+=torch.nan_to_num(fr_error.double(),nan=100.,posinf=100.,neginf=-100.).square()*live[:,None]
    self.contacts+=contacts.double()*live[:,None]
    self.gravity+=torch.nan_to_num(gravity.double())*live[:,None]
    safe_height=torch.nan_to_num(height.double())
    self.height+=safe_height*live
    self.height_min=torch.where(live,torch.minimum(self.height_min,safe_height),self.height_min)
    self.height_max=torch.where(live,torch.maximum(self.height_max,safe_height),self.height_max)
    self.physical_failure |= physical_failure & live
    for name,valid in gate_conditions.items():
      if name not in self.gate_failures:
        self.gate_failures[name]=torch.zeros_like(self.samples)
      self.gate_failures[name]+=((~valid)&live).long()
    finished=live & (events!=0)
    self.outcomes[finished]=events[finished]; self.reasons[finished]=reasons[finished]
    self.active[finished]=False

  def records(self):
    count=self.samples.clamp_min(1).double()
    rms_axis=(self.fr_square/count[:,None]).sqrt().cpu().tolist()
    rms_norm=(self.fr_square.sum(-1)/count).sqrt().cpu().tolist()
    contact_fraction=(self.contacts/count[:,None]).cpu().tolist()
    gravity=(self.gravity/count[:,None]).cpu().tolist()
    heights=(self.height/count).cpu().tolist()
    minimum=self.height_min.cpu().tolist(); maximum=self.height_max.cpu().tolist()
    duration=self.duration.cpu().tolist(); physical=self.physical_failure.cpu().tolist()
    outcomes=self.outcomes.cpu().tolist(); reasons=self.reasons.cpu().tolist()
    gates={name:(failures/count).cpu().tolist() for name,failures in self.gate_failures.items()}
    return [dict(world=i,outcome=outcomes[i],reason_code=reasons[i],duration_s=duration[i],
      samples=int(self.samples[i]),fr_rms_error_m=rms_norm[i],fr_rms_axis_m=rms_axis[i],
      contact_fraction_FL_FR_RL_RR=contact_fraction[i],mean_projected_gravity_b=gravity[i],
      mean_height_m=heights[i],min_height_m=minimum[i],max_height_m=maximum[i],
      physical_failure_observed=physical[i],
      reached_validation_horizon=duration[i]>=self.horizon_seconds-1e-5 and not physical[i],
      censored_before_horizon=duration[i]<self.horizon_seconds-1e-5 and not physical[i],
      state_gate_failure_fraction={name:values[i] for name,values in gates.items()})
      for i in range(len(outcomes))]


def _summary(records):
  count=len(records)
  return dict(worlds=count,successes=sum(row['outcome']==1 for row in records),
    success_rate=sum(row['outcome']==1 for row in records)/count,
    reached_horizon_rate=sum(row['reached_validation_horizon'] for row in records)/count,
    physical_failure_rate=sum(row['physical_failure_observed'] for row in records)/count,
    mean_fr_rms_error_m=sum(row['fr_rms_error_m'] for row in records)/count,
    mean_duration_s=sum(row['duration_s'] for row in records)/count,
    outcome_reasons={reason:sum(row['reason']==reason for row in records)
                    for reason in sorted({row['reason'] for row in records})},
    mean_state_gate_failure_fraction={name:sum(row['state_gate_failure_fraction'].get(name,0.)
      for row in records)/count for name in records[0]['state_gate_failure_fraction']})


def summarize_records(records):
  """Report command class and endpoint height separately, with explicit counts."""
  if not records:
    raise ValueError('at least one evaluated world is required')
  classes={name:_summary([row for row in records if row['command_class']==name])
           for name in sorted({row['command_class'] for row in records})}
  height_groups=[]
  for kind in sorted(classes):
    heights=sorted({round(row['h_end_m'],6) for row in records if row['command_class']==kind})
    for height in heights:
      selected=[row for row in records if row['command_class']==kind and round(row['h_end_m'],6)==height]
      height_groups.append(dict(command_class=kind,h_end_m=height,**_summary(selected)))
  return dict(overall=_summary(records),by_command_class=classes,
              by_command_class_and_end_height=height_groups)


def state_snapshot(env,term):
  data=env.scene['robot'].data
  height=data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
  gravity=data.projected_gravity_b
  contacts=term.contacts
  error=term.fr_error
  gates=dict(height=height>.44,gravity_x=gravity[:,0]<-.8,gravity_z=gravity[:,2].abs()<.35,
    angular_velocity=data.root_link_ang_vel_b.norm(dim=-1)<2.,front_off=~contacts[:,:2].any(-1),
    linear_velocity=data.root_link_lin_vel_w.norm(dim=-1)<term.cfg.handoff_linear_speed,
    recent_rear_support=(term.rear_contact_age<term.cfg.recent_rear_seconds).all(-1),
    fr_tracking=error.norm(dim=-1)<term.cfg.handoff_fr_error,joint_velocity=data.joint_vel[:,term.joint_ids].norm(dim=-1)<20.)
  return dict(fr_error=error,contacts=contacts,gravity=gravity,height=height,
    gate_conditions=gates,physical_failure=term._physical_failure(),
    events=term.outcome_event,reasons=term.event_reason)


def initialize_static_baseline(env):
  """Evaluation-only endpoint initialization, preserving current physics and q_des."""
  from src.tasks.pedipulation_bridge_t.commands import VERIFY
  from src.tasks.pedipulation_bridge_t.experts import JOINT_NAMES
  from src.tasks.pedipulation_transition_t.motion import make_model
  term=env.command_manager.get_term('twist')
  action=env.action_manager.get_term('joint_pos')
  robot=env.scene['robot']
  n=env.num_envs
  world=torch.arange(n,device=env.device)
  bank=world%len(term.paths.starts)
  moving=torch.zeros(n,device=env.device,dtype=torch.bool)
  witness=torch.as_tensor(term.paths.data['qpos_witnesses'],device=env.device,dtype=torch.float32)[bank,-1,-1]
  # Witness storage follows the reference model's native qpos order. Resolve
  # named joints before writing the common environment's canonical action IDs.
  model=make_model()
  q_indices=[int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
  q=witness[:,q_indices]
  pose=witness[:,:7].clone(); pose[:,:3]+=env.scene.env_origins
  robot.write_root_link_pose_to_sim(pose)
  robot.write_root_link_velocity_to_sim(torch.zeros(n,6,device=env.device))
  robot.write_joint_state_to_sim(q,torch.zeros_like(q),joint_ids=action.joint_ids)
  history=(q-action.default_angles-action.motor_offsets)/action.cfg.scale
  action._raw_action.copy_(history); action.previous_action.copy_(history)
  action.previous_joint_vel.zero_(); action.delay_steps.zero_()
  robot.set_joint_position_target(q,joint_ids=action.joint_ids)
  initial_target=action.default_angles+action.cfg.scale*history+action.motor_offsets
  torch.testing.assert_close(initial_target,q,atol=1e-6,rtol=0.)
  term.low_offset.copy_(term.paths.starts[bank]); term.end_offset.copy_(term.paths.ends[bank])
  term.h_start[:]=term.paths.h_start_range[1]; term.h_end.copy_(term.end_offset[:,2])
  ratio=(term.h_start-term.low_offset[:,2])/(term.h_end-term.low_offset[:,2])
  term.start_offset.copy_(torch.lerp(term.low_offset,term.end_offset,ratio[:,None]))
  term.moving.copy_(moving)
  term.requested_velocity.zero_()
  term.owner[:]=VERIFY; term.action_owner[:]=VERIFY
  term.live_start[:]=True; term.start_time.zero_(); term.yaw.zero_()
  term.bridge_elapsed[:]=term.cfg.bridge_rise_seconds/term.reference_speed
  term.verify_elapsed.zero_(); term.tail_steps.zero_(); term.attempt_elapsed.zero_()
  term.invalid_elapsed.zero_(); term.candidate_elapsed.zero_()
  term.outcome_event.zero_(); term.event_reason.zero_()
  term._last_advance_step=env.common_step_counter
  term._update_command()
  env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(dt=0.); env.sim.sense()
  term.rear_contact_age[:]=torch.where(term.contacts[:,2:],0.,1.)
  return bank,moving


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device',default='cuda:0')
  parser.add_argument('--num-envs',type=int,default=64)
  parser.add_argument('--seed',type=int,default=20261006)
  parser.add_argument('--output',type=Path,default=Path('outputs/bridge_biped_baseline.json'))
  args=parser.parse_args()
  if args.num_envs<2:
    parser.error('use at least two worlds for the standing endpoint baseline')
  if args.device=='cpu':
    os.environ['CUDA_VISIBLE_DEVICES']=''
  from mjlab.envs import ManagerBasedRlEnv
  from src.tasks.pedipulation_bridge_t.env_cfg import bridge_env_cfg
  from src.tasks.pedipulation_bridge_t.commands import EVENT_REASON_NAMES
  from src.tasks.pedipulation_bridge_t.paths import PATH_ASSET
  torch.set_num_threads(1)
  cfg=bridge_env_cfg()
  cfg.scene.num_envs=args.num_envs; cfg.seed=args.seed
  cfg.commands['twist'].debug_vis=False
  env=ManagerBasedRlEnv(cfg,device=args.device)
  try:
    env.reset(seed=args.seed)
    action=env.action_manager.get_term('joint_pos')
    term=env.command_manager.get_term('twist')
    experts=action._ensure_experts()
    frozen_before={name:{key:value.detach().clone() for key,value in actor.state_dict().items()}
      for name,actor in experts.actors.items()}
    physics_before={name:getattr(env.sim.model,name).clone()
      for name in ('body_mass','body_inertia','body_ipos','geom_friction','dof_armature')}
    offsets_before=action.motor_offsets.clone()
    with torch.inference_mode():
      bank,moving=initialize_static_baseline(env)
      initial=state_snapshot(env,term)
      initial_contacts=initial['contacts'].cpu().tolist()
      initial_gravity=initial['gravity'].cpu().tolist()
      initial_height=initial['height'].cpu().tolist()
      initial_errors=initial['fr_error'].norm(dim=-1).cpu().tolist()
      initial_gates={name:value.cpu().tolist() for name,value in initial['gate_conditions'].items()}
      ends=term.end_offset.cpu().tolist()
      commands=term.command.cpu().tolist()
      stats=FirstOutcomeStats(args.num_envs,env.device,term.cfg.verify_seconds)
      original_advance=term.advance
      def observed_advance():
        previous=term._last_advance_step
        original_advance()
        if previous!=term._last_advance_step:
          # Called by termination before ManagerBasedRlEnv resets a finished
          # world. Its final physical sample is retained, never its reset pose.
          stats.observe(**state_snapshot(env,term),dt=env.step_dt)
      term.advance=observed_advance
      maximum=math.ceil((term.cfg.verify_seconds+term.cfg.invalid_seconds+.1)/env.step_dt)
      for _ in range(maximum):
        env.step(torch.zeros(args.num_envs,12,device=env.device))
        if not stats.active.any():
          break
      if stats.active.any():
        raise RuntimeError('finite biped validation ended without one outcome per world')
    rows=stats.records()
    for row in rows:
      i=row['world']
      row.update(path_index=int(bank[i]),h_start_m=term.paths.h_start_range[1],h_end_m=ends[i][2],
        command_class='moving' if bool(moving[i]) else 'stationary',command=commands[i],
        reason=EVENT_REASON_NAMES[row['reason_code']],initial_fr_error_m=initial_errors[i],
        initial_contacts_FL_FR_RL_RR=initial_contacts[i],initial_projected_gravity_b=initial_gravity[i],
        initial_height_m=initial_height[i],initial_state_gate={name:values[i] for name,values in initial_gates.items()})
    for name,before in physics_before.items():
      torch.testing.assert_close(getattr(env.sim.model,name).clone(),before,rtol=0.,atol=0.)
    torch.testing.assert_close(action.motor_offsets,offsets_before,rtol=0.,atol=0.)
    for name,actor in experts.actors.items():
      assert not actor.training and all(not p.requires_grad and p.grad is None for p in actor.parameters())
      for key,value in actor.state_dict().items():
        torch.testing.assert_close(value,frozen_before[name][key],rtol=0.,atol=0.)
    checkpoint=Path(action.cfg.biped_checkpoint)
    record=dict(kind='frozen_biped_static_endpoint_initialization_baseline',
      trained_handoff_evaluation=False,training_performed=False,evaluation_only_state_initialization=True,
      initialization='final audited qpos witness; zero physical velocities; action history preserves q_des including motor offsets',
      physical_configuration='common quadruped teacher model; nominal course0 physics',
      physics_and_motor_offsets_preserved=True,frozen_experts_unchanged=True,
      device=args.device,seed=args.seed,num_envs=args.num_envs,horizon_seconds=term.cfg.verify_seconds,
      bridge_actor_dim=84,bridge_critic_dim=172,frozen_actor_dim=51,checkpoint=str(checkpoint.resolve()),
      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      path_asset=str(PATH_ASSET.resolve()),path_sha256=hashlib.sha256(PATH_ASSET.read_bytes()).hexdigest(),
      survival_definition='reached validation horizon without physical failure; early state-gate rejection is censored',
      sampling='standing validation across cyclic path endpoints; zero initial velocity; first outcome only',
      verification_velocity=list(term.cfg.verification_velocity),
      static_audit_note='Static witnesses use nominal course0 physics; static reachability alone does not prove dynamic teacher stability.',
      **summarize_records(rows),worlds=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('BRIDGE_BIPED_BASELINE '+json.dumps({key:value for key,value in record.items() if key!='worlds'}),flush=True)
  finally:
    env.close()


if __name__=='__main__':
  main()
