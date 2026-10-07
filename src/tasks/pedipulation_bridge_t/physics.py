"""Episodic physics from nominal model fields, gated by the tracking course."""
import re
import torch
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from src.tasks.pedipulation.mdp.events import restitution_damping_ratio


@requires_model_fields('body_mass','body_inertia','body_ipos','geom_friction',
  'geom_solref','dof_armature','dof_damping','dof_frictionloss',recompute=RecomputeLevel.none)
def declare_tracking_physics(env, env_ids):
  # Allocate per-world fields at startup without applying randomization.
  pass


def reset_tracking_physics(env, ids, fraction):
  robot=env.scene['robot']; action=env.action_manager.get_term('joint_pos')
  model=env.sim.model; n=len(ids)
  amplitude=fraction*env.command_manager.get_term('twist').disturbed[ids].float()[:,None]
  def sample(low,high,width=1):
    return torch.empty(n,width,device=env.device).uniform_(low,high)
  bodies=robot.indexing.body_ids; geoms=robot.indexing.geom_ids; dofs=robot.indexing.joint_v_adr
  defaults=env.sim.get_default_field
  for name,index in (('body_mass',bodies),('body_inertia',bodies),('body_ipos',bodies),
                     ('geom_friction',geoms),('geom_solref',geoms),('dof_armature',dofs),
                     ('dof_damping',dofs),('dof_frictionloss',dofs)):
    getattr(model,name)[ids[:,None],index]=defaults(name)[index][None]
  mass=defaults('body_mass')[bodies]
  factors=1.+sample(-.1,.1,len(bodies))*amplitude
  base_local=robot.find_bodies('base_link')[0][0]; base=bodies[base_local]
  added=sample(-1.,2.)*amplitude
  factors[:,base_local]=(mass[base_local]+added[:,0])/mass[base_local]
  model.body_mass[ids[:,None],bodies]=mass[None]*factors
  model.body_inertia[ids[:,None],bodies]=defaults('body_inertia')[bodies][None]*factors[...,None]
  com=sample(-.05,.05,3)*amplitude
  model.body_ipos[ids,base]=defaults('body_ipos')[base]+com
  friction=1.+(sample(.2,1.2)-1.)*amplitude
  restitution=sample(0.,.3)*amplitude
  model.geom_friction[ids[:,None],geoms,0]=friction
  model.geom_solref[ids[:,None],geoms,0]=.02
  model.geom_solref[ids[:,None],geoms,1]=restitution_damping_ratio(restitution)
  jf=sample(.01,.1)*amplitude; damping=sample(0.,.1)*amplitude
  model.dof_frictionloss[ids[:,None],dofs]=defaults('dof_frictionloss')[dofs][None]+jf
  model.dof_damping[ids[:,None],dofs]=defaults('dof_damping')[dofs][None]+damping
  kp=1.+sample(-.1,.1,12)*amplitude; kd=1.+sample(-.1,.1,12)*amplitude
  action.kp_multipliers[ids]=kp; action.kd_multipliers[ids]=kd
  action.motor_offsets[ids]=sample(-.035,.035,12)*amplitude
  configured=env.cfg.scene.entities['robot'].articulation.actuators
  for actuator in robot.actuators:
    canonical=[list(robot.joint_names).index(name) for name in actuator.target_names]
    ordered=[action.joint_ids.tolist().index(i) for i in canonical]
    nominal=next(a for a in configured if any(re.fullmatch(p,actuator.target_names[0]) for p in a.target_names_expr))
    actuator.set_gains(ids,nominal.stiffness*kp[:,ordered],nominal.damping*kd[:,ordered])
  arm=defaults('dof_armature')[dofs].mean().expand(n,1)
  action.dr_observation[ids]=torch.cat((friction,added,com,kp,kd,arm,jf,damping,restitution,restitution),-1)
  env.sim.recompute_constants(RecomputeLevel.set_const)
