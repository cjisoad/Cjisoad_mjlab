"""All-leg shell clearance and latched safety at every physical integration."""
import numpy as np
import mujoco
import torch
from scipy.spatial import ConvexHull

def knee_mesh_vertices(model):
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    geoms=[];vertices=[]
    prefix='' if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'FL_calf_joint')>=0 else 'robot/'
    for leg in ('FL','FR','RL','RR'):
        knee=data.xanchor[model.joint(f'{prefix}{leg}_calf_joint').id]
        for part in ('thigh','calf'):
            for index in (0,1):
                g=model.geom(f'{prefix}{leg}_{part}_visual_{index}').id;mesh=model.geom_dataid[g]
                start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
                v=model.mesh_vert[start:start+count].copy()
                if part=='calf':
                    world=v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
                    v=v[np.linalg.norm(world-knee,axis=1)<=.12]
                if len(v):
                    v=v[ConvexHull(v).vertices] if len(v)>4 else v
                    geoms.append(g);vertices.append(v)
    return geoms,vertices

def pack_meshes(geoms,vertices,device,dtype):
    indices=torch.tensor(np.concatenate([np.full(len(v),g) for g,v in zip(geoms,vertices)]),device=device,dtype=torch.long)
    return indices,torch.tensor(np.concatenate(vertices),device=device,dtype=dtype)

def mesh_clearance(positions,rotations,geoms,vertices):
    ids,v=pack_meshes(geoms,vertices,positions.device,positions.dtype)
    r=rotations.reshape(positions.shape[0],-1,3,3)
    return ((r[:,ids,2]*v).sum(-1)+positions[:,ids,2]).amin(-1)

class SafetyMonitor:
    def __init__(self,env,cfg):
        self.env=env;self.cfg=cfg;self.physics_samples_total=0
        g,v=knee_mesh_vertices(env.sim.mj_model)
        self.ids,self.vertices=pack_meshes(g,v,env.device,torch.float32)
        n=env.num_envs
        self.knee_min=torch.full((n,),float('inf'),device=env.device)
        self.current_knee=self.knee_min.clone()
        self.nonfoot=torch.zeros(n,dtype=torch.bool,device=env.device)
        self.fr_contact=self.nonfoot.clone()
        self.elapsed=torch.zeros(n,device=env.device)
        self.fl_duration=self.elapsed.clone()
        self.landing_time=self.knee_min.clone()
        self.samples=torch.zeros(n,dtype=torch.long,device=env.device)
    def reset(self,ids=None):
        ids=slice(None) if ids is None else ids
        for name in ('knee_min','current_knee','landing_time'):getattr(self,name)[ids]=float('inf')
        for name in ('nonfoot','fr_contact','elapsed','fl_duration','samples'):getattr(self,name)[ids]=0
    def sample(self,dt):
        self.physics_samples_total+=int(dt>0)
        env=self.env;p=env.sim.data.geom_xpos;r=env.sim.data.geom_xmat.reshape(env.num_envs,-1,3,3)
        self.current_knee.copy_(((r[:,self.ids,2]*self.vertices).sum(-1)+p[:,self.ids,2]).amin(-1)-env.scene.env_origins[:,2])
        self.knee_min.copy_(torch.minimum(self.knee_min,self.current_knee))
        # Read current solver fields directly, avoiding control-period sensor caches.
        body=env.scene['landing_nonfoot']._extract_sensor_data()
        feet=env.scene['feet_ground_contact']._extract_sensor_data()
        self.nonfoot.logical_or_((body.force.norm(dim=-1).reshape(env.num_envs,-1)>1.).any(-1))
        forces=feet.force.reshape(env.num_envs,4,-1,3).sum(2).norm(dim=-1)
        self.fr_contact.logical_or_(forces[:,1]>1.)
        self.elapsed.add_(dt);self.samples.add_(int(dt>0))
        self.fl_duration.copy_(torch.where(forces[:,0]>5.,self.fl_duration+dt,0.))
        confirmed=(self.fl_duration>=self.cfg.landing_contact_s-1e-6)&~torch.isfinite(self.landing_time)
        self.landing_time[confirmed]=self.elapsed[confirmed]-self.fl_duration[confirmed]
    def snapshot(self):
        return {k:getattr(self,k).clone() for k in ('knee_min','nonfoot','fr_contact','landing_time')}
    @property
    def unsafe(self):
        return (self.knee_min<self.cfg.knee_margin_m)|self.nonfoot|self.fr_contact|~torch.isfinite(self.current_knee)

def install_monitor(env,cfg):
    if hasattr(env,'landing_monitor'):return env.landing_monitor
    monitor=SafetyMonitor(env,cfg);env.landing_monitor=monitor
    original_step=env.sim.step
    def step():
        original_step()
        # Refresh post-integration geometry/contacts, including the final substep.
        env.sim.forward()
        monitor.sample(env.physics_dt)
    env.sim.step=step
    action=env.action_manager.get_term('joint_pos');original_reset=action.reset
    def reset(env_ids=None):
        original_reset(env_ids);monitor.reset(env_ids)
    action.reset=reset
    return monitor

def knee_penalty(env):
    m=env.landing_monitor
    return ((m.cfg.knee_target_m-m.current_knee).clamp(min=0)/.025).square().clamp(max=4.)

def landing_failure(env):
    if not hasattr(env,'landing_monitor'):return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    m=env.landing_monitor
    return (m.knee_min<m.cfg.training_knee_stop_m)|m.nonfoot|m.fr_contact|~torch.isfinite(m.current_knee)
