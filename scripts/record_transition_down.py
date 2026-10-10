"""Record one deterministic learned descent, including applied FR target dynamics."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--motion-file',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
  p.add_argument('--group',choices=('nominal','combined'),default='nominal');p.add_argument('--seed',type=int,default=42)
  args=p.parse_args()
  if args.output.exists():raise ValueError('Use a fresh recording directory')
  os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('OMP_NUM_THREADS','1')
  os.environ.setdefault('MPLCONFIGDIR','/tmp/transition_down_mpl');os.environ.setdefault('XDG_CACHE_HOME','/tmp/transition_down_cache')
  sys.path.insert(0,str(ROOT))
  import imageio.v2 as imageio
  import numpy as np
  import torch
  from tensordict import TensorDict
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import TerminationTermCfg
  from mjlab.rl import RslRlVecEnvWrapper
  from src.tasks.transition_down_t.env_cfg import down_env_cfg
  from src.tasks.transition_down_t.rl import DownOnPolicyRunner,down_ppo_runner_cfg
  from src.tasks.transition_down_t.standing import tripod_endpoint_metrics
  from src.tasks.transition_down_t.entry_evaluation import EntryTrials,json_safe
  torch.set_num_threads(1);args.output.mkdir(parents=True)
  cfg=down_env_cfg(play=True);cfg.scene.num_envs=1;cfg.seed=args.seed
  cfg.commands['motion'].motion_file=str(args.motion_file.resolve());cfg.commands['motion'].debug_vis=False
  if args.group=='combined':
    original=down_env_cfg().commands['motion'];cmd=cfg.commands['motion']
    cmd.pose_range=original.pose_range;cmd.velocity_range=original.velocity_range;cmd.joint_position_range=original.joint_position_range
  cfg.viewer.width=960;cfg.viewer.height=720;cfg.viewer.max_extra_envs=0
  trials=None
  def capture(env):
    if trials is not None:
      term=env.command_manager.get_term('motion');manager=env.termination_manager
      trials.record(term.time_steps>=350,tripod_endpoint_metrics(env),manager.terminated,manager.time_outs,step=env.common_step_counter)
    return torch.zeros(1,dtype=torch.bool,device=env.device)
  cfg.terminations['record_capture']=TerminationTermCfg(func=capture)
  env=ManagerBasedRlEnv(cfg,device=args.device,render_mode='rgb_array')
  try:
    agent=down_ppo_runner_cfg();runner=DownOnPolicyRunner(RslRlVecEnvWrapper(env,clip_actions=10.),asdict(agent),device=args.device)
    runner.load(str(args.checkpoint),load_cfg={'actor':True},map_location=args.device)
    actor=runner.alg.actor;actor.eval();obs,_=env.reset(seed=args.seed)
    action=env.action_manager.get_term('joint_pos');trials=EntryTrials(1,env.step_dt,device=args.device)
    samples=[];writer=imageio.get_writer(args.output/'policy.mp4',fps=50,codec='libx264',ffmpeg_log_level='error')
    original_apply=action.apply_actions
    substep_samples=[]
    def trace_apply():
      previous=action.target_limiter.velocity.clone()
      original_apply()
      substep_samples.append(np.concatenate((action.target_limiter.position[0].cpu().numpy(),
        action.target_limiter.velocity[0].cpu().numpy(),((action.target_limiter.velocity-previous)/env.physics_dt)[0].cpu().numpy())))
    action.apply_actions=trace_apply
    try:
      with torch.inference_mode():
        for step in range(500):
          writer.append_data(env.render())
          requested=actor(TensorDict(obs,batch_size=[1]),stochastic_output=False).clamp(-10.,10.)
          obs,_,terminated,truncated,_=env.step(requested)
          samples.append(np.concatenate(([step*env.step_dt],requested[0].cpu().numpy(),action.raw_action[0].cpu().numpy())))
          if bool(trials.finished[0] or terminated[0] or truncated[0]):break
    finally:writer.close()
    np.savez_compressed(args.output/'actions.npz',control=np.array(samples),substep_targets=np.array(substep_samples),
      control_dt=env.step_dt,physics_dt=env.physics_dt)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=np.array(substep_samples);time=np.arange(len(values))*env.physics_dt
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for i,label in enumerate(('FR target (rad)','FR velocity (rad/s)','FR acceleration (rad/s2)')):
      axes[i].plot(time,values[:,12*i+3:12*i+6]);axes[i].set_ylabel(label);axes[i].grid(True)
    axes[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(args.output/'fr_dynamics.png');plt.close(fig)
    report=json_safe(dict(checkpoint=str(args.checkpoint.resolve()),group=args.group,seed=args.seed,records=trials.records(),
      steps=len(samples),FR_substep_velocity_max=float(abs(values[:,15:18]).max()),
      FR_substep_acceleration_max=float(abs(values[:,27:30]).max())))
    (args.output/'record.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  finally:env.close()

if __name__=='__main__':main()
