"""Four first-frame disturbance groups; separate optional startup/noise trial."""
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
  p.add_argument('--agent',choices=('trained','zero'),default='trained')
  p.add_argument('--checkpoint-file',type=Path)
  p.add_argument('--motion-file',type=Path,required=True)
  p.add_argument('--episodes-per-group',type=int,default=100)
  p.add_argument('--startup-randomization',action='store_true')
  p.add_argument('--observation-noise',action='store_true')
  p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=42)
  p.add_argument('--output',type=Path,required=True)
  args=p.parse_args(argv)
  if args.episodes_per_group<1:p.error('Require positive episodes-per-group')
  if args.agent=='trained' and args.checkpoint_file is None:p.error('trained requires --checkpoint-file')
  return args


def evaluate(args):
  os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MPLCONFIGDIR','/tmp/transition_down_mpl')
  os.environ.setdefault('XDG_CACHE_HOME','/tmp/transition_down_cache');sys.path.insert(0,str(ROOT))
  import torch
  from tensordict import TensorDict
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import TerminationTermCfg
  from mjlab.rl import RslRlVecEnvWrapper
  from src.tasks.transition_down_t.env_cfg import down_env_cfg
  from src.tasks.transition_down_t.rl import DownOnPolicyRunner,down_ppo_runner_cfg
  from src.tasks.transition_down_t.entry_evaluation import EntryTrials,json_safe,_summary
  from src.tasks.transition_down_t.standing import tripod_endpoint_metrics
  from src.tasks.transition_down_t.data import file_sha256
  if args.output.exists():raise ValueError('Use a fresh full-start evaluation output')
  torch.set_num_threads(1);groups={}
  for group_index,group in enumerate(('nominal','base','joints','combined')):
    cfg=down_env_cfg(play=True);cfg.scene.num_envs=args.episodes_per_group;cfg.seed=args.seed+group_index
    training=down_env_cfg();cmd=cfg.commands['motion']
    cmd.motion_file=str(args.motion_file.resolve());cmd.debug_vis=False;cmd.sampling_mode='start'
    cmd.pose_range=training.commands['motion'].pose_range if group in ('base','combined') else {}
    cmd.velocity_range=training.commands['motion'].velocity_range if group in ('base','combined') else {}
    cmd.joint_position_range=training.commands['motion'].joint_position_range if group in ('joints','combined') else (0.,0.)
    if args.startup_randomization:cfg.events=training.events
    cfg.observations['actor'].enable_corruption=args.observation_noise
    trials=None
    def capture(env):
      if trials is not None:
        term=env.command_manager.get_term('motion');manager=env.termination_manager
        causes={name:manager.get_term(name) for name in manager.active_terms if name!='full_start_capture'}
        trials.record(term.time_steps>=term.motion.time_step_total-1,tripod_endpoint_metrics(env),
          manager.terminated,manager.time_outs,termination_reasons=causes,step=env.common_step_counter)
      return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    cfg.terminations['full_start_capture']=TerminationTermCfg(func=capture)
    env=ManagerBasedRlEnv(cfg,device=args.device)
    try:
      actor=None
      if args.agent=='trained':
        agent=down_ppo_runner_cfg()
        runner=DownOnPolicyRunner(RslRlVecEnvWrapper(env,clip_actions=10.),asdict(agent),device=args.device)
        runner.load(str(args.checkpoint_file),load_cfg={'actor':True},map_location=args.device)
        actor=runner.alg.actor;actor.eval()
      obs,_=env.reset(seed=args.seed+group_index*100000)
      term=env.command_manager.get_term('motion')
      if not bool((term.time_steps==0).all()):raise AssertionError('All full-start attempts must start at frame0')
      initial=dict(root_position_delta=(term.robot_anchor_pos_w-term.anchor_pos_w).cpu().tolist(),
        joint_position_delta=(term.robot_joint_pos-term.joint_pos).cpu().tolist())
      trials=EntryTrials(args.episodes_per_group,env.step_dt,device=args.device)
      with torch.inference_mode():
        for _ in range(math.ceil(10./env.step_dt)):
          actions=actor(TensorDict(obs,batch_size=[env.num_envs]),stochastic_output=False) if actor else torch.zeros(env.num_envs,12,device=args.device)
          obs,_,_,_,_=env.step(actions.clamp(-10.,10.))
          if bool(trials.finished.all()):break
      if not bool(trials.finished.all()):raise RuntimeError('Uncaptured full-start attempts')
      records=trials.records()
      for i,record in enumerate(records):record['initial']={key:value[i] for key,value in initial.items()}
      groups[group]=dict(**_summary(records),records=records)
    finally:env.close()
  report=json_safe(dict(agent=args.agent,runtime_smoke_only=args.agent=='zero',seed=args.seed,
    motion_sha256=file_sha256(args.motion_file),checkpoint_sha256=file_sha256(args.checkpoint_file) if args.checkpoint_file else None,
    startup_randomization=args.startup_randomization,observation_noise=args.observation_noise,
    all_attempts_start_frame0=True,strict_diagnostic='uninterrupted 1s world-foot RMS <0.06m',groups=groups))
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  print('DOWN_FULL_START '+json.dumps({g:{k:v for k,v in r.items() if k!='records'} for g,r in groups.items()}),flush=True)
  return report

if __name__=='__main__':evaluate(parse_args())
