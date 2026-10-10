import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/transition_down_mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/transition_down_cache')
import argparse
p=argparse.ArgumentParser(description='Real entry PPO, resume, and 32 independent zero-policy trials')
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',type=__import__('pathlib').Path,required=True)
p.add_argument('--device',default='cpu')
p.add_argument('--entry-bank',required=True)
p.add_argument('--motion-file',required=True)
args=p.parse_args()
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.transition_down_t.data import DEFAULT_ENTRY_BANK
from src.tasks.transition_down_t.entry_course import down_entry_env_cfg
from src.tasks.transition_down_t.entry_rl import DownEntryOnPolicyRunner
from src.tasks.transition_down_t.entry_evaluation import evaluate_entries
from src.tasks.transition_down_t.rl import down_ppo_runner_cfg

torch.set_num_threads(1)
output=args.output
output.mkdir(parents=True,exist_ok=True)
cfg=down_entry_env_cfg(args.entry_bank,num_envs=8,motion_file=args.motion_file)
cfg.seed=42
env=ManagerBasedRlEnv(cfg,device=args.device)
try:
  agent=down_ppo_runner_cfg()
  agent.num_steps_per_env=8
  agent.algorithm.num_learning_epochs=1
  agent.algorithm.num_mini_batches=2
  runner=DownEntryOnPolicyRunner(RslRlVecEnvWrapper(env,clip_actions=10.),asdict(agent),str(output),device=args.device)
  runner.warm_start(args.checkpoint)
  assert runner.current_learning_iteration==0 and runner.term.course.level==0
  assert not runner.alg.optimizer.state
  assert env.common_step_counter==0
  before={k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
  runner.learn(2)
  assert any(not torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in before.items())
  assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
  runner.term.course.level=2
  runner.term.course.streak=1
  runner.alg.learning_rate=1e-5
  for group in runner.alg.optimizer.param_groups:group['lr']=1e-5
  runner.save(str(output/'resume.pt'))
  expected=runner.term.course.state_dict()
  runner.term.course.level=0
  runner.term.course.streak=0
  runner.resume(output/'resume.pt')
  assert runner.term.course.state_dict()==expected
  assert runner.current_learning_iteration==2
  assert runner.alg.learning_rate==1e-5
  assert runner.alg.optimizer.state
  objective=env.cfg.rewards['standing_hold']
  objective.weight=.2
  try:runner.resume(output/'resume.pt')
  except ValueError:pass
  else:raise AssertionError('Changed objective was not rejected')
  objective.weight=.1
  class Zero(torch.nn.Module):
    def forward(self,obs,stochastic_output=False):return torch.zeros(len(obs),12,device=args.device)
  report=evaluate_entries(Zero(),args.entry_bank,level=3,device=args.device,attempts=32,motion_file=args.motion_file)
  assert report['distinct_trajectories']==32 and report['attempts']==32
  (output/'zero_entries.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  result=dict(warm_start=True,actor_updated=True,fresh_optimizer=True,strict_resume=True,
    adaptive_lr_restored=True,course_restored=True,changed_objective_rejected=True,
    zero_entry_attempts=32,distinct_trajectories=32,zero_successes=report['successes'])
  (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
  print('DOWN_ENTRY_SMOKE_PASSED',json.dumps(result),flush=True)
finally:env.close()
