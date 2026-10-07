"""Bounded real-physics bridge collection/PPO and strict checkpoint checks."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/bridge-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/bridge-cache')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device',default='cpu')
  parser.add_argument('--num-envs',type=int,default=32)
  parser.add_argument('--cycles',type=int,default=2)
  parser.add_argument('--ppo',action='store_true')
  parser.add_argument('--output',type=Path,default=Path('outputs/bridge_smoke'))
  args=parser.parse_args()
  if not 2<=args.num_envs<=4096 or not 1<=args.cycles<=20:
    parser.error('Require 2..4096 environments and 1..20 complete cycles')
  if args.device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']=''
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from src.tasks.pedipulation_bridge_t.env_cfg import bridge_env_cfg
  from src.tasks.pedipulation_bridge_t.rl import BridgeOnPolicyRunner,bridge_ppo_runner_cfg
  from src.tasks.pedipulation_bridge_t.guidance import ACTOR_DIM,CRITIC_DIM
  torch.set_num_threads(1)
  args.output.mkdir(parents=True,exist_ok=True)
  cfg=bridge_env_cfg(); cfg.scene.num_envs=args.num_envs
  raw=ManagerBasedRlEnv(cfg,device=args.device)
  try:
    env=RslRlVecEnvWrapper(raw,clip_actions=10.)
    agent=bridge_ppo_runner_cfg()
    runner=BridgeOnPolicyRunner(env,asdict(agent),str(args.output) if args.ppo else None,args.device)
    obs=env.get_observations()
    assert obs['actor'].shape==(args.num_envs,ACTOR_DIM)
    assert obs['critic'].shape==(args.num_envs,CRITIC_DIM)
    frozen={name:{key:value.detach().clone() for key,value in model.state_dict().items()}
      for name,model in runner.experts.actors.items()}
    actor_before=[p.detach().clone() for p in runner.alg.actor.parameters()]
    if args.ppo:
      runner.learn(args.cycles)
      record=dict(runner.last_metrics)
      record.update(actor_updated=any(not torch.equal(a,b) for a,b in zip(actor_before,runner.alg.actor.parameters())),
        updates=runner.updates)
    else:
      records=[]
      for _ in range(args.cycles):
        _, metrics=runner.collect_attempts(); records.append(metrics)
      record={'cycles':records}
    for name,model in runner.experts.actors.items():
      assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
      for key,value in model.state_dict().items(): torch.testing.assert_close(value,frozen[name][key],rtol=0,atol=0)
    record.update(frozen_experts_unchanged=True,actor_dim=ACTOR_DIM,critic_dim=CRITIC_DIM,device=args.device,
      terminal_credit='measured_validation_outcome_no_tail_discount')
    if args.ppo:
      checkpoint=args.output/'roundtrip.pt'; runner.save(str(checkpoint))
      sample=env.get_observations()
      expected=runner.alg.actor(sample).detach().clone()
      with torch.no_grad(): next(runner.alg.actor.parameters()).add_(.1)
      runner.load(str(checkpoint))
      torch.testing.assert_close(runner.alg.actor(sample),expected,rtol=0,atol=0)
      env.clip_actions=5.
      try: runner.load(str(checkpoint))
      except ValueError: pass
      else: raise AssertionError('Changed action clipping must reject bridge checkpoint')
      env.clip_actions=10.
      record.update(checkpoint_roundtrip=True,incompatible_contract_rejected=True,
        checkpoint=str(checkpoint.resolve()))
    (args.output/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print('BRIDGE_SMOKE_RESULT '+json.dumps(record),flush=True)
    if args.ppo:
      assert record['actor_updated'] and runner.updates>0, 'No effective bridge PPO update; inspect source yield'
  finally: raw.close()


if __name__=='__main__': main()
