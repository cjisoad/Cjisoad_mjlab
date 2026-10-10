"""Train the approved reverse transition from scratch in an explicit directory."""
import argparse
from pathlib import Path
import os
import json
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--motion-file',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--device',default='cuda:0')
  p.add_argument('--num-envs',type=int,default=4096)
  p.add_argument('--iterations',type=int,default=30001)
  p.add_argument('--seed',type=int,default=42)
  args=p.parse_args()
  if args.num_envs<1 or args.iterations<1:p.error('Positive worlds and updates required')
  if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a fresh nominal output directory')
  os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MPLCONFIGDIR','/tmp/transition_down_mpl')
  os.environ.setdefault('XDG_CACHE_HOME','/tmp/transition_down_cache');sys.path.insert(0,str(ROOT))
  import torch
  import src.tasks
  from scripts.train import TrainConfig,run_train
  from src.tasks.transition_down_t.data import verify_approved_assets,file_sha256
  verify_approved_assets(motion_file=args.motion_file)
  torch.set_num_threads(1)
  if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
  elif args.device.startswith('cuda:'):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for GPU training')
    os.environ['CUDA_VISIBLE_DEVICES']=args.device.split(':')[1]
  else:raise ValueError('Use cpu or cuda:<index>')
  cfg=TrainConfig.from_task('Transition-Down-Go2-v0')
  cfg.env.scene.num_envs=args.num_envs;cfg.agent.seed=args.seed;cfg.agent.max_iterations=args.iterations
  cfg.agent.save_interval=100
  from dataclasses import replace
  cfg=replace(cfg,motion_file=str(args.motion_file.resolve()))
  args.output.mkdir(parents=True,exist_ok=True)
  (args.output/'run.json').write_text(json.dumps(dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
    initialization='from_scratch',motion_sha256=file_sha256(args.motion_file),torch_version=torch.__version__,
    gpu=torch.cuda.get_device_name(0) if args.device!='cpu' else None),indent=2)+'\n')
  run_train('Transition-Down-Go2-v0',cfg,args.output)
  (args.output/'finished.json').write_text(json.dumps(dict(updates_completed=args.iterations,
    final_checkpoint=str((args.output/f'model_{args.iterations-1}.pt').resolve())))+'\n')

if __name__=='__main__':main()
