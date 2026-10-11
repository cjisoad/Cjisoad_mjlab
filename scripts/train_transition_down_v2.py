"""Train reviewed V2 landing until evaluation passes or the total budget expires."""
import argparse,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MPLCONFIGDIR','/tmp/transition_down_v2_mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/transition_down_v2_cache')
def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--max-updates',type=int,default=40000,help='Total resource budget, never a stage length')
    p.add_argument('--eval-interval',type=int,default=250)
    p.add_argument('--gate-config',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--smoke',action='store_true')
    p.add_argument('--quad-teacher',type=Path,default=ROOT/'mark/quad_teacher.pt')
    p.add_argument('--biped-teacher',type=Path,default=ROOT/'mark/biped_teacher.pt')
    a=p.parse_args(argv)
    if min(a.num_envs,a.max_updates,a.eval_interval)<1:p.error('Positive worlds, budget and evaluation interval required')
    if a.smoke and a.num_envs<2:p.error('Smoke needs at least two worlds')
    return a
if __name__=='__main__':
    from src.tasks.transition_down_v2.training import train
    result=train(parse_args())
    raise SystemExit(0 if result['status'] in ('accepted','smoke_completed') else 2)
