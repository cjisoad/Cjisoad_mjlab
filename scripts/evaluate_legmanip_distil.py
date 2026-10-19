"""Evaluate student behavior or the teacher's substep/final-target timing gap."""
import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-file')
    parser.add_argument('--policy', choices=('student', 'teacher', 'teacher-hold', 'compare-teacher-timing'), default='student')
    parser.add_argument('--scenario', choices=('all', 'quad', 'biped', 'cycle'), default='all')
    parser.add_argument('--episodes', type=int, default=16, help='Episodes per scenario')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--split', choices=('validation', 'final'), default='validation')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', type=Path, default=Path('outputs/legmanip_distil/evaluation.json'))
    args = parser.parse_args(argv)
    if args.policy == 'student' and not args.checkpoint_file:
        parser.error('--checkpoint-file is required for student evaluation')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import torch
    from src.tasks.legmanip_distil.evaluation import evaluate_policy
    torch.set_num_threads(1)
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    kwargs = dict(checkpoint=args.checkpoint_file, scenario=args.scenario, episodes=args.episodes,
                  num_envs=args.num_envs, max_steps=args.max_steps, device=device,
                  split=args.split, seed=args.seed)
    if args.policy == 'compare-teacher-timing':
        result = {policy: evaluate_policy(policy=policy, **kwargs) for policy in ('teacher', 'teacher-hold')}
    else:
        result = evaluate_policy(policy=args.policy, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Evaluation written to', args.output)
    print(json.dumps({key: value['summary'] for key, value in result.items()}
                     if args.policy == 'compare-teacher-timing' else result['summary'], indent=2))


if __name__ == '__main__':
    main()
