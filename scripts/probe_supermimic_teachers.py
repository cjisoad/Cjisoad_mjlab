"""Measure frozen teacher labels across the keyboard velocity envelope.

Default: 10 seconds per command, three independent worlds per case. All 24
cases run concurrently. Only each world's first episode is measured, so an
automatic reset after a fall cannot silently improve its tracking statistics.
This probe performs no training and has no performance acceptance gate.
"""
import argparse
import json
import os
from pathlib import Path


def command_grid():
    return [{'mode': mode, 'velocity': [vx, 0., yaw]}
            for mode in ('quad', 'biped')
            for vx in (-.2, 0., .15, .6)
            for yaw in (0., -1., 1.)]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=500, help='Control steps per command (500 = 10s)')
    parser.add_argument('--num-envs', type=int, default=3, help='Independent worlds per command')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', type=Path, default=Path('outputs/supermimic/teacher_quality.json'))
    return parser


def run_probe(args):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import quat_apply
    from src.tasks.legmanip_distil.intent import IntentCommandCfg
    from src.tasks.legmanip_distil.supermimic.env import supermimic_env_cfg
    from src.tasks.legmanip_distil.teacher import FSMTeacher
    from src.tasks.teacher_common.anchor_frame import anchor_basis, to_anchor
    torch.set_num_threads(1)
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    cases = command_grid()
    count = len(cases) * args.num_envs
    cfg = supermimic_env_cfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = count
    cfg.commands['intent'] = IntentCommandCfg(automatic=False, initial_state='quadruped', entry_split='validation')
    cfg.episode_length_s = max(60., args.steps * .02 + 1.)
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        env.reset()
        intent = env.command_manager.get_term('intent')
        case_ids = torch.arange(len(cases), device=device).repeat_interleave(args.num_envs)
        biped = torch.tensor([case['mode'] == 'biped' for case in cases], device=device)[case_ids]
        # Independent physical entry families use the existing coherent bank.
        intent.cfg.initial_state = 'biped'
        env.reset(env_ids=biped.nonzero().flatten())
        velocity = torch.tensor([case['velocity'] for case in cases], device=device)[case_ids]
        commands = torch.zeros(count, 6, device=device)
        commands[:, :3] = velocity
        commands[biped, 3:] = intent.nominal
        intent.set_command(torch.arange(count, device=device), commands)
        teacher = FSMTeacher(env)
        teacher.term._update_command()
        active = torch.ones(count, dtype=torch.bool, device=device)
        samples = torch.zeros(count, device=device)
        survival = torch.zeros_like(samples)
        falls = torch.zeros_like(samples)
        timeouts = torch.zeros_like(samples)
        invalid = torch.zeros_like(samples)
        actual_sum = torch.zeros(count, 3, device=device)
        signed_sum = torch.zeros_like(actual_sum)
        absolute_sum = torch.zeros_like(actual_sum)
        square_sum = torch.zeros_like(actual_sum)
        coordinator_sum = torch.zeros_like(actual_sum)
        with torch.inference_mode():
            for _ in range(args.steps):
                before = active.clone()
                coordinator_sum += teacher.term.command[:, :3] * before[:, None]
                labels = teacher()
                teacher.set_control_mask(torch.ones_like(active))
                _, _, terminated, truncated, _ = env.step(labels)
                state = env.supermimic_terminal['next_state']
                finite = torch.isfinite(state).all(-1) & (state[:, 3:7].norm(dim=-1) > 1.e-6)
                q = torch.nan_to_num(state[:, 3:7])
                norm = q.norm(dim=-1, keepdim=True)
                identity = torch.zeros_like(q); identity[:, 0] = 1.
                q = torch.where(norm > 1.e-6, q / norm.clamp_min(1.e-6), identity)
                gravity = state.new_tensor([0., 0., -1.]).expand(count, 3)
                basis = anchor_basis(q, gravity)
                linear = to_anchor(basis, torch.nan_to_num(state[:, 7:10]))
                angular = to_anchor(basis, quat_apply(q, torch.nan_to_num(state[:, 10:13])))
                measured = torch.stack((linear[:, 0], linear[:, 1], angular[:, 2]), -1)
                error = measured - velocity
                valid = before & finite
                weight = valid[:, None]
                samples += valid
                survival += before
                invalid += before & ~finite
                actual_sum += measured * weight
                signed_sum += error * weight
                absolute_sum += error.abs() * weight
                square_sum += error.square() * weight
                falls += before & terminated
                timeouts += before & truncated
                active &= ~(terminated | truncated | ~finite)
        rows = []
        for index, case in enumerate(cases):
            ids = case_ids == index
            denominator = samples[ids].sum().clamp_min(1.)
            frames = survival[ids].sum().clamp_min(1.)
            rows.append(dict(case, worlds=args.num_envs, valid_frames=int(samples[ids].sum()),
                falls=int(falls[ids].sum()), timeouts=int(timeouts[ids].sum()),
                invalid_frames=int(invalid[ids].sum()), survivors=int(active[ids].sum()),
                mean_survival_seconds=float(survival[ids].mean() * env.step_dt),
                mean_actual_velocity=(actual_sum[ids].sum(0)/denominator).cpu().tolist(),
                mean_signed_error=(signed_sum[ids].sum(0)/denominator).cpu().tolist(),
                mean_absolute_error=(absolute_sum[ids].sum(0)/denominator).cpu().tolist(),
                root_mean_square_error=(square_sum[ids].sum(0)/denominator).sqrt().cpu().tolist(),
                mean_coordinator_velocity=(coordinator_sum[ids].sum(0)/frames).cpu().tolist()))
        return {'purpose': 'frozen_teacher_label_quality', 'training': False,
                'performance_gate': None, 'device': str(device), 'seed': args.seed,
                'steps': args.steps, 'control_dt': env.step_dt,
                'seconds_per_case': args.steps * env.step_dt,
                'worlds_per_case': args.num_envs, 'total_worlds': count,
                'velocity_axes': ['anchor_vx', 'anchor_vy', 'anchor_wz'],
                'measurement': 'pre-reset physical state; first episode only; error against raw requested intent',
                'note': 'Includes startup transients; short smoke results do not establish mature command tracking.',
                'teacher': teacher.metadata(), 'cases': rows}
    finally:
        env.close()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.num_envs <= 0:
        parser.error('--steps and --num-envs must be positive')
    result = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Teacher quality probe written to', args.output)
    print(json.dumps({'cases': len(result['cases']), 'seconds_per_case': result['seconds_per_case'],
                      'falls': sum(case['falls'] for case in result['cases']),
                      'invalid_frames': sum(case['invalid_frames'] for case in result['cases'])}))


if __name__ == '__main__':
    main()
