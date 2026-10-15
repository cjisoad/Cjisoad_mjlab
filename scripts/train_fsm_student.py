#!/usr/bin/env python3
"""Two-stage BC -> DAgger training for the 48D FSM GRU student."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

from src.tasks.fsm_student import GRUStudent, GRUStudentConfig, student_observations
from src.tasks.fsm_student.distillation import (
    ArrayWindowDataset, WindowDataset, evaluate_epoch, save_metrics,
    save_student_checkpoint, load_student_checkpoint, train_epoch, warm_start_mask,
)
from src.tasks.fsm_student.recording import classify_frames


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('bc', 'dagger', 'all'))
    parser.add_argument('--dataset', type=Path, default=ROOT/'outputs/fsm_teacher_dataset')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/fsm_student/student.pt')
    parser.add_argument('--student-checkpoint', type=Path, default=None,
        help='Initial checkpoint for dagger; all uses the just-created BC checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=positive_int, default=30)
    parser.add_argument('--dagger-epochs', type=positive_int, default=5)
    parser.add_argument('--iterations', type=positive_int, default=3)
    parser.add_argument('--rollout-steps', type=positive_int, default=1500)
    parser.add_argument('--num-envs', type=positive_int, default=16)
    parser.add_argument('--batch-size', type=positive_int, default=256)
    parser.add_argument('--sequence-length', type=positive_int, default=32)
    parser.add_argument('--warm-start-seconds', type=float, default=1.,
        help='Teacher control after every physical reset while the student warms its GRU; 0 disables')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--dagger-lr', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.)
    parser.add_argument('--replay-repeat', type=positive_int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-windows', type=positive_int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.warm_start_seconds) or args.warm_start_seconds < 0:
        parser.error('--warm-start-seconds must be nonnegative and finite')
    for name in ('lr', 'dagger_lr', 'grad_clip'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive and finite')
    if args.stage == 'dagger' and args.student_checkpoint is None:
        parser.error('dagger requires --student-checkpoint')
    return args


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _manifest(dataset):
    path = Path(dataset)/'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def build_model(dataset, device):
    manifest = _manifest(dataset)
    contract = manifest.get('action_contract') or {}
    defaults = contract.get('default_angles')
    scale = contract.get('scale', .25)
    if defaults is None or len(defaults) != 12:
        raise ValueError('Dataset manifest does not contain the 12-joint action contract')
    model = GRUStudent(GRUStudentConfig(default_angles=tuple(float(x) for x in defaults),
        action_scale=float(scale))).to(device)
    return model, manifest


def _loader(dataset, batch_size, *, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        drop_last=False, generator=generator)


def _balanced_loader(dataset, batch_size, *, seed, shuffle=True):
    """Balance supervised frames, including context categories and short prefixes."""
    counts = torch.zeros(10, dtype=torch.long)
    for index in range(len(dataset)):
        item = dataset[index]
        counts += torch.bincount(item['categories'][item['mask']], minlength=10)
    present = counts > 0
    if not present.any():
        raise ValueError('Cannot build a category-balanced loader from empty data')
    weights = torch.zeros(10)
    weights[present] = counts.sum()/(present.sum()*counts[present].float())
    loader = _loader(dataset, batch_size, shuffle=shuffle, seed=seed)
    loader.frame_category_weights = weights
    loader.category_frame_counts = counts.tolist()
    return loader


def _train_bc(args, output):
    seed_everything(args.seed)
    model, manifest = build_model(args.dataset, args.device)
    train = WindowDataset(args.dataset, split=0, max_windows=args.max_windows, include_prefix=True)
    validation = WindowDataset(args.dataset, split=1,
        max_windows=None if args.max_windows is None else max(1, args.max_windows//5), include_prefix=True)
    if len(train) == 0 or len(validation) == 0:
        raise ValueError('BC dataset must contain both training and validation windows')
    actual_sequence_length = int(train[0]['observations'].shape[0])
    if args.sequence_length != actual_sequence_length:
        raise ValueError(
            f'BC sequence length {args.sequence_length} does not match '
            f'teacher dataset window length {actual_sequence_length}')
    train_loader = _balanced_loader(train, args.batch_size, seed=args.seed)
    validation_loader = _balanced_loader(validation, args.batch_size, shuffle=False, seed=args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer,
            device=args.device, grad_clip=args.grad_clip)
        validation_loss = evaluate_epoch(model, validation_loader, device=args.device)
        row = {'epoch': epoch, 'train_loss': train_loss, 'validation_loss': validation_loss}
        history.append(row)
        print(f'[bc] epoch={epoch}/{args.epochs} train={train_loss:.6f} validation={validation_loss:.6f}', flush=True)
    metrics = {'stage': 'bc', 'history': history, 'dataset': str(Path(args.dataset).resolve()),
        'categories': manifest.get('categories'), 'train_windows': len(train),
        'validation_windows': len(validation), 'sequence_length': args.sequence_length,
        'loss': 'masked_full_frame_smooth_l1', 'balance': 'inverse_supervised_frame_category_frequency',
        'supervised_category_frame_counts': train_loader.category_frame_counts,
        'include_episode_prefixes': True}
    save_student_checkpoint(output, model, optimizer=optimizer, stage='bc', epoch=args.epochs,
        metrics=metrics, seed=args.seed)
    save_metrics(Path(output).with_suffix('.metrics.json'), metrics)
    return model, manifest


def _select_observations(values, ids):
    if isinstance(values, dict):
        return {key: value.index_select(0, ids) for key, value in values.items()}
    if isinstance(values, (tuple, list)):
        return type(values)(_select_observations(value, ids) for value in values)
    return values.index_select(0, ids)


@torch.inference_mode()
def query_teacher_actions(env, term, rise, down, teacher_observations):
    """Query the frozen actor matching each full-chain owner on this state."""
    import keyboard_transition_fsm as fsm
    BIPED, DOWN, QUADRUPED, RISE = fsm.BIPED, fsm.DOWN, fsm.QUADRUPED, fsm.RISE
    action = env.action_manager.get_term('joint_pos')
    owners = term.owner
    labels = torch.zeros(env.num_envs, 12, device=owners.device, dtype=action.raw_action.dtype)
    endpoint = ((owners == QUADRUPED) | (owners == BIPED)).nonzero().flatten()
    quad = endpoint[owners[endpoint] == QUADRUPED]
    biped = endpoint[owners[endpoint] == BIPED]
    if len(quad):
        labels[quad] = action.source_teacher.actions(env, quad)
    if len(biped):
        labels[biped] = action.biped_teacher.actions(env, biped)
    rise_ids = (owners == RISE).nonzero().flatten()
    down_ids = (owners == DOWN).nonzero().flatten()
    if len(rise_ids):
        labels[rise_ids] = rise(_select_observations(teacher_observations, rise_ids))
    if len(down_ids):
        labels[down_ids] = down(_select_observations(teacher_observations, down_ids))
    if not torch.isfinite(labels).all():
        raise FloatingPointError('Teacher returned nonfinite DAgger labels')
    return labels.clamp(-100., 100.)


def _teacher_args(args, manifest):
    import keyboard_transition_fsm as fsm
    values = manifest.get('args') or {}
    argv = ['--device', args.device, '--num-envs', str(args.num_envs)]
    for flag, key in (('quad-checkpoint-file', 'quad_checkpoint_file'),
                      ('biped-checkpoint-file', 'biped_checkpoint_file'),
                      ('rise-checkpoint-file', 'rise_checkpoint_file'),
                      ('down-checkpoint-file', 'down_checkpoint_file'),
                      ('rise-motion-file', 'rise_motion_file'),
                      ('motion-file', 'motion_file'), ('entry-bank', 'entry_bank')):
        if values.get(key):
            argv.extend(['--'+flag, str(values[key])])
    return fsm.parse_args(argv)


def _atomic_npz(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.savez_compressed(handle, **data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


@torch.no_grad()
def collect_dagger_rollout(args, model, manifest, iteration, output_dir):
    """Run student actions and label the same pre-step states with frozen teacher."""
    import keyboard_transition_fsm as fsm
    from record_fsm_teacher import ScriptedEpisode
    teacher_args = _teacher_args(args, manifest)
    env = wrapped = None
    try:
        env, wrapped, action, term, rise, down, paths = fsm.load_full_chain(teacher_args)
        action.use_teacher_policy = False
        model.eval()
        hidden = model.initial_hidden(args.num_envs)
        scenario_name = ('quad', 'biped', 'rise')[(iteration-1) % 3]
        term.cfg.initial_state = 'biped' if scenario_name == 'biped' else 'quadruped'
        term.motion_term.initial_state = term.cfg.initial_state
        wrapped.reset()
        nominal = term.nominal_biped_offset.detach().cpu().numpy()
        rise_seconds = (term.rise_reference.num_frames-1)/term.rise_reference.fps
        down_seconds = (term.down_reference.num_frames-1)/term.down_reference.fps
        scenario = ScriptedEpisode(scenario_name, num_envs=args.num_envs, nominal=nominal,
            seed=args.seed+iteration, rise_seconds=rise_seconds, down_seconds=down_seconds,
            moving=(iteration % 2 == 0))
        ids = torch.arange(args.num_envs, device=env.device)
        observations=[]; labels=[]; student_outputs=[]; commands=[]; categories=[]; valid=[]; done=[]; starts=[]
        teacher_controls=[]; episode_steps=[]
        starts_next = np.ones(args.num_envs, dtype=bool)
        for step in range(args.rollout_steps):
            command = scenario.command()
            term.submit(ids, command, np.any(command[:, 3:] != 0, axis=1))
            teacher_obs = env.observation_manager.compute()
            owner = term.owner.detach().cpu().numpy().copy()
            phase = term.phase.detach().cpu().numpy().copy()
            blend = term.blend.detach().cpu().numpy().copy()
            student_obs = student_observations(env, command)
            teacher_label = query_teacher_actions(env, term, rise, down, teacher_obs)
            student_action, hidden = model(student_obs, hidden)
            if not torch.isfinite(student_action).all():
                raise FloatingPointError('Student returned nonfinite DAgger actions')
            teacher_control = warm_start_mask(env.episode_length_buf,
                step_dt=env.step_dt, seconds=args.warm_start_seconds)
            teacher_controls.append(teacher_control.cpu().numpy().copy())
            episode_steps.append(env.episode_length_buf.cpu().numpy().copy())
            # Endpoint teachers must be injected after wrapper clipping, exactly
            # as in teacher playback; transition teachers use the routed actor.
            action.teacher_policy_mask = teacher_control
            selected = torch.where(teacher_control[:, None], teacher_label, student_action)
            _, _, finished, _ = wrapped.step(selected)
            finished = finished.detach().cpu().numpy().astype(bool)
            observations.append(student_obs.cpu().numpy().copy())
            labels.append(teacher_label.cpu().numpy().copy())
            student_outputs.append(student_action.detach().cpu().numpy().copy())
            commands.append(command.copy())
            categories.append(classify_frames(owner, phase, blend, command))
            valid.append(~finished)
            done.append(finished)
            starts.append(starts_next.copy())
            starts_next = finished
            if finished.any():
                hidden[:, torch.as_tensor(finished, device=env.device)] = 0.
            scenario.advance(env.step_dt, term.owner.detach().cpu().numpy(),
                term.phase.detach().cpu().numpy(), np.ones(args.num_envs, dtype=bool))
        data = {
            'observations': np.stack(observations), 'actions': np.stack(labels),
            'student_actions': np.stack(student_outputs),
            'commands': np.stack(commands), 'category': np.stack(categories),
            'valid': np.stack(valid), 'done': np.stack(done),
            'episode_starts': np.stack(starts),
            'teacher_control': np.stack(teacher_controls),
            'episode_steps': np.stack(episode_steps),
            'trajectory_id': np.arange(args.num_envs, dtype=np.int64)+iteration*100000,
            'split': np.zeros(args.num_envs, dtype=np.uint8),
        }
        for key, value in data.items():
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise FloatingPointError(f'DAgger replay {key} contains nonfinite values')
        path = Path(output_dir)/f'dagger_{iteration:02d}.npz'
        _atomic_npz(path, data)
        action_gap = np.abs(data['student_actions']-data['actions'])
        return data, path, dict(scenario=scenario_name, steps=args.rollout_steps,
            finished_worlds=int(np.stack(done).sum()), replay_windows=len(ArrayWindowDataset(data,
                sequence_length=args.sequence_length)),
            warm_start_seconds=args.warm_start_seconds,
            teacher_frames=int(data['teacher_control'].sum()),
            student_frames=int((~data['teacher_control']).sum()),
            teacher_terminations=int((data['done'] & data['teacher_control']).sum()),
            student_terminations=int((data['done'] & ~data['teacher_control']).sum()),
            valid_category_frames=np.bincount(data['category'][data['valid']], minlength=10).tolist(),
            mean_student_teacher_abs=float(action_gap.mean()),
            max_student_teacher_abs=float(action_gap.max()),
            label_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    finally:
        if env is not None: env.close()


def _check_output(path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Output already exists; use --overwrite: {path}')
    if path.exists() and overwrite:
        path.unlink()
    metrics = path.with_suffix('.metrics.json')
    if metrics.exists() and overwrite:
        metrics.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _run_dagger(args, output, *, initial_checkpoint):
    seed_everything(args.seed)
    model, manifest = build_model(args.dataset, args.device)
    model, metadata = load_student_checkpoint(initial_checkpoint, device=args.device)
    bc = WindowDataset(args.dataset, split=0, max_windows=args.max_windows, include_prefix=True)
    if len(bc) == 0:
        raise ValueError('BC dataset is empty')
    bc_sequence_length = int(bc[0]['observations'].shape[0])
    if args.sequence_length != bc_sequence_length:
        raise ValueError(
            f'DAgger sequence length {args.sequence_length} does not match '
            f'teacher dataset window length {bc_sequence_length}')
    replay_dir = Path(output).with_suffix('.replay')
    replay_dir.mkdir(parents=True, exist_ok=True)
    iteration_metrics=[]
    replay_sets=[]
    for iteration in range(1, args.iterations+1):
        data, replay_path, rollout = collect_dagger_rollout(args, model, manifest, iteration, replay_dir)
        replay = ArrayWindowDataset(data, sequence_length=args.sequence_length)
        if not rollout['student_frames']:
            raise ValueError('DAgger rollout contains no student-controlled frames; '
                'increase --rollout-steps or reduce --warm-start-seconds')
        if not len(replay):
            raise ValueError('DAgger rollout contains no valid replay frames')
        replay_sets.append(replay)
        combined = ConcatDataset([bc, *(
            [dataset for dataset in replay_sets for _ in range(args.replay_repeat)])])
        loader = _balanced_loader(combined, args.batch_size, seed=args.seed+iteration)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.dagger_lr)
        losses=[]
        for epoch in range(1, args.dagger_epochs+1):
            loss=train_epoch(model, loader, optimizer, device=args.device, grad_clip=args.grad_clip)
            losses.append(loss)
            print(f'[dagger] iter={iteration}/{args.iterations} epoch={epoch}/{args.dagger_epochs} loss={loss:.6f} replay={len(replay)}', flush=True)
        iteration_metrics.append(dict(iteration=iteration, replay=str(replay_path), rollout=rollout,
            losses=losses, replay_windows=len(replay),
            replay_windows_total=sum(len(dataset) for dataset in replay_sets),
            supervised_category_frame_counts=loader.category_frame_counts))
        iteration_path = Path(output).with_suffix(f'.iter{iteration:02d}.pt')
        if iteration_path.exists() and args.overwrite:
            iteration_path.unlink()
        save_student_checkpoint(iteration_path, model,
            optimizer=optimizer, stage='dagger', epoch=iteration*args.dagger_epochs,
            metrics=iteration_metrics[-1], seed=args.seed)
    metrics={'stage':'dagger','initial_checkpoint':str(Path(initial_checkpoint).resolve()),
        'dataset':str(Path(args.dataset).resolve()),'iterations':iteration_metrics,
        'sequence_length':args.sequence_length,'replay_repeat':args.replay_repeat,
        'warm_start_seconds':args.warm_start_seconds, 'loss':'masked_full_frame_smooth_l1',
        'balance':'inverse_supervised_frame_category_frequency'}
    save_student_checkpoint(output, model, stage='dagger', epoch=args.iterations*args.dagger_epochs,
        metrics=metrics, seed=args.seed)
    save_metrics(Path(output).with_suffix('.metrics.json'), metrics)


def main(argv=None):
    args = parse_args(argv)
    args.dataset = args.dataset.expanduser().resolve(); args.output = args.output.expanduser().resolve()
    if not args.dataset.is_dir():
        raise FileNotFoundError(args.dataset)
    if args.stage == 'bc':
        output = _check_output(args.output, args.overwrite)
        _train_bc(args, output)
        return 0
    if args.stage == 'dagger':
        output = _check_output(args.output, args.overwrite)
        _run_dagger(args, output, initial_checkpoint=args.student_checkpoint.expanduser().resolve())
        return 0
    bc_output = args.output.with_suffix('.bc.pt')
    final_output = _check_output(args.output, args.overwrite)
    if bc_output.exists() and not args.overwrite:
        raise FileExistsError(f'BC output already exists; use --overwrite: {bc_output}')
    if args.overwrite:
        for path in (bc_output, bc_output.with_suffix('.metrics.json')):
            if path.exists(): path.unlink()
    _train_bc(args, bc_output)
    _run_dagger(args, final_output, initial_checkpoint=bc_output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
