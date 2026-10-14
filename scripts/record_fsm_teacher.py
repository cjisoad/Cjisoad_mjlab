#!/usr/bin/env python3
"""Record the full-chain FSM teacher with balanced recurrent training anchors."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import keyboard_transition_fsm as fsm
from src.tasks.fsm_student.observations import observation_contract, student_observations
from src.tasks.fsm_student.recording import (
    CATEGORIES, BalancedWindowIndex, atomic_npz, classify_frames, exclude_failure_tails, save_shard,
)


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def parse_args(argv=None):
    teacher = fsm.parse_args([])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=positive_int, default=16)
    parser.add_argument('--windows-per-category', type=positive_int, default=2000,
        help='Training windows per category; context precedes the supervised final frame')
    parser.add_argument('--validation-fraction', type=float, default=.2,
        help='Validation quota / training quota (rounded up)')
    parser.add_argument('--sequence-length', type=positive_int, default=32)
    parser.add_argument('--max-batches', type=positive_int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--failure-tail-seconds', type=float, default=.5)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--allow-partial', action='store_true')
    for name in ('quad_checkpoint_file', 'biped_checkpoint_file', 'rise_checkpoint_file',
                 'down_checkpoint_file', 'rise_motion_file', 'motion_file', 'entry_bank'):
        parser.add_argument('--'+name.replace('_', '-'), type=Path, default=getattr(teacher, name))
    args = parser.parse_args(argv)
    if not math.isfinite(args.validation_fraction) or not 0 < args.validation_fraction <= 1:
        parser.error('--validation-fraction must be in (0, 1]')
    if not math.isfinite(args.failure_tail_seconds) or args.failure_tail_seconds < 0:
        parser.error('--failure-tail-seconds must be finite and nonnegative')
    # Preserve the production teacher thresholds and workspace contract.
    for name in ('initial_state', 'workspace', 'rise_trigger_height', 'down_trigger_height', 'trigger_seconds'):
        setattr(args, name, getattr(teacher, name))
    if args.output_dir is None:
        args.output_dir = ROOT/'outputs'/('fsm_teacher_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    return args


class ScriptedEpisode:
    """Batched operator commands; global reference transitions stay synchronized.

    Completion gates use actual owners in live worlds, with a minimum duration.
    A timeout ends the batch and is recorded explicitly in the manifest.
    """
    def __init__(self, name, *, num_envs, nominal, seed, rise_seconds, down_seconds, moving=False):
        self.name, self.num_envs = name, num_envs
        self.nominal = np.asarray(nominal, dtype=np.float32)
        self.moving = moving
        self.rng = np.random.default_rng(seed)
        self.twist = np.column_stack((self.rng.uniform(.08, .22, num_envs),
            self.rng.uniform(-.035, .035, num_envs), self.rng.uniform(-.15, .15, num_envs))).astype(np.float32)
        self.low = np.column_stack((self.rng.uniform(.08, .12, num_envs),
            self.rng.uniform(-.02, .02, num_envs), np.full(num_envs, .12))).astype(np.float32)
        if name == 'quad':
            self.stages = [('settle', .6), ('quad_walk', 3.), ('quad_stand', 1.5),
                ('lift', 1.8), ('tripod_walk', 3.), ('tripod_operation', 3.), ('return', 2.5)]
        elif name == 'rise':
            self.stages = [('settle', .6), ('lift', 1.9), ('high', 4.),
                ('rise', rise_seconds+1.), ('biped_settle', 1.), ('biped_walk', 3.),
                ('biped_operation', 3.), ('down', down_seconds+1.), ('return', 2.5)]
        elif name == 'biped':
            self.stages = [('biped_settle', 1.), ('biped_walk', 3.), ('biped_operation', 3.),
                ('down', down_seconds+1.), ('return', 2.5)]
        else:
            raise ValueError(f'Unknown scenario {name}')
        self.stage = 0
        self.elapsed = 0.
        self.finished = False
        self.timed_out = False

    @property
    def stage_name(self):
        return self.stages[self.stage][0]

    def command(self):
        values = np.zeros((self.num_envs, 6), dtype=np.float32)
        name = self.stage_name
        if name == 'quad_walk':
            values[:, :3] = self.twist
        if name in ('lift', 'tripod_walk', 'tripod_operation', 'high', 'rise'):
            values[:, 3:] = self.low
            if name == 'high':
                values[:, 5] = .32
            elif name == 'rise':
                values[:, 5] = .36
            elif name == 'tripod_operation':
                values[:, 3] += .025*np.sin(self.elapsed*1.2)
                values[:, 4] += .02*np.sin(self.elapsed*.8)
                values[:, 5] += .025*np.sin(self.elapsed)
            if name == 'tripod_walk' or (name == 'lift' and self.moving):
                values[:, :3] = self.twist
        if name in ('biped_settle', 'biped_walk', 'biped_operation', 'down'):
            values[:, 3:] = self.nominal
            if name == 'biped_walk':
                values[:, :3] = self.twist*.6
            elif name == 'biped_operation':
                # Small feasible excursions; remain clear of the down trigger.
                values[:, 3] += .025*np.sin(self.elapsed)
                values[:, 4] += .02*np.sin(self.elapsed*.8)
                values[:, 5] += .025*np.sin(self.elapsed*.7)
            elif name == 'down':
                values[:, 5] = .24
        if name == 'return' and self.moving and self.name == 'quad':
            values[:, :3] = self.twist
        return values

    def advance(self, dt, owners, phases, alive):
        self.elapsed += dt
        name, minimum = self.stages[self.stage]
        selected = np.asarray(owners)[alive]
        ready = bool(len(selected))
        if name == 'rise':
            ready &= bool(np.all(selected == fsm.BIPED))
        elif name == 'down':
            ready &= bool(np.all(selected == fsm.QUADRUPED))
        elif name == 'return':
            ready &= bool(np.all(np.asarray(phases)[alive] == 0))
        if not np.any(alive):
            self.finished = True
            return
        if self.elapsed >= minimum and ready:
            if self.stage == len(self.stages)-1:
                self.finished = True
            else:
                self.stage += 1
                self.elapsed = 0.
        elif self.elapsed > minimum+5.:
            self.timed_out = self.finished = True


class AppliedActionCapture:
    """Capture actual per-physics-step output before env.step can reset it."""
    def __init__(self, action):
        self.action = action
        self.original = action._write_target
        self.actions = []
        self.targets = []

    def __enter__(self):
        def write(selected):
            self.original(selected)
            self.actions.append(self.action.raw_action.detach().clone())
            self.targets.append(self.action._entity.data.joint_pos_target[:, self.action.joint_ids].detach().clone())
        self.action._write_target = write
        return self

    def begin(self):
        self.actions.clear()
        self.targets.clear()

    def finish(self, decimation):
        if len(self.actions) != decimation:
            raise RuntimeError(f'Expected {decimation} actuator writes, got {len(self.actions)}')
        return self.actions[-1], self.targets[-1], torch.stack(self.actions, dim=1)

    def __exit__(self, *exc):
        self.action._write_target = self.original


def choose_scenario(counts, quotas):
    deficits = np.maximum(0, quotas[:, None]-counts)/quotas[:, None]
    category = int(np.argmax(deficits.max(axis=0)))
    if category in (4, 5, 9):
        return 'biped', category
    if category == 8:
        return 'rise', category
    return 'quad', category


def atomic_json(path, data):
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    # Never overwrite an earlier recording or its index.
    output.mkdir(parents=True, exist_ok=False)
    validation = math.ceil(args.windows_per_category*args.validation_fraction)
    index = BalancedWindowIndex(train_per_category=args.windows_per_category,
        validation_per_category=validation, sequence_length=args.sequence_length, seed=args.seed)
    manifest = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        categories=list(CATEGORIES), sequence_length=args.sequence_length,
        quotas=index.quotas.tolist(), observation_contract=observation_contract(), shards=[],
        balance='equal last-frame anchors per category within each split; preceding frames are context',
        split='trajectory-level; all frames of each trajectory belong to one split',
        label_timing='pre-step observation and actual final substep action captured before automatic reset',
        complete=False, interrupted=False)
    env = None
    try:
        env, wrapped, action, term, rise, down, paths = fsm.load_full_chain(args,
            configure_env=lambda cfg: setattr(cfg, 'seed', args.seed))
        manifest['sources'] = {key: {'path': str(path), 'sha256': sha256(path)} for key, path in paths.items()}
        manifest['code_sources'] = {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT/'src/tasks/fsm_student/recording.py',
            ROOT/'src/tasks/fsm_student/observations.py', ROOT/'scripts/keyboard_transition_fsm.py',
            ROOT/'src/tasks/transition_t/keyboard.py')}
        manifest['control_dt'] = float(env.step_dt)
        manifest['physics_dt'] = float(env.step_dt/env.cfg.decimation)
        manifest['decimation'] = int(env.cfg.decimation)
        manifest['action_contract'] = dict(dimension=12, joint_names=observation_contract()['joint_names'],
            scale=action.cfg.scale, default_angles=action.default_angles[0].cpu().tolist(),
            representation='common_joint_position_offset', motor_offsets='saved per frame',
            bc_label='applied_teacher_target_for_this_control_step_in_common_action_coordinates')
        policy = fsm.FullChainPolicy(term, rise, down)
        nominal = term.nominal_biped_offset.detach().cpu().numpy()
        rise_seconds = (term.rise_reference.num_frames-1)/term.rise_reference.fps
        down_seconds = (term.down_reference.num_frames-1)/term.down_reference.fps
        ids = torch.arange(args.num_envs, device=env.device)
        tail = math.ceil(args.failure_tail_seconds/env.step_dt)

        def checkpoint():
            manifest['counts'] = index.counts.tolist()
            manifest['candidates_seen'] = index.seen.tolist()
            manifest['complete'] = index.complete
            atomic_npz(output/'windows.npz', index.arrays())
            atomic_json(output/'manifest.json', manifest)

        checkpoint()
        with torch.inference_mode(), AppliedActionCapture(action) as capture:
            for batch in range(args.max_batches):
                if index.complete:
                    break
                name, requested_category = choose_scenario(index.counts, index.quotas)
                term.cfg.initial_state = 'biped' if name == 'biped' else 'quadruped'
                term.motion_term.initial_state = term.cfg.initial_state
                wrapped.reset()
                policy.reset()
                scenario = ScriptedEpisode(name, num_envs=args.num_envs, nominal=nominal,
                    seed=args.seed+batch, rise_seconds=rise_seconds, down_seconds=down_seconds,
                    moving=(batch % 2 == 1))
                # At least one whole trajectory in each split when N>=2.
                split = np.zeros(args.num_envs, dtype=np.uint8)
                if args.num_envs == 1:
                    split[0] = int(np.argmax(np.maximum(0, index.quotas-index.counts[:, requested_category]) / index.quotas))
                else:
                    validation_worlds = max(1, min(args.num_envs-1,
                        round(args.num_envs*args.validation_fraction/(1+args.validation_fraction))))
                    split[np.random.default_rng(args.seed+batch).choice(args.num_envs, validation_worlds, replace=False)] = 1
                alive = np.ones(args.num_envs, dtype=bool)
                rows = []
                print(f'[record] batch={batch} scenario={name} target={CATEGORIES[requested_category]}', flush=True)
                try:
                    while not scenario.finished:
                        command = scenario.command()
                        # Dead worlds may reset but cannot trigger competing global references.
                        live_ids = ids[torch.as_tensor(alive, device=env.device)]
                        term.submit(live_ids, command[alive], np.any(command[alive, 3:] != 0, axis=1))
                        term.keyboard_input_enabled[~torch.as_tensor(alive, device=env.device)] = False
                        teacher_obs = env.observation_manager.compute()
                        owner = term.owner.cpu().numpy().copy()
                        phase = term.phase.cpu().numpy().copy()
                        blend = term.blend.cpu().numpy().copy()
                        observation = student_observations(env, command).cpu().numpy().copy()
                        offsets = action.motor_offsets.cpu().numpy().copy()
                        velocity = env.scene['robot'].data.root_link_lin_vel_b.cpu().numpy().copy()
                        capture.begin()
                        _, _, done, _ = wrapped.step(policy(teacher_obs))
                        applied, targets, substeps = capture.finish(env.cfg.decimation)
                        terminal = done.cpu().numpy().astype(bool)
                        rows.append(dict(observations=observation, actions=applied.cpu().numpy().copy(),
                            joint_targets=targets.cpu().numpy().copy(), substep_actions=substeps.cpu().numpy().copy(),
                            commands=command, category=classify_frames(owner, phase, blend, command),
                            owner=owner, phase=phase, blend=blend, valid=alive.copy(), done=terminal,
                            episode_starts=(rows[-1]['done'].copy() if rows else
                                np.ones(args.num_envs, dtype=bool)),
                            motor_offsets=offsets, actual_velocity=velocity))
                        alive &= ~terminal
                        scenario.advance(env.step_dt, term.owner.cpu().numpy(), term.phase.cpu().numpy(), alive)
                        if len(rows) % 250 == 0:
                            print(f'[record] t={len(rows)*env.step_dt:.1f}s stage={scenario.stage_name} alive={int(alive.sum())}/{args.num_envs}', flush=True)
                except KeyboardInterrupt:
                    manifest['interrupted'] = True
                if rows:
                    data = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
                    data['valid'] = exclude_failure_tails(data['valid'], data['done'], tail)
                    data['trajectory_id'] = np.arange(args.num_envs, dtype=np.int64)+batch*args.num_envs
                    data['split'] = split
                    filename = f'shard_{batch:05d}.npz'
                    save_shard(output/filename, data)
                    index.add_shard(batch, data)
                    manifest['shards'].append(dict(id=batch, file=filename, scenario=name,
                        moving_support_transition=scenario.moving, frames=len(rows),
                        failed_worlds=int((~alive).sum()), scenario_completed=scenario.finished and not scenario.timed_out,
                        timed_out=scenario.timed_out, sha256=sha256(output/filename),
                        valid_category_frames=np.bincount(data['category'][data['valid']], minlength=len(CATEGORIES)).tolist()))
                checkpoint()
                print(f'[record] train={index.counts[0].tolist()} validation={index.counts[1].tolist()}', flush=True)
                if manifest['interrupted']:
                    break
    finally:
        if env is not None:
            env.close()
    print(f'[record] output={output} complete={index.complete}', flush=True)
    if not index.complete:
        for split_name, counts, quota in zip(('train', 'validation'), index.counts, index.quotas):
            print(f'[record] {split_name} missing: '+', '.join(
                f'{name}={int(quota-count)}' for name, count in zip(CATEGORIES, counts) if count < quota), flush=True)
    return 0 if index.complete or args.allow_partial else 2


if __name__ == '__main__':
    raise SystemExit(main())
