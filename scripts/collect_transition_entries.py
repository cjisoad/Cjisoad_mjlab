"""Collect physical source-teacher handoff states in nominal transition physics.

Example: python scripts/collect_transition_entries.py --device cuda:0 --output entries.npz
Only the frozen locomotion teacher acts. No reference poses are synthesized.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = ROOT / 'outputs/loco_return_20261009/checkpoints/model_14999.pt'
DEFAULT_TRANSITION = ROOT / 'logs/rsl_rl/transition_t/2026-10-09_01-13-48_transition_t_direct_4090_20261009/model_30000.pt'
SPEED_EDGES = (0., .1, .3, .5, .8, 1.2, float('inf'))
ERROR_EDGES = (0., .05, .1, .2, .35, float('inf'))


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--num-envs', type=int, default=256)
  parser.add_argument('--batches', type=int, default=16)
  parser.add_argument('--seconds', type=float, default=8.)
  parser.add_argument('--sample-interval', type=float, default=.2)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--output', type=Path, default=ROOT / 'outputs/transition_entries/entries.npz')
  parser.add_argument('--teacher-checkpoint', type=Path, default=DEFAULT_TEACHER)
  parser.add_argument('--transition-checkpoint', type=Path, default=DEFAULT_TRANSITION)
  args = parser.parse_args(argv)
  for name in ('num_envs', 'batches', 'seconds', 'sample_interval'):
    value = getattr(args, name)
    if not math.isfinite(value) or value <= 0:
      parser.error(f'--{name.replace("_", "-")} must be finite and positive')
  return args


def sample_targets(num_envs, rng):
  """Stratify each requested coordinate, including both range endpoints."""
  lower = np.array([-.5, -.4, -.8, -.3, -.12, .28])
  upper = np.array([1., .4, .8, .3, .12, .35])
  values = np.empty((num_envs, 6))
  for column in range(6):
    if num_envs == 1:
      fractions = rng.uniform(size=1)
    else:
      fractions = (np.arange(num_envs) + rng.uniform(size=num_envs)) / num_envs
      fractions[0], fractions[-1] = 0., 1.
      rng.shuffle(fractions)
    values[:, column] = lower[column] + fractions * (upper[column]-lower[column])
  # Preserve enough independent easy trajectories for the first course's
  # held-out evaluation; wide random XY rarely lands near the reference.
  available = np.flatnonzero(((values > lower) & (values < upper)).all(axis=1))
  easy = min(num_envs//4, len(available))
  chosen = rng.choice(available, size=easy, replace=False)
  values[chosen, :3] = rng.uniform([-.15, -.08, -.15], [.15, .08, .15], (easy, 3))
  values[chosen, 3:] = rng.uniform([.07947, -.01855, .32], [.13947, .04145, .35], (easy, 3))
  # Prevent roundoff from crossing the public command limits.
  return values.clip(lower, upper)


def capture_snapshot(env, executed_history, trajectory_ids):
  """Copy root WORLD twists and selected teacher actions into the bank schema."""
  import torch
  from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat
  robot = env.scene['robot']
  data = robot.data
  action = env.action_manager.get_term('joint_pos')
  term = env.command_manager.get_term('twist')
  motion = env.command_manager.get_term('motion')
  root = torch.cat((data.root_link_pos_w-env.scene.env_origins, data.root_link_quat_w,
    data.root_link_lin_vel_w, data.root_link_ang_vel_w), -1)
  anchor = motion.motion_anchor_body_index
  ref_quat = motion.motion.body_quat_w[0, anchor].expand(len(root), -1)
  rotation = quat_mul(yaw_quat(data.root_link_quat_w), quat_inv(yaw_quat(ref_quat)))
  ref_root = quat_apply(rotation, motion.motion.body_pos_w[0, anchor].expand(len(root), -1))
  translation = data.root_link_pos_w-ref_root
  translation = translation.clone()
  translation[:, 2] = 0.  # XY/yaw alignment preserves reference world height.
  ref_fr = motion.motion.body_pos_w[0, motion.cfg.body_names.index('FR_foot')]
  aligned_fr = quat_apply(rotation, ref_fr.expand(len(root), -1))+translation
  foot_ids = robot.find_bodies(('FR_foot',), preserve_order=True)[0]
  fr = data.body_link_pos_w[:, foot_ids[0]]
  values = dict(root_state=root, joint_pos=data.joint_pos, joint_vel=data.joint_vel,
    raw_action=action.raw_action, previous_action=action.previous_action,
    manager_actions=executed_history, previous_joint_vel=action.previous_joint_vel,
    limiter_position=action.target_limiter.position,
    limiter_velocity=action.target_limiter.velocity,
    actual_speed=data.root_link_lin_vel_w[:, :2].norm(dim=-1),
    fr_error=(fr-aligned_fr).norm(dim=-1), fr_offset=term.foot_pos_anchor-term.zero,
    command=term.command, trajectory_id=trajectory_ids)
  return {name: value.detach().cpu().numpy().copy() for name, value in values.items()}


def valid_entries(snapshot, failed):
  """Accept transient/held high FR states without speed or tracking filters."""
  valid = ~np.asarray(failed, dtype=bool).copy()
  for value in snapshot.values():
    valid &= np.isfinite(value).reshape(len(valid), -1).all(-1)
  valid &= (snapshot['fr_offset'][:, 2] >= .25) & (snapshot['fr_offset'][:, 2] <= .39)
  return valid


def save_bank(output, samples, metadata):
  if not samples or not sum(len(sample['trajectory_id']) for sample in samples):
    raise ValueError('No valid handoff states collected; increase seconds/batches or inspect failures')
  arrays = {key: np.concatenate([sample[key] for sample in samples]) for key in samples[0]}
  if not all(np.isfinite(value).all() for value in arrays.values()):
    raise ValueError('Nonfinite bank values are forbidden')
  counts, _, _ = np.histogram2d(arrays['actual_speed'], arrays['fr_error'],
    bins=(SPEED_EDGES, ERROR_EDGES))
  coverage = dict(states=len(arrays['trajectory_id']),
    trajectories=len(np.unique(arrays['trajectory_id'])),
    speed_edges=[str(value) if not math.isfinite(value) else value for value in SPEED_EDGES],
    fr_error_edges=[str(value) if not math.isfinite(value) else value for value in ERROR_EDGES],
    bin_counts=counts.astype(int).tolist(),
    missing_speed_bins=np.flatnonzero(counts.sum(axis=1) == 0).tolist(),
    actual_speed_min=float(arrays['actual_speed'].min()), actual_speed_max=float(arrays['actual_speed'].max()),
    fr_error_min=float(arrays['fr_error'].min()), fr_error_max=float(arrays['fr_error'].max()))
  metadata = dict(metadata, coverage=coverage)
  output = Path(output)
  output.parent.mkdir(parents=True, exist_ok=True)
  # Atomic replacement avoids partial archives if a collection is interrupted.
  temporary = output.with_name(output.name+'.tmp')
  with temporary.open('wb') as stream:
    np.savez_compressed(stream, **arrays,
      metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
  temporary.replace(output)
  return coverage


def file_sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024*1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def compiled_physics_sha256(model):
  """Use the exact physical-array contract used by transition checkpoints."""
  digest = hashlib.sha256()
  for name in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
    'body_pos', 'body_quat', 'jnt_range', 'jnt_axis',
    'geom_type', 'geom_size', 'geom_pos', 'geom_quat',
    'geom_friction', 'dof_armature', 'dof_damping'):
    digest.update(name.encode())
    digest.update(getattr(model, name).tobytes())
  return digest.hexdigest()


def collect(args):
  """Run independent trajectories; discard terminal and post-reset samples."""
  os.environ.setdefault('OMP_NUM_THREADS', '1')
  os.environ.setdefault('MPLCONFIGDIR', '/tmp/transition_t_mpl')
  os.environ.setdefault('XDG_CACHE_HOME', '/tmp/transition_t_cache')
  sys.path.insert(0, str(ROOT))
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.utils.torch import configure_torch_backends
  from src.tasks.transition_t.keyboard import keyboard_transition_env_cfg, AlignedKeyboardTeacher
  from src.tasks.pedipulation.robot import SNAPSHOT_PATH
  teacher_path = args.teacher_checkpoint.expanduser().resolve()
  transition_path = args.transition_checkpoint.expanduser().resolve()
  for path in (teacher_path, transition_path):
    if not path.is_file():
      raise FileNotFoundError(f'Checkpoint not found: {path}')
  configure_torch_backends()
  torch.set_num_threads(1)
  cfg = keyboard_transition_env_cfg(args.num_envs)
  cfg.seed = args.seed
  cfg.commands['twist'].initial_stage = 3
  cfg.commands['twist'].tripod_velocity_limits = (1., .4, .8)
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    term = env.command_manager.get_term('twist')
    motion = env.command_manager.get_term('motion')
    action = env.action_manager.get_term('joint_pos')
    action.source_teacher = AlignedKeyboardTeacher(teacher_path, 'loco_pedipulation_t', args.device)
    if action.source_teacher.contract['actor_dim'] != 54:
      raise ValueError('Entry collection requires the frozen actor54 locomotion teacher')
    # Keep command/reference behavior but never permit automatic policy handoff.
    term.advance = lambda: None
    rng = np.random.default_rng(args.seed)
    interval = max(1, math.ceil(args.sample_interval/env.step_dt-1e-9))
    steps = math.ceil(args.seconds/env.step_dt)
    samples, failures = [], 0
    # Random session prefix plus monotonic reset IDs prevents ID collisions
    # between batches, resets, and separately generated archives.
    next_trajectory = (uuid.uuid4().int & ((1 << 30)-1)) << 32
    ids = torch.arange(env.num_envs, device=env.device)
    zero = torch.zeros(env.num_envs, 12, device=env.device)
    all_requested = []
    with torch.inference_mode():
      for batch in range(args.batches):
        env.reset(seed=args.seed+batch)
        action.motor_offsets.zero_()
        action.kp_multipliers.fill_(1.)
        action.kd_multipliers.fill_(1.)
        history = torch.zeros(env.num_envs, 3, 12, device=env.device)
        trajectories = ids + next_trajectory
        next_trajectory += env.num_envs
        targets = torch.tensor(sample_targets(env.num_envs, rng), dtype=torch.float32, device=env.device)
        all_requested.append(targets.cpu().numpy())
        # Different settle/raise clocks yield moving and held samples.
        settle = torch.tensor(rng.uniform(.6, 1.2, env.num_envs), device=env.device)
        raise_time = torch.tensor(rng.uniform(.8, 2., env.num_envs), device=env.device)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        before = sum(len(sample['trajectory_id']) for sample in samples)
        for step in range(steps):
          elapsed = step*env.step_dt
          commands = targets.clone()
          progress = ((elapsed-settle)/raise_time).clamp(0., 1.)
          commands[:, 3:] *= progress[:, None]
          active = elapsed >= settle
          term.submit(ids, commands, active)
          _, _, terminated, timed_out, _ = env.step(zero)
          failed = terminated.bool() | timed_out.bool()
          failures += int((alive & failed).sum())
          alive &= ~failed
          # Manager receives zero placeholders; the action term selects the
          # actual teacher. Record that selected common action, never zeros.
          history[:, 2] = history[:, 1].clone()
          history[:, 1] = history[:, 0].clone()
          history[:, 0] = action.raw_action
          if step % interval == 0:
            snapshot = capture_snapshot(env, history, trajectories)
            keep = valid_entries(snapshot, (~alive).cpu().numpy())
            if keep.any():
              samples.append({key: value[keep] for key, value in snapshot.items()})
          # Failed worlds are excluded for the remainder of this batch, even
          # though the framework has reset them. Start fresh next batch.
          if not alive.any():
            break
        after = sum(len(sample['trajectory_id']) for sample in samples)
        print(json.dumps(dict(batch=batch+1, batches=args.batches, collected=after-before,
          total=after, surviving_trajectories=int(alive.sum()), failures=failures)), flush=True)
    requested = np.concatenate(all_requested)
    physics_sha = compiled_physics_sha256(env.sim.mj_model)
    source_sha = file_sha256(SNAPSHOT_PATH)
    metadata = dict(schema_version=2, teacher_checkpoint=str(teacher_path),
      target_limiter=action.target_limiter.contract(), action_scale=action.cfg.scale,
      action_zero=action.default_angles[0].cpu().tolist(),
      teacher_sha256=file_sha256(teacher_path), transition_checkpoint=str(transition_path),
      transition_sha256=file_sha256(transition_path), motion_sha256=motion.reference.sha256,
      compiled_physics_sha256=physics_sha, source_physics_sha256=source_sha,
      root_state_coordinates='env_origin_relative_position, quaternion_wxyz, world_linear_angular_velocity',
      action_coordinates='common_default_joint_angles_plus_0.25_times_action',
      manager_action_order=['current', 'previous', 'previous_previous'],
      split_rule='trajectory_id modulo 5: 0 heldout; 1,2,3,4 train',
      source_physics=dict(environment='keyboard_transition_env_cfg', domain_randomization=False,
        compiled_physics_sha256=physics_sha, source_physics_sha256=source_sha,
        motor_offsets=0., action_scale=action.cfg.scale,
        teacher_substep_delay=True, delay_history='previous_requested_action', decimation=cfg.decimation,
        physics_dt=env.physics_dt, control_dt=env.step_dt,
        tripod_velocity_limits=list(cfg.commands['twist'].tripod_velocity_limits),
        sim=asdict(cfg.sim)),
      collection=dict(device=args.device, num_envs=args.num_envs, batches=args.batches,
        seconds=args.seconds, sample_interval=args.sample_interval, seed=args.seed,
        near_reference_requested_fraction=.25,
        failed_trajectories=failures, actual_fr_dz_range=[.25, .39],
        requested_min=requested.min(axis=0).tolist(), requested_max=requested.max(axis=0).tolist()))
    report = save_bank(args.output, samples, metadata)
    print('TRANSITION_ENTRY_BANK '+json.dumps(dict(output=str(args.output), **report), allow_nan=False), flush=True)
    if report['missing_speed_bins']:
      print('COVERAGE_WARNING missing actual speed bins: '+str(report['missing_speed_bins']), flush=True)
    return report
  finally:
    env.close()


if __name__ == '__main__':
  collect(parse_args())
