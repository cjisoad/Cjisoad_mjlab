"""Physical entry-state distribution and held-out curriculum decisions."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


def module():
  assert importlib.util.find_spec('src.tasks.transition_t.entry_bank') is not None, 'entry state bank is missing'
  return importlib.import_module('src.tasks.transition_t.entry_bank')


def write_bank(path, *, corrupt=None):
  n = 100
  root = np.zeros((n, 13), np.float32)
  root[:, 2] = .3
  root[:, 3] = 1.
  speed = np.tile(np.array([.1, .3, .5, .9], np.float32), 25)
  root[:, 7] = speed
  data = dict(root_state=root, joint_pos=np.zeros((n, 12), np.float32),
    joint_vel=np.ones((n, 12), np.float32), raw_action=np.full((n, 12), .3, np.float32),
    previous_action=np.full((n, 12), .2, np.float32),
    manager_actions=np.stack([np.full((n, 12), v, np.float32) for v in (.3, .2, .1)], 1),
    previous_joint_vel=np.ones((n, 12), np.float32), actual_speed=speed,
    fr_error=np.tile(np.array([.02, .07, .13, .3], np.float32), 25),
    fr_offset=np.zeros((n, 3), np.float32), command=np.zeros((n, 6), np.float32),
    trajectory_id=np.arange(n, dtype=np.int64),
    metadata_json=np.array(json.dumps(dict(schema_version=1, motion_sha256='motion', teacher_sha256='teacher'))))
  if corrupt:
    corrupt(data)
  np.savez_compressed(path, **data)


class EntryBankTests(unittest.TestCase):
  def test_unlimited_support_joint_state_is_preserved_with_fr_only_limits(self):
    contract = dict(velocity_limits=[None]*3+[2.]*3+[None]*6,
      acceleration_limits=[None]*3+[30.]*3+[None]*6)
    def limited(data):
      data['limiter_position'] = np.full((100, 12), .075, np.float32)
      data['limiter_velocity'] = np.full((100, 12), 100., np.float32)
      data['limiter_velocity'][:, 3:6] = .5
      data['metadata_json'] = np.array(json.dumps(dict(schema_version=2, motion_sha256='motion',
        teacher_sha256='teacher', target_limiter=contract, action_scale=.25, action_zero=[0.]*12)))
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'bank.npz'; write_bank(p, corrupt=limited)
      bank = module().EntryStateBank(p, motion_sha256='motion', target_limiter_contract=contract)
      assert bank.data['limiter_velocity'][0, 0] == 100.

  def test_limited_banks_require_matching_config_and_physical_limiter_state(self):
    m = module()
    contract = dict(velocity_limits=[2.]*12, acceleration_limits=[30.]*12)
    def limited(data):
      data['limiter_position'] = np.full((100, 12), .075, np.float32)
      data['limiter_velocity'] = np.full((100, 12), .5, np.float32)
      data['metadata_json'] = np.array(json.dumps(dict(schema_version=2, motion_sha256='motion',
        teacher_sha256='teacher', target_limiter=contract, action_scale=.25, action_zero=[0.]*12)))
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'bank.npz'; write_bank(p)
      with self.assertRaisesRegex(ValueError, 'limiter|recollect'):
        m.EntryStateBank(p, motion_sha256='motion', target_limiter_contract=contract)
      write_bank(p, corrupt=limited)
      bank = m.EntryStateBank(p, motion_sha256='motion', target_limiter_contract=contract)
      torch.testing.assert_close(bank.data['limiter_velocity'], torch.full((100, 12), .5))
      with self.assertRaisesRegex(ValueError, 'action scale'):
        m.EntryStateBank(p, motion_sha256='motion', target_limiter_contract=contract, action_scale=.5)
      with self.assertRaisesRegex(ValueError, 'action zero'):
        m.EntryStateBank(p, motion_sha256='motion', target_limiter_contract=contract, action_zero=[.1]*12)
      with self.assertRaisesRegex(ValueError, 'limiter'):
        m.EntryStateBank(p, motion_sha256='motion', target_limiter_contract={'different': True})
      def too_fast(data):
        limited(data); data['limiter_velocity'][0, 0] = 3.
      write_bank(p, corrupt=too_fast)
      with self.assertRaisesRegex(ValueError, 'limiter.*velocity'):
        m.EntryStateBank(p, motion_sha256='motion', target_limiter_contract=contract)
      def inconsistent(data):
        limited(data); data['limiter_position'][0, 0] += .1
      write_bank(p, corrupt=inconsistent)
      with self.assertRaisesRegex(ValueError, 'limiter.*action'):
        m.EntryStateBank(p, motion_sha256='motion', target_limiter_contract=contract)

  def test_schema_rejects_nonfinite_and_wrong_reference(self):
    m = module()
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'bank.npz'
      write_bank(p)
      with self.assertRaisesRegex(ValueError, 'reference|motion'):
        m.EntryStateBank(p, motion_sha256='other')
      write_bank(p, corrupt=lambda a: a['joint_vel'].__setitem__((0, 0), np.nan))
      with self.assertRaisesRegex(ValueError, 'finite'):
        m.EntryStateBank(p, motion_sha256='motion')

  def test_trajectory_splits_and_difficulty_are_enforced(self):
    m = module()
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'bank.npz'; write_bank(p)
      bank = m.EntryStateBank(p, motion_sha256='motion')
      train = bank.sample(200, split='train', max_speed=.4, max_fr_error=.1)
      held = bank.sample(200, split='holdout', max_speed=.4, max_fr_error=.1)
      assert (bank.data['trajectory_id'][train] % 5 != 0).all()
      assert (bank.data['trajectory_id'][held] % 5 == 0).all()
      assert (bank.data['actual_speed'][train] <= .4).all()
      assert (bank.data['fr_error'][held] <= .1).all()
      assert not set(train.tolist()) & set(held.tolist())

  def test_empty_difficulty_is_explicit_error(self):
    m = module()
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'bank.npz'; write_bank(p)
      bank = m.EntryStateBank(p, motion_sha256='motion')
      with self.assertRaisesRegex(ValueError, 'eligible|empty'):
        bank.sample(1, split='train', max_speed=.01, max_fr_error=.01)

  def test_sampler_balances_occupied_bins_instead_of_duplicate_frames(self):
    m = module()
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'bank.npz'
      def skew(a):
        a['actual_speed'][:] = .1; a['root_state'][:, 7] = .1; a['fr_error'][:] = .02
        a['actual_speed'][1] = .9; a['root_state'][1, 7] = .9; a['fr_error'][1] = .3
      write_bank(p, corrupt=skew)
      bank = m.EntryStateBank(p, motion_sha256='motion')
      torch.manual_seed(17)
      ids = bank.sample(2000, split='train', max_speed=1.2, max_fr_error=.5)
      hard_ratio = float((ids == 1).float().mean())
      assert .4 < hard_ratio < .6, hard_ratio

  def test_evaluation_sampler_uses_distinct_trajectories_when_available(self):
    m = module()
    with tempfile.TemporaryDirectory() as d:
      p = Path(d)/'bank.npz'; write_bank(p)
      bank = m.EntryStateBank(p, motion_sha256='motion')
      ids = bank.sample_trials(20, split='holdout', max_speed=1.2, max_fr_error=.5)
      assert len(ids) == 20
      assert len(bank.data['trajectory_id'][ids].unique()) == 20
      assert (bank.data['trajectory_id'][ids] % 5 == 0).all()
      ids = bank.sample_trials(64, split='holdout', max_speed=.25, max_fr_error=.08)
      assert len(ids) == 64
      assert len(bank.data['trajectory_id'][ids].unique()) == 5

  def test_repeated_heldout_trajectories_cannot_promote_curriculum(self):
    course = module().EntryCurriculum()
    assert not course.observe(64, 64, distinct_trajectories=48)
    assert not course.observe(64, 64, distinct_trajectories=48)
    assert course.level == 0
    assert course.streak == 0

  def test_curriculum_requires_two_sufficient_held_out_success_windows(self):
    m = module()
    course = m.EntryCurriculum()
    assert course.level == 0
    assert not course.observe(8, 8, distinct_trajectories=8)
    assert not course.observe(60, 64, distinct_trajectories=64)
    assert course.observe(60, 64, distinct_trajectories=64)
    assert course.level == 1
    assert not course.observe(20, 64, distinct_trajectories=64)
    assert course.streak == 0
    saved = course.state_dict()
    resumed = m.EntryCurriculum()
    resumed.load_state_dict(saved)
    assert resumed.state_dict() == saved
    assert resumed.settings['entry_probability'] < .6


if __name__ == '__main__':
  unittest.main()
