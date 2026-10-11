"""Recompute reference-dependent difficulty without synthesizing teacher states."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from src.tasks.pedipulation_transition_t.motion import make_model
from .entry_bank import EntryStateBank
from .motion import JOINT_NAMES, TransitionMotion


def retarget_entry_bank(bank_file, motion_file, output, *, source_motion_file):
  """Keep every physical state/history/split, replacing only FR error and metadata."""
  source_path, output = Path(bank_file).resolve(), Path(output).resolve()
  if output.exists():
    raise FileExistsError(output)
  old, new = TransitionMotion(source_motion_file), TransitionMotion(motion_file)
  bank = EntryStateBank(source_path, motion_sha256=old.sha256)
  with np.load(source_path, allow_pickle=False) as archive:
    arrays = {key: archive[key].copy() for key in archive.files if key != 'metadata_json'}
  model = make_model(); data = mujoco.MjData(model)
  joints = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
  root_index = new.body_name_to_index['base_link']
  fr_index = new.body_name_to_index['FR_foot']
  reference_delta = new.body_pos_w[0, fr_index]-new.body_pos_w[0, root_index]
  reference_yaw = Rotation.from_quat(new.body_quat_w[0, root_index, [1, 2, 3, 0]]).as_euler('xyz')[2]
  quats = arrays['root_state'][:, [4, 5, 6, 3]]
  state_yaws = Rotation.from_quat(quats).as_euler('xyz')[:, 2]
  aligned_delta = Rotation.from_euler('z', (state_yaws-reference_yaw)[:, None]).apply(reference_delta)
  # XY follows the incoming root, while reference world height stays fixed.
  target = aligned_delta.copy()
  target[:, :2] += arrays['root_state'][:, :2]
  target[:, 2] += new.body_pos_w[0, root_index, 2]
  errors = np.empty(len(target), dtype=np.float32)
  foot = model.body('FR_foot').id
  for i, root in enumerate(arrays['root_state']):
    data.qpos[:7] = root[:7]
    data.qpos[joints] = arrays['joint_pos'][i]
    mujoco.mj_kinematics(model, data)
    errors[i] = np.linalg.norm(data.xpos[foot]-target[i])
  arrays['fr_error'] = errors
  metadata = dict(bank.metadata)
  metadata['motion_sha256'] = new.sha256
  metadata['reference_retarget'] = dict(version=1,
    source_bank_sha256=bank.sha256, source_motion_sha256=old.sha256,
    target_motion_sha256=new.sha256, method='native_mujoco_FR_body_FK_first_frame_xy_yaw_alignment',
    physical_states_and_actions_unchanged=True,
    source_coverage=metadata.get('coverage'))
  metadata['coverage'] = dict(states=len(errors),
    trajectories=len(np.unique(arrays['trajectory_id'])),
    actual_speed_min=float(arrays['actual_speed'].min()), actual_speed_max=float(arrays['actual_speed'].max()),
    fr_error_min=float(errors.min()), fr_error_max=float(errors.max()))
  output.parent.mkdir(parents=True, exist_ok=True)
  temporary = output.with_name(output.name+'.tmp')
  try:
    with temporary.open('xb') as stream:
      np.savez_compressed(stream, **arrays,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    result = EntryStateBank(temporary, motion_sha256=new.sha256)
    report = dict(states=len(errors), source_bank_sha256=bank.sha256,
      target_bank_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),
      source_motion_sha256=old.sha256, target_motion_sha256=new.sha256,
      max_fr_error_change_m=float(np.max(np.abs(errors-bank.data['fr_error'].numpy()))),
      coverage=result.coverage())
    # Exclusive destination creation prevents replacing another prepared bank.
    with output.open('xb') as stream:
      stream.write(temporary.read_bytes())
    return report
  finally:
    temporary.unlink(missing_ok=True)
