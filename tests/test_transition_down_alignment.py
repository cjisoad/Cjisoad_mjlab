"""Biped heading stays continuous through vertical and preserves live state."""
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul, yaw_quat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/biped_to_tripod_preparation_20261010'


class DownAlignmentTests(unittest.TestCase):
  def test_vertical_heading_and_roll_are_continuous(self):
    from src.tasks.transition_down_t.commands import teacher_heading_quat
    pitch = torch.tensor([-1.56, -1.58])
    zero = torch.zeros_like(pitch)
    q = quat_from_euler_xyz(zero, pitch, zero)
    heading = teacher_heading_quat(q)
    torch.testing.assert_close(heading, torch.tensor([[1.,0.,0.,0.]]).expand(2,-1))
    self.assertLess(float((yaw_quat(q)[0]*yaw_quat(q)[1]).sum().abs()), .1)
    roll = quat_from_euler_xyz(torch.tensor([.4]), torch.zeros(1), torch.zeros(1))
    torch.testing.assert_close(teacher_heading_quat(roll), torch.tensor([[1.,0.,0.,0.]]))
    degenerate = quat_from_euler_xyz(torch.tensor([torch.pi/2]), torch.zeros(1), torch.zeros(1))
    self.assertTrue(bool(torch.isfinite(teacher_heading_quat(degenerate)).all()))

  def test_same_real_bank_errors_reproduce_heading_diagnosis(self):
    from src.tasks.transition_down_t.commands import teacher_heading_quat
    with np.load(DATA/'source_teacher_rollouts.npz') as z:
      root = torch.tensor(z['root_state'])
      feet = torch.tensor(z['feet'][:,1])
      valid = torch.tensor(z['alive'] & (z['time_s']>=2.) & (z['gravity'][:,0]<-.8)
        & (z['root_state'][:,2]>.4) & ~z['found'][:,:2].any(1) & z['found'][:,2:].any(1))
    with np.load(DATA/'biped_to_tripod_reference.npz') as z:
      names=z['body_names'].tolist(); b=names.index('base_link'); f=names.index('FR_foot')
      q=torch.tensor(z['body_quat_w'][0,b],dtype=root.dtype).expand(len(root),-1)
      p=torch.tensor(z['body_pos_w'][0,b],dtype=root.dtype).expand(len(root),-1)
      foot=torch.tensor(z['body_pos_w'][0,f],dtype=root.dtype).expand(len(root),-1)
    errors=[]
    for heading in (yaw_quat,teacher_heading_quat):
      rotation=quat_mul(heading(root[:,3:7]),quat_inv(heading(q)))
      shift=root[:,:3]-quat_apply(rotation,p);shift[:,2]=0.
      errors.append((feet-quat_apply(rotation,foot)-shift).norm(dim=-1)[valid].median())
    self.assertAlmostEqual(float(errors[0]),.681614,places=4)
    self.assertAlmostEqual(float(errors[1]),.033253,places=4)

  def test_live_entry_changes_reference_only_and_subset(self):
    from src.tasks.transition_down_t.commands import DownMotion
    cfg=SimpleNamespace(body_names=('base_link',))
    robot=SimpleNamespace(data=SimpleNamespace(body_link_pos_w=torch.tensor([[[1.,2.,.5]],[[3.,4.,.6]]]),
      body_link_quat_w=torch.tensor([[[.7,0.,-.7141428,0.]],[[1.,0.,0.,0.]]])))
    term=object.__new__(DownMotion)
    term.cfg=cfg;term.robot=robot;term.robot_anchor_body_index=0;term.motion_anchor_body_index=0
    term._env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(2,3)),common_step_counter=100)
    term.motion=SimpleNamespace(body_quat_w=torch.tensor([[[.7,0.,-.7141428,0.]]]),body_pos_w=torch.tensor([[[0.,0.,.49]]]))
    term.time_steps=torch.tensor([12,13]);term.completed=torch.tensor([True,True])
    term.reference_rotation=torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]])
    term.reference_translation=torch.zeros(2,3);term.time_left=torch.zeros(2)
    term._last_update_step=torch.zeros(2,dtype=torch.long);term._reset_step=torch.tensor([0,0])
    term.cfg.resampling_time_range=(1e9,1e9);term._refresh_relative=lambda: None
    before={k:v.clone() for k,v in vars(robot.data).items()}
    term.start_from_current_state(torch.tensor([0]))
    for key,value in before.items():torch.testing.assert_close(value,getattr(robot.data,key),rtol=0.,atol=0.)
    self.assertEqual(term.time_steps.tolist(),[0,13]);self.assertEqual(term.completed.tolist(),[False,True])
    torch.testing.assert_close(term.reference_translation[0],torch.tensor([1.,2.,0.]))
    torch.testing.assert_close(term.reference_translation[1],torch.zeros(3))
    self.assertEqual(term._reset_step.tolist(),[0,0])

if __name__=='__main__':unittest.main()
