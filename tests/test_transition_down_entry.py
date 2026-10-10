from dataclasses import asdict
import tempfile
from pathlib import Path
import unittest
import torch

class DownEntryTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.transition_down_t.entry_course import down_entry_env_cfg
    from src.tasks.transition_down_t.data import DEFAULT_ENTRY_BANK
    torch.set_num_threads(1)
    cfg=down_entry_env_cfg(DEFAULT_ENTRY_BANK,num_envs=3,play=True)
    cfg.commands['motion'].entry_probability=1.;cfg.commands['motion'].nominal_probability=0.
    cls.env=ManagerBasedRlEnv(cfg,device='cpu')
  @classmethod
  def tearDownClass(cls):cls.env.close()
  def test_real_entry_restoration_and_stable_heading(self):
    env=self.env;obs,_=env.reset(seed=42);t=env.command_manager.get_term('motion')
    ids=t.entry_index;bank=t.bank.data;robot=env.scene['robot'].data;a=env.action_manager.get_term('joint_pos')
    for key in ('joint_pos','joint_vel'):torch.testing.assert_close(getattr(robot,key),bank[key][ids],rtol=0,atol=0)
    torch.testing.assert_close(robot.root_link_pos_w,bank['root_state'][ids,:3]+env.scene.env_origins,atol=1e-6,rtol=0)
    torch.testing.assert_close(robot.root_link_lin_vel_w,bank['root_state'][ids,7:10],atol=1e-6,rtol=0)
    for key in ('raw_action','previous_action','previous_joint_vel'):torch.testing.assert_close(getattr(a,key),bank[key][ids],rtol=0,atol=0)
    for i,target in enumerate((env.action_manager.action,env.action_manager.prev_action,env.action_manager.prev_prev_action)):
      torch.testing.assert_close(target,bank['manager_actions'][ids,i],rtol=0,atol=0)
    for key in ('position','velocity'):torch.testing.assert_close(getattr(a.target_limiter,key),bank['limiter_'+key][ids],rtol=0,atol=0)
    from src.tasks.transition_down_t.commands import teacher_heading_quat
    from mjlab.utils.lab_api.math import quat_inv,quat_mul
    ref=t.motion.body_quat_w[0,t.motion_anchor_body_index].expand(3,-1)
    torch.testing.assert_close(t.reference_rotation,quat_mul(teacher_heading_quat(bank['root_state'][ids,3:7]),quat_inv(teacher_heading_quat(ref))))
    self.assertTrue(bool((t.physical_reset_age_steps>=10).all()))
    self.assertEqual(obs['actor'].shape,(3,75));self.assertEqual(obs['critic'].shape,(3,192))
    for level in range(4):self.assertGreaterEqual(t.bank.coverage()['holdout'][level]['trajectories'],128)
  def test_subset_restore_does_not_change_other_worlds(self):
    env=self.env;env.reset(seed=7);t=env.command_manager.get_term('motion')
    before=env.sim.data.qpos.clone();history=env.action_manager.action.clone()
    t.restore_entries(torch.tensor([1]),t.entry_index[1:2])
    torch.testing.assert_close(env.sim.data.qpos[[0,2]],before[[0,2]],rtol=0,atol=0)
    torch.testing.assert_close(env.action_manager.action[[0,2]],history[[0,2]],rtol=0,atol=0)
