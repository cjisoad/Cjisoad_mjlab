"""Three real support feet and an airborne FR are required at descent end."""
from types import SimpleNamespace
import unittest
import torch

class TripodTests(unittest.TestCase):
  def metrics(self, found, force=6., fr_height=.04):
    from src.tasks.transition_down_t.standing import tripod_endpoint_metrics
    feet=torch.zeros(1,4,3);feet[:,1,2]=fr_height
    q=torch.tensor([[1.,0.,0.,0.]])
    motion=SimpleNamespace(cfg=SimpleNamespace(body_names=['FL_foot','FR_foot','RL_foot','RR_foot']),
      anchor_quat_w=q,robot_anchor_quat_w=q,anchor_pos_w=torch.zeros(1,3),robot_anchor_pos_w=torch.zeros(1,3),
      body_pos_w=feet,robot_anchor_lin_vel_w=torch.zeros(1,3),robot_anchor_ang_vel_w=torch.zeros(1,3))
    robot=SimpleNamespace(find_bodies=lambda *a,**k:([0,1,2,3],None),
      data=SimpleNamespace(body_link_pos_w=feet,joint_vel=torch.zeros(1,12)))
    forces=torch.zeros(1,4,1,3);forces[:,:,:,2]=force
    scene={'robot':robot,'feet_ground_contact':SimpleNamespace(data=SimpleNamespace(
      found=torch.tensor(found).reshape(1,4,1),force=forces))}
    return tripod_endpoint_metrics(SimpleNamespace(num_envs=1,scene=scene,
      command_manager=SimpleNamespace(get_term=lambda _:motion)))

  def test_support_force_and_airborne_fr_gates(self):
    from src.tasks.transition_down_t.standing import tripod_conditions,strict_tripod_endpoint_candidate
    cases=[([1,0,1,1],6.,.04,True),([0,0,1,1],6.,.04,False),
      ([1,1,1,1],6.,.04,False),([1,0,1,1],5.,.04,False),([1,0,1,1],6.,.032,False)]
    for contact,force,height,expected in cases:
      m=self.metrics(contact,force,height)
      self.assertEqual(bool(torch.stack(tuple(tripod_conditions(m).values())).all()),expected)
      self.assertEqual(bool(strict_tripod_endpoint_candidate(torch.tensor([True]),m)),expected)

  def test_world_drift_is_strict_diagnostic_only(self):
    from src.tasks.transition_down_t.standing import tripod_conditions,strict_tripod_endpoint_candidate
    m=self.metrics([1,0,1,1]);m['feet_rms'][:]=.2
    self.assertTrue(bool(torch.stack(tuple(tripod_conditions(m).values())).all()))
    self.assertFalse(bool(strict_tripod_endpoint_candidate(torch.tensor([True]),m)))
