import unittest
import numpy as np
import torch
from src.tasks.transition_down_v2.safety import knee_mesh_vertices, mesh_clearance, SafetyMonitor
from src.tasks.transition_down_v2.evaluation import LandingTrials
from src.tasks.transition_down_v2.gates import GateConfig

def metrics(n=1):
    return dict(orientation_error=torch.zeros(n),height_error=torch.zeros(n),linear_speed=torch.zeros(n),
        angular_speed=torch.zeros(n),joint_speed=torch.zeros(n),feet_rms=torch.zeros(n),
        tripod_supported=torch.ones(n,dtype=torch.bool),fr_clear=torch.ones(n,dtype=torch.bool))
def safety(knee=.06,nonfoot=False,fr=False,landing=1.7):
    return dict(knee_min=torch.tensor([knee]),nonfoot=torch.tensor([nonfoot]),
        fr_contact=torch.tensor([fr]),landing_time=torch.tensor([landing]))

class LandingAcceptance(unittest.TestCase):
    def test_intermediate_unsafe_sample_cannot_be_hidden_by_good_endpoint(self):
        t=LandingTrials(1,.02)
        t.record(torch.tensor([False]),metrics(),torch.tensor([False]),torch.tensor([False]),safety=safety(.035),step=1)
        self.assertTrue(bool(t.finished[0]));self.assertFalse(bool(t.standing_held[0]))
        self.assertEqual(t.summary()['unsafe_attempts'],1)

    def test_hold_is_uninterrupted_and_late_landing_fails(self):
        t=LandingTrials(1,.02)
        for i in range(40):t.record(torch.tensor([True]),metrics(),torch.tensor([False]),torch.tensor([False]),safety=safety(),step=i)
        m=metrics();m['linear_speed'][:]=1
        t.record(torch.tensor([True]),m,torch.tensor([False]),torch.tensor([False]),safety=safety(),step=40)
        for i in range(41,81):t.record(torch.tensor([True]),metrics(),torch.tensor([False]),torch.tensor([False]),safety=safety(),step=i)
        self.assertFalse(bool(t.standing_held[0]))
        late=LandingTrials(1,.02)
        late.record(torch.tensor([True]),metrics(),torch.tensor([False]),torch.tensor([False]),safety=safety(landing=2.5),step=1)
        self.assertTrue(bool(late.finished[0]))
        self.assertFalse(bool(late.standing_held[0]))

    def test_mesh_clearance_matches_native_geometry(self):
        import mujoco
        from src.tasks.pedipulation_transition_t.motion import make_model
        from src.tasks.transition_down_v2.data import MOTION_FILE
        model=make_model();d=mujoco.MjData(model)
        with np.load(MOTION_FILE) as z:poses=z['qpos'][[0,60,-1]]
        geoms,vertices=knee_mesh_vertices(model)
        for pose in poses:
            d.qpos[:]=pose;mujoco.mj_forward(model,d)
            actual=mesh_clearance(torch.tensor(d.geom_xpos[None],dtype=torch.float64),
                torch.tensor(d.geom_xmat[None],dtype=torch.float64),geoms,vertices)
            expected=min(float((v@d.geom_xmat[g].reshape(3,3)[2]+d.geom_xpos[g,2]).min()) for g,v in zip(geoms,vertices))
            self.assertAlmostEqual(float(actual[0]),expected,places=7)

if __name__=='__main__':unittest.main()
