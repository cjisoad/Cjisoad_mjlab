import torch
from src.tasks.legmanip_distil.supermimic.augmentation import mirror_state, mirror_observation, mirror_action

def test_mirror_twice_recovers_state_actions_and_observations_with_asymmetric_default():
    state = torch.randn(5,37)
    obs = torch.randn(5,48)
    action = torch.randn(5,12)
    default = torch.randn(12)
    torch.testing.assert_close(mirror_state(mirror_state(state)), state)
    torch.testing.assert_close(mirror_observation(mirror_observation(obs,default),default), obs)
    torch.testing.assert_close(mirror_action(mirror_action(action,default),default), action,atol=1e-5,rtol=1e-5)


def test_symmetry_uses_carried_original_and_mirrored_histories():
    from src.tasks.legmanip_distil.rl.model import make_student
    from src.tasks.legmanip_distil.supermimic.augmentation import sequence_symmetry,mirror_observation,mirror_action
    torch.manual_seed(8)
    student=make_student(num_envs=2)
    obs=torch.randn(3,2,48)*.2
    done=torch.zeros(3,2,dtype=torch.bool)
    h=torch.randn(1,2,256)*.4
    mh=torch.randn(1,2,256)*.4
    default=torch.zeros(2,12)
    errors,gates=sequence_symmetry(student,obs,done,torch.ones(3,2,dtype=torch.bool),
                                 default,h,mh,return_errors=True)
    p,_=student.forward_sequence(obs,done,h)
    m,_=student.forward_sequence(mirror_observation(obs,default),done,mh)
    torch.testing.assert_close(errors,(m-mirror_action(p,default)).square().mean(-1))
