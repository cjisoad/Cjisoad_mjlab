"""Real-world parity and isolation for imagined four-teacher queries."""
import torch

def test_shadow_teacher_matches_real_labels_without_mutating_live_world():
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.legmanip_distil.env_cfg import legmanip_distil_env_cfg
    from src.tasks.legmanip_distil.teacher import FSMTeacher
    from src.tasks.legmanip_distil.supermimic.teacher import ImaginedTeacher, capture_state
    cfg = legmanip_distil_env_cfg()
    cfg.scene.num_envs = 2
    cfg.commands['intent'].scenario = 'cycle'
    env = ManagerBasedRlEnv(cfg, device='cpu')
    try:
        env.reset()
        real = FSMTeacher(env)
        shadow = ImaginedTeacher(real)
        try:
            for owner in (0, 2, 3, 4):
                term = env.command_manager.get_term('twist')
                if owner in (2, 4):
                    term.start_transition(torch.arange(2), owner)
                else:
                    term.owner[:] = owner
                before = env.sim.data.qpos.clone()
                shadow.seed()
                if owner == 0:
                    velocity = torch.tensor([[.12,.01,-.03],[.18,-.01,.04]])
                    shadow.set_walking_velocity(torch.arange(2), velocity)
                    torch.testing.assert_close(shadow.command[:, :3], velocity)
                    torch.testing.assert_close(shadow.teacher.term.command[:, :3], velocity)
                    torch.testing.assert_close(shadow.teacher.term.requested_velocity, velocity)
                    shadow.reset()
                state = capture_state(env)
                rng = torch.get_rng_state()
                labels = real().clone()
                torch.set_rng_state(rng)
                predicted = shadow.query(state)
                torch.testing.assert_close(labels, predicted, atol=2e-5, rtol=0.)
                torch.testing.assert_close(env.sim.data.qpos.clone(), before)
                assert (term.owner == owner).all()
                real.set_control_mask(torch.zeros(2, dtype=torch.bool))
                env.step(labels)
                next_state = capture_state(env)
                shadow.advance(next_state, labels)
                assert not shadow.teacher.action.executed_velocity.any()
                rng = torch.get_rng_state()
                next_labels = real().clone()
                torch.set_rng_state(rng)
                torch.testing.assert_close(shadow.query(next_state), next_labels,
                                           atol=2e-5, rtol=0.)
        finally:
            shadow.close()
    finally:
        env.close()
