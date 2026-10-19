"""Deployment and evaluation must remain independent of the teacher."""
from pathlib import Path
import torch
import pytest


def test_keyboard_r_is_six_zeros_even_with_velocity_key(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from keyboard_teacher import TeacherKeyboardController, workspace_for_mode
    from src.tasks.legmanip_distil.deployment import update_keyboard_intent
    controller = TeacherKeyboardController(workspace_for_mode('union'))
    controller.velocity = [.3, .1, .2]
    controller.target = (.1, 0., .45)
    class Intent:
        def return_to_quadruped(self, ids):
            self.values = [0.] * 6
        def set_command(self, ids, values):
            self.values = values[0]
    intent = Intent()
    update_keyboard_intent(intent, controller, {'r', 'w'}, .02)
    assert intent.values == [0.] * 6
    assert controller.command == (0.,) * 6


def test_evaluation_requires_physical_goals_and_complete_scenario():
    from src.tasks.legmanip_distil.evaluation import episode_result
    record = dict(scenario='cycle', steps=500, completed=True, physical_failure=False,
                  deadline=False, biped_hold=0., tripod_hold=1., final_quad_hold=1.,
                  velocity_error_sum=0., foot_error_sum=0., foot_samples=1,
                  entry_trajectory=None, substep_difference_sum=0.,
                  substep_difference_max=0.)
    assert not episode_result(record, .02)['success']
    record['biped_hold'] = .5
    assert episode_result(record, .02)['success']
    record['physical_failure'] = True
    assert not episode_result(record, .02)['success']
    record['physical_failure'] = False
    record['completed'] = False
    assert not episode_result(record, .02)['success']


def test_student_world_blocks_teacher_and_motion_loading(tmp_path, monkeypatch):
    from src.tasks.legmanip_distil.rl import make_student, save_student
    from src.tasks.legmanip_distil.observations import action_contract, observation_contract
    from src.tasks.legmanip_distil.deployment import create_student_world, run_headless
    from src.tasks.legmanip_distil.env_cfg import legmanip_distil_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.legmanip_distil.teacher import FSMTeacher
    from src.tasks.transition_t.commands import TransitionMotion
    from src.tasks.transition_down_v2.data import LandingBank
    def forbidden(*args, **kwargs):
        pytest.fail('Standalone quadruped deployment tried to load a teacher or motion/entry archive')
    for cls in (FSMTeacher, TransitionMotion, LandingBank):
        monkeypatch.setattr(cls, '__init__', forbidden)
    cfg = legmanip_distil_env_cfg(play=True)
    env = ManagerBasedRlEnv(cfg, device='cpu')
    student = make_student()
    manifest = dict(version=1, completed_updates=0, student_cfg=student.model_cfg,
                    observation_contract=observation_contract(), action_contract=action_contract(env))
    env.close()
    checkpoint = tmp_path / 'student.pt'
    save_student(checkpoint, student, manifest)
    env, wrapped, model, _ = create_student_world(checkpoint)
    try:
        assert set(env.cfg.commands) == {'intent'}
        result = run_headless(wrapped, model, 4)
        assert result['steps'] == 4
        assert not result['teacher_loaded']
        wrapped.reset()
        state = model.get_hidden_state()
        assert state is None or torch.count_nonzero(state) == 0
    finally:
        env.close()
