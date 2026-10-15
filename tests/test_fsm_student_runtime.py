"""Current-frame student inference on the real, unchanged full-chain teacher env."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


def test_gru_student_observes_real_teacher_worlds_without_reference_inputs():
    import keyboard_transition_fsm as fsm
    from src.tasks.fsm_student import GRUStudent, GRUStudentConfig, GRUStudentPolicy, student_observations

    loaded = fsm.load_full_chain(fsm.parse_args(['--device', 'cpu', '--num-envs', '2']))
    env, wrapped, action, term, rise, down, _ = loaded
    try:
        torch.testing.assert_close(torch.tensor(GRUStudentConfig().default_angles), action.default_angles[0])
        model = GRUStudent(GRUStudentConfig(
            default_angles=tuple(action.default_angles[0].tolist()), action_scale=action.cfg.scale))
        policy = GRUStudentPolicy(model, num_envs=2)
        command = torch.tensor([[.2, 0., 0., 0., 0., .1], [.3, 0., 0., 0., 0., .08]])
        term.submit([0, 1], command, [True, True])
        observation = student_observations(env, command)
        assert observation.shape == (2, 48)
        predicted = policy(observation)
        assert predicted.shape == (2, 12)
        assert torch.isfinite(predicted).all()
        teacher_obs = env.observation_manager.compute()
        _, _, done, _ = wrapped.step(fsm.FullChainPolicy(term, rise, down)(teacher_obs))
        assert not done.any()
        observation = student_observations(env, command)
        # Alter only the teacher's executed command, as a reference override does.
        # The student's command channels must continue to describe user intent.
        saved = term._command.clone()
        term._command[:, 3:] = .5
        torch.testing.assert_close(student_observations(env, command), observation, atol=0., rtol=0.)
        term._command.copy_(saved)
        assert torch.isfinite(policy(observation)).all()
    finally:
        env.close()


def test_keyboard_warm_start_preserves_hidden_and_selects_teacher_per_world(tmp_path):
    import keyboard_transition_fsm as fsm
    from src.tasks.fsm_student import GRUStudent, student_observations

    loaded = fsm.load_full_chain(fsm.parse_args(['--device', 'cpu', '--num-envs', '2']))
    env, wrapped, action, term, rise, down, _ = loaded
    try:
        checkpoint = tmp_path/'student.pt'
        GRUStudent().save(checkpoint)
        inputs = fsm.FullKeyboardInputs(term)
        policy = fsm.FullChainStudentPolicy(env, inputs, checkpoint, warm_start_seconds=.04)
        env.episode_length_buf[:] = torch.tensor([0, 2])
        action.source_teacher.actions = lambda env, ids: torch.full((len(ids), 12), 3.)
        out = policy(wrapped.get_observations())
        assert action.teacher_policy_mask.tolist() == [True, False]
        action.process_actions(out)
        torch.testing.assert_close(action._requested_action[0], torch.full((12,), 3.))
        torch.testing.assert_close(action._requested_action[1], out[1].clamp(-10., 10.))
        hidden = policy.policy.hidden.clone()
        env.episode_length_buf[:] = torch.tensor([2, 3])
        command = torch.tensor(inputs.controller.command).expand(2, -1)
        expected, _ = policy.policy.model(student_observations(env, command), hidden)
        actual = policy(wrapped.get_observations())
        torch.testing.assert_close(actual, expected)
        assert not action.teacher_policy_mask.any()
        wrapped.reset()
        policy(wrapped.get_observations())
        assert action.teacher_policy_mask.all()
    finally:
        env.close()
