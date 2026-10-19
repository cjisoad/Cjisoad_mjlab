"""Student-only environment loading and recurrent state management."""
import torch
from mjlab.rl import RslRlVecEnvWrapper

from .observations import action_contract, observation_contract


def validate_contract(metadata, env):
    for key, expected in (('observation_contract', observation_contract()),
                          ('action_contract', action_contract(env))):
        if metadata.get(key) != expected:
            raise ValueError(f'Student {key} does not match the deployment environment')


class StudentVecEnv(RslRlVecEnvWrapper):
    """Reset GRU memory exactly when physics resets, including viewer resets."""
    def __init__(self, env, student):
        self.student = student
        super().__init__(env, clip_actions=None)

    @torch.inference_mode()
    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        self.student.reset(dones.bool())
        return observations, rewards, dones, extras

    @torch.inference_mode()
    def reset(self):
        result = super().reset()
        self.student.reset()
        return result


def create_student_world(checkpoint, *, device='cpu', num_envs=1,
                         scenario=None, split='validation', seed=42):
    from mjlab.envs import ManagerBasedRlEnv
    from .env_cfg import legmanip_distil_env_cfg
    from .rl import load_student

    if num_envs < 1:
        raise ValueError('num_envs must be positive')
    student, metadata = load_student(checkpoint, device=device, num_envs=num_envs)
    cfg = legmanip_distil_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.commands['intent'].entry_split = split
    if scenario is not None:
        cfg.commands['intent'].automatic = True
        cfg.commands['intent'].scenario = scenario
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        validate_contract(metadata, env)
        wrapped = StudentVecEnv(env, student)
        return env, wrapped, student, metadata
    except Exception:
        env.close()
        raise


def update_keyboard_intent(intent, controller, keys, dt, *, focused=True, paused=False):
    """R is always six zeros, even while a velocity key is pressed."""
    if focused and not paused and 'r' in keys:
        controller.reset()
        intent.return_to_quadruped([0])
    else:
        controller.update(keys, dt, focused=focused, paused=paused)
        intent.set_command([0], [controller.command])


@torch.inference_mode()
def run_headless(wrapped, student, steps):
    if steps < 1:
        raise ValueError('steps must be positive')
    observations = wrapped.get_observations()
    resets, maximum = 0, 0.
    for _ in range(steps):
        actions = student(observations)
        maximum = max(maximum, actions.abs().max().item())
        observations, _, dones, _ = wrapped.step(actions)
        resets += int(dones.sum())
    return {'steps': steps, 'environments': wrapped.num_envs,
            'episode_resets': resets, 'max_abs_requested_action': maximum,
            'teacher_loaded': False}
