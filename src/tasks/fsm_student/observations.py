"""Student proprioception and explicit user intent, without teacher FSM inputs."""
import torch

from src.tasks.pedipulation.constants import JOINT_NAMES


STUDENT_OBSERVATION_DIM = 48


def observation_contract():
    return {
        'dimension': STUDENT_OBSERVATION_DIM,
        'fields': [
            ['angular_velocity_body', 3, .25],
            ['projected_gravity_body', 3, 1.],
            ['user_vx_vy_wz_FR_offset', 6, [2., 2., .25, 1., 1., 1.]],
            ['joint_position_minus_common_default', 12, 1.],
            ['joint_velocity', 12, .05],
            ['previous_applied_common_action', 12, 1.],
        ],
        'joint_names': list(JOINT_NAMES),
        'command_source': 'explicit_user_intent_before_teacher_reference_override',
        'clip': 100.,
    }


def student_observations(env, user_command):
    """Build [world, 48] from sensors and the operator's six-value command.

    Keyboard shortcuts such as R are handled by the operator before this call.
    Do not pass a teacher phase, owner, or motion-reference command here.
    Call once before each control step, when raw_action is the previous target
    actually applied to the actuator in common action coordinates.
    """
    robot = env.scene['robot'].data
    action = env.action_manager.get_term('joint_pos')
    position = robot.joint_pos[:, action.joint_ids]
    count = position.shape[0]
    command = torch.as_tensor(user_command, device=position.device, dtype=position.dtype)
    if command.shape != (count, 6) or not torch.isfinite(command).all():
        raise ValueError('Student requires one finite six-value user command per world')
    observation = torch.cat((
        robot.root_link_ang_vel_b*.25,
        robot.projected_gravity_b,
        command*command.new_tensor((2., 2., .25, 1., 1., 1.)),
        position-action.default_angles,
        robot.joint_vel[:, action.joint_ids]*.05,
        action.raw_action,
    ), dim=-1)
    return observation.clamp(-100., 100.)
