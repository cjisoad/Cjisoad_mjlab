"""Deployment observation and action contracts shared by training and playback."""
import torch

from src.tasks.pedipulation.constants import JOINT_NAMES


def observation_contract():
    return {
        'dimension': 48,
        'fields': [
            ['angular_velocity_body', 3, .25],
            ['projected_gravity_body', 3, 1.],
            ['user_vx_vy_wz_FR_offset', 6, [2., 2., .25, 1., 1., 1.]],
            ['joint_position_minus_common_default', 12, 1.],
            ['joint_velocity', 12, .05],
            ['previous_applied_common_action', 12, 1.],
        ],
        'joint_names': list(JOINT_NAMES),
        'command_source': 'canonical_user_intent',
        'return_command': [0.] * 6,
        'clip': 100.,
    }


def action_contract(env):
    action = env.action_manager.get_term('joint_pos')
    return {
        'dimension': 12,
        'joint_names': list(JOINT_NAMES),
        'default_angles': action.default_angles[0].detach().cpu().tolist(),
        'action_scale': action.cfg.scale,
        'control_dt': env.step_dt,
        'label': 'teacher_native_common_action',
        'execution': 'zero_order_hold_direct_pd',
        'target_limiter': None,
        'delay': False,
        'action_clip': 100.,
    }


def student_observations(env):
    robot = env.scene['robot'].data
    action = env.action_manager.get_term('joint_pos')
    command = env.command_manager.get_term('intent').command
    obs = torch.cat((
        robot.root_link_ang_vel_b * .25,
        robot.projected_gravity_b,
        command * command.new_tensor((2., 2., .25, 1., 1., 1.)),
        robot.joint_pos[:, action.joint_ids] - action.default_angles,
        robot.joint_vel[:, action.joint_ids] * .05,
        action.raw_action,
    ), dim=-1)
    return obs.clamp(-100., 100.)
