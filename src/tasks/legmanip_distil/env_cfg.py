"""Independent distillation and deployment environments with matched physics."""
import torch
from mjlab.envs import mdp
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg

from src.tasks.transition_t.env_cfg import transition_env_cfg
from .actions import DistilActionCfg
from .intent import IntentCommandCfg
from .observations import student_observations


def physical_failure(env):
    robot = env.scene['robot'].data
    values = (robot.joint_pos, robot.joint_vel, robot.root_link_pos_w,
              robot.root_link_quat_w, robot.root_link_lin_vel_w,
              robot.root_link_ang_vel_w, robot.body_link_pos_w,
              robot.body_link_quat_w, robot.body_link_lin_vel_w,
              robot.body_link_ang_vel_w, robot.projected_gravity_b)
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for value in values:
        failed |= ~torch.isfinite(value).flatten(1).all(-1)
    failed |= robot.root_link_quat_w.norm(dim=-1) < 1.e-6
    force = env.scene['base_ground_contact'].data.force
    if force is not None:
        failed |= force.norm(dim=-1).reshape(env.num_envs, -1).amax(-1) > 10.
    return failed


def scenario_finished(env):
    term = env.command_manager.get_term('intent')
    return term.finished | term.timed_out


def legmanip_distil_env_cfg(play=False):
    # This builds only config objects; it does not load any motion or teacher.
    cfg = transition_env_cfg(play=True)
    cfg.scene.num_envs = 1 if play else 256
    cfg.seed = 42
    cfg.actions = {'joint_pos': DistilActionCfg(entity_name='robot', scale=.25)}
    cfg.commands = {'intent': IntentCommandCfg(automatic=not play)}
    if not play:
        from .commands import teacher_command_cfg
        cfg.commands['twist'] = teacher_command_cfg()
    cfg.observations = {'student': ObservationGroupCfg(
        terms={'policy': ObservationTermCfg(func=student_observations)},
        concatenate_terms=True, enable_corruption=False)}
    cfg.rewards = {}
    cfg.metrics = {}
    cfg.curriculum = {}
    cfg.events = {}
    cfg.terminations = {
        'physical_failure': TerminationTermCfg(func=physical_failure),
        'time_out': TerminationTermCfg(func=mdp.time_out, time_out=True),
        'scenario_finished': TerminationTermCfg(func=scenario_finished, time_out=True),
    }
    cfg.episode_length_s = 60.
    return cfg
