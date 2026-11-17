"""Substep actuation logging and pre-reset physical transition capture."""
from dataclasses import dataclass
import torch
from mjlab.managers import TerminationTermCfg
from ..actions import DistilAction, DistilActionCfg
from ..env_cfg import legmanip_distil_env_cfg, physical_failure
from .sampling import CoverageIntentCfg
from .physics import state_from_env


class RecordedAction(DistilAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.executed_targets = torch.zeros(self.num_envs, env.cfg.decimation, 12, device=self.device)

    def apply_actions(self):
        super().apply_actions()
        self.executed_targets[:, self._substep - 1].copy_(self.executed_position)


@dataclass(kw_only=True)
class RecordedActionCfg(DistilActionCfg):
    def build(self, env):
        return RecordedAction(self, env)


def capture_transition(env):
    action = env.action_manager.get_term('joint_pos')
    force = env.scene['feet_ground_contact'].data.force
    contacts = force.reshape(env.num_envs, 4, -1, 3).sum(2).norm(dim=-1) > 5.
    base_force = env.scene['base_ground_contact'].data.force
    collision = torch.zeros(env.num_envs, 1, device=env.device)
    if base_force is not None:
        collision[:, 0] = (base_force.reshape(env.num_envs, -1, 3).norm(dim=-1).amax(-1) > 10.).float()
    env.supermimic_terminal = {
        'next_state': state_from_env(env).clone(),
        'targets': action.executed_targets.clone(),
        'contacts': contacts.float().clone(), 'collision': collision,
    }
    intent = env.command_manager.get_term('intent')
    if isinstance(intent.cfg, CoverageIntentCfg):
        intent.record_frames()
        intent.record_failure(physical_failure(env))
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def supermimic_env_cfg(play=False):
    cfg = legmanip_distil_env_cfg(play)
    if not play:
        cfg.commands['intent'] = CoverageIntentCfg()
        cfg.actions = {'joint_pos': RecordedActionCfg(entity_name='robot', scale=.25)}
        cfg.terminations = {'supermimic_capture': TerminationTermCfg(func=capture_transition), **cfg.terminations}
    return cfg
