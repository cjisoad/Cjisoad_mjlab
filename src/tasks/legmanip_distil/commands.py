"""Frozen FSM behavior with two immutable, independently clocked references."""
from dataclasses import asdict, dataclass

import torch

from src.tasks.teacher_common.commands import LocoPedipulationTeacherCommand, LocoPedipulationTeacherCommandCfg
from src.tasks.teacher_common.anchor_frame import anchor_basis, to_anchor
from src.tasks.transition_t.commands import TransitionCommand, TransitionCommandCfg
from src.tasks.transition_down_t.commands import DownMotion, align_reference
from src.tasks.loco_pedipulation.mdp.commands import QUAD, HOLD
from .intent import ROOT

QUADRUPED, RISE, BIPED, DOWN = 0, 2, 3, 4
RISE_MOTION = str(ROOT / 'src/tasks/transition_t/data/tripod_to_biped_v3_speed125_train.npz')
DOWN_MOTION = str(ROOT / 'src/assets/motions/go2/landing_v2/biped_to_tripod_v2.npz')


class ReferenceState(TransitionCommand):
    """Reference tensors only; neither sampling nor resetting writes physics."""
    start_from_current_state = DownMotion.start_from_current_state

    def __init__(self, path, env):
        cfg = TransitionCommandCfg(motion_file=path, sampling_mode='start', debug_vis=False)
        super().__init__(cfg, env)
        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset_reference(self, ids, root):
        self.time_steps[ids] = 0
        self.completed[ids] = False
        self.active[ids] = False
        self._last_update_step[ids] = self._env.common_step_counter
        self._reset_step[ids] = self._env.common_step_counter
        self.reset_age_steps[ids] = 0
        self.time_left[ids] = 1.e9
        align_reference(self, ids, root[:, :3], root[:, 3:7])
        position, quaternion = self.robot_anchor_pos_w.clone(), self.robot_anchor_quat_w.clone()
        position[ids], quaternion[ids] = root[:, :3], root[:, 3:7]
        self._refresh_relative(position, quaternion)

    def advance(self):
        self._last_update_step[~self.active] = self._env.common_step_counter
        super()._update_command()


@dataclass(kw_only=True)
class FSMCommandCfg(LocoPedipulationTeacherCommandCfg):
    rise_motion_file: str = RISE_MOTION
    down_motion_file: str = DOWN_MOTION
    rise_trigger_height: float = .35
    down_trigger_height: float = .25
    trigger_seconds: float = .10

    def build(self, env):
        return FSMCommand(self, env)


def teacher_command_cfg():
    from src.tasks.teacher_common.env_cfg import loco_pedipulation_teacher_env_cfg
    source = loco_pedipulation_teacher_env_cfg(play=True).commands['twist']
    cfg = FSMCommandCfg(**asdict(source))
    cfg.resampling_time_range = (1.e9, 1.e9)
    cfg.debug_vis = False
    return cfg


class FSMCommand(LocoPedipulationTeacherCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not 0 < cfg.down_trigger_height < cfg.rise_trigger_height or cfg.trigger_seconds <= 0:
            raise ValueError('Invalid FSM height thresholds or trigger duration')
        self.owner = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rise_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.down_elapsed = torch.zeros_like(self.rise_elapsed)
        self.returning = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fr_engaged = self.returning.clone()
        self.transition_elapsed = torch.zeros_like(self.rise_elapsed)
        self._last_advance_step = torch.full_like(self.owner, -1)
        self.references = {RISE: ReferenceState(cfg.rise_motion_file, env),
                           DOWN: ReferenceState(cfg.down_motion_file, env)}

    def _resample_command(self, ids):
        intent = self._env.command_manager.get_term('intent')
        self.manual[ids] = True
        self.owner[ids] = torch.where(intent.starts_biped[ids], BIPED, QUADRUPED)
        self.rise_elapsed[ids] = 0.
        self.down_elapsed[ids] = 0.
        self.transition_elapsed[ids] = 0.
        self.returning[ids] = False
        self.fr_engaged[ids] = intent.starts_biped[ids]
        self._last_advance_step[ids] = self._env.common_step_counter
        self._command[ids] = intent.command[ids]
        self.requested_velocity[ids] = intent.command[ids, :3]
        self.requested_offset[ids] = intent.command[ids, 3:]
        self.enabled[ids] = (intent.command[ids, 3:] != 0).any(-1)
        biped = ids[intent.starts_biped[ids]]
        self.phase[biped] = HOLD
        self.blend[biped] = 1.
        for reference in self.references.values():
            reference.reset_reference(ids, intent.reset_root[ids])

    def start_transition(self, ids, owner):
        if owner not in self.references:
            raise ValueError('Only rise and down own motion references')
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        for reference in self.references.values():
            reference.active[ids] = False
        reference = self.references[owner]
        reference.start_from_current_state(ids)
        reference.active[ids] = True
        self.owner[ids] = owner
        self.transition_elapsed[ids] = 0.
        self.rise_elapsed[ids] = 0.
        self.down_elapsed[ids] = 0.

    def reference_offset(self, owner):
        motion = self.references[owner]
        fr = motion.cfg.body_names.index('FR_foot')
        basis = anchor_basis(motion.anchor_quat_w, self.robot.data.gravity_vec_w)
        return to_anchor(basis, motion.body_pos_w[:, fr] - motion.anchor_pos_w) - self.zero

    def _update_command(self):
        for reference in self.references.values():
            reference.advance()
        intent = self._env.command_manager.get_term('intent')
        # Canonical six-zero intent carries R. The original R latch bypasses
        # the height hold and preserves the biped target until DOWN takes over.
        returning = (self.owner == BIPED) & (intent.command == 0).all(-1)
        self.returning |= returning
        endpoint = (self.owner == QUADRUPED) | (self.owner == BIPED)
        ids = (endpoint & ~returning).nonzero().flatten()
        values = intent.command[ids]
        super().set_command(ids, velocity=values[:, :3], target_offset=values[:, 3:])
        biped_ids = ids[self.owner[ids] == BIPED]
        self._command[biped_ids] = intent.command[biped_ids]
        dz, dt = intent.command[:, 5], self._env.step_dt
        advancing = self._last_advance_step != self._env.common_step_counter
        self._last_advance_step[:] = self._env.common_step_counter
        self.rise_elapsed = torch.where((self.owner == QUADRUPED) & (dz >= self.cfg.rise_trigger_height - 1.e-9),
                                        self.rise_elapsed + advancing * dt, 0.)
        self.down_elapsed = torch.where((self.owner == BIPED) & (dz <= self.cfg.down_trigger_height + 1.e-9),
                                        self.down_elapsed + advancing * dt, 0.)
        for owner, elapsed in ((RISE, self.rise_elapsed), (DOWN, self.down_elapsed)):
            trigger = elapsed >= self.cfg.trigger_seconds - 1.e-9
            if owner == DOWN:
                trigger |= returning
            ids = trigger.nonzero().flatten()
            if len(ids):
                self.start_transition(ids, owner)
        for owner, destination in ((RISE, BIPED), (DOWN, QUADRUPED)):
            reference = self.references[owner]
            ids = ((self.owner == owner) & reference.completed).nonzero().flatten()
            if len(ids):
                self._command[ids, :3] = 0.
                self._command[ids, 3:] = self.reference_offset(owner)[ids]
                if owner == DOWN:
                    self._command[ids[self.returning[ids]]] = 0.
                    self.returning[ids] = False
                self.requested_velocity[ids] = 0.
                self.requested_offset[ids] = self._command[ids, 3:]
                self.owner[ids] = destination
                reference.active[ids] = False
        preserved = self.command.clone()
        super()._update_command()
        controlled = self.owner != QUADRUPED
        self._command[controlled] = preserved[controlled]
        self.phase[controlled] = HOLD
        self.blend[controlled] = 1.
        for owner in (RISE, DOWN):
            ids = (self.owner == owner).nonzero().flatten()
            self._command[ids, :3] = 0.
            self._command[ids, 3:] = self.reference_offset(owner)[ids]
        self.transition_elapsed += ((self.owner == RISE) | (self.owner == DOWN)) * advancing * dt
        self.fr_engaged[(self.owner == QUADRUPED) & (self.phase == QUAD)] = False
        self.fr_engaged |= controlled | ((self.phase == HOLD) & (self.blend >= 1.))
