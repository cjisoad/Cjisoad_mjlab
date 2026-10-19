"""Canonical user commands and independent scripted training episodes."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from mjlab.managers import CommandTerm, CommandTermCfg

ROOT = Path(__file__).resolve().parents[3]

# (stage, minimum seconds). Transition waits can be extended by the teacher
# completion gate during training, never by a global batch condition.
SCENARIOS = (
    (('settle', .6), ('quad_walk', 3.), ('quad_stand', 1.5), ('lift', 1.8),
     ('tripod_walk', 3.), ('tripod_operation', 3.), ('return', 2.5)),
    (('biped_settle', 1.), ('biped_walk', 3.), ('biped_operation', 3.),
     ('down', 5.), ('return', 2.5)),
    (('settle', .6), ('lift', 1.9), ('high', 4.), ('rise', 8.),
     ('biped_settle', 1.), ('biped_walk', 3.), ('biped_operation', 3.),
     ('down', 5.), ('return', 2.5)),
)
STAGES = tuple(dict.fromkeys(name for scenario in SCENARIOS for name, _ in scenario))


@dataclass(kw_only=True)
class IntentCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.e9, 1.e9)
    automatic: bool = True
    scenario: Literal['mixed', 'quad', 'biped', 'cycle'] = 'mixed'
    initial_state: Literal['quadruped', 'biped'] = 'quadruped'
    entry_split: Literal['train', 'validation', 'final'] = 'train'
    entry_bank_file: str = str(ROOT / 'src/assets/motions/go2/landing_v2/biped_entry_v2.npz')

    def build(self, env):
        return IntentCommand(self, env)


class IntentCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene['robot']
        self._command = torch.zeros(self.num_envs, 6, device=self.device)
        self.scenario = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stage = torch.zeros_like(self.scenario)
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.timed_out = self.finished.clone()
        self.starts_biped = self.finished.clone()
        self.twist = torch.zeros(self.num_envs, 3, device=self.device)
        self.low_target = torch.zeros_like(self.twist)
        self.low_goal = torch.zeros_like(self.twist)
        self.high_target = torch.zeros_like(self.twist)
        self._last_step = torch.full_like(self.scenario, -1)
        self.entry_bank = None
        self.entry_indices = torch.full_like(self.scenario, -1)
        self.reset_root = self.robot.data.default_root_state.clone()
        self.stage_ids = torch.zeros(3, max(map(len, SCENARIOS)), dtype=torch.long, device=self.device)
        self.stage_seconds = torch.zeros_like(self.stage_ids, dtype=torch.float32)
        self.stage_counts = torch.tensor(list(map(len, SCENARIOS)), device=self.device)
        for i, scenario in enumerate(SCENARIOS):
            for j, (name, seconds) in enumerate(scenario):
                self.stage_ids[i, j] = STAGES.index(name)
                self.stage_seconds[i, j] = seconds
        # Workspace is kinematics only: no teacher weights or motion archives.
        from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace, LOW, HIGH
        workspace = build_foot_workspace()
        self.nominal = torch.tensor(workspace.nominal_biped_offset, device=self.device, dtype=torch.float32)
        self.foot_zero = torch.tensor(workspace.zero, device=self.device, dtype=torch.float32)
        self.low_bank = torch.tensor(workspace.offsets[workspace.segment == LOW], device=self.device, dtype=torch.float32)
        self.high_bank = torch.tensor(workspace.offsets[workspace.segment == HIGH], device=self.device, dtype=torch.float32)

    @property
    def command(self):
        return self._command

    @property
    def stage_name_ids(self):
        return self.stage_ids[self.scenario, self.stage]

    def set_command(self, ids, values):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(-1)
        values = torch.as_tensor(values, device=self.device, dtype=self._command.dtype)
        if values.shape != (len(ids), 6) or not torch.isfinite(values).all():
            raise ValueError('Intent requires finite [selected_envs, 6] commands')
        self._command[ids] = values
        if hasattr(self._env, 'observation_manager'):
            self._env.observation_manager._obs_buffer = None

    def return_to_quadruped(self, ids):
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).reshape(-1)
        self.set_command(ids, torch.zeros(len(ids), 6, device=self.device))

    def _load_bank(self):
        if self.entry_bank is None:
            from src.tasks.transition_down_v2.data import LandingBank, MOTION_SHA256
            self.entry_bank = LandingBank(self.cfg.entry_bank_file, motion_sha256=MOTION_SHA256,
                                         device=self.device)
        return self.entry_bank

    def _resample_command(self, ids):
        if self.cfg.automatic:
            if self.cfg.scenario == 'mixed':
                draw = torch.rand(len(ids), device=self.device)
                self.scenario[ids] = torch.where(draw < .25, 0, torch.where(draw < .5, 1, 2))
            else:
                self.scenario[ids] = ('quad', 'biped', 'cycle').index(self.cfg.scenario)
            self.starts_biped[ids] = self.scenario[ids] == 1
        else:
            self.starts_biped[ids] = self.cfg.initial_state == 'biped'
        self.stage[ids] = 0
        self.elapsed[ids] = 0.
        self.finished[ids] = False
        self.timed_out[ids] = False
        self._last_step[ids] = self._env.common_step_counter
        self._command[ids] = 0.
        self.entry_indices[ids] = -1
        action = self._env.action_manager.get_term('joint_pos')
        root = self.robot.data.default_root_state[ids].clone()
        root[:, :3] += self._env.scene.env_origins[ids]
        root[:, 7:] = 0.
        joints = self.robot.data.default_joint_pos[ids].clone()
        self.robot.write_root_state_to_sim(root, env_ids=ids)
        self.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids)
        self.robot.clear_state(env_ids=ids)
        self.reset_root[ids] = root
        action.executed_position[ids] = joints[:, action.joint_ids]
        biped = ids[self.starts_biped[ids]]
        if len(biped):
            bank = self._load_bank()
            selected = bank.sample(len(biped), split=self.cfg.entry_split, max_speed=.25, max_fr_error=.08)
            self.entry_indices[biped] = selected
            root = bank.data['root_state'][selected].clone()
            root[:, :3] += self._env.scene.env_origins[biped]
            self.robot.write_root_state_to_sim(root, env_ids=biped)
            self.robot.write_joint_state_to_sim(bank.data['joint_pos'][selected],
                                               bank.data['joint_vel'][selected], env_ids=biped)
            self.robot.clear_state(env_ids=biped)
            self.reset_root[biped] = root
            action._raw_action[biped] = bank.data['raw_action'][selected]
            action.previous_action[biped] = bank.data['previous_action'][selected]
            action.last_request[biped] = bank.data['raw_action'][selected]
            action.executed_position[biped] = bank.data['limiter_position'][selected]
            action.executed_velocity[biped] = bank.data['limiter_velocity'][selected]
            self._env.action_manager.action[biped] = action.raw_action[biped]
            self._command[biped, 3:] = self.nominal
        if self.cfg.automatic:
            for axis, bounds in enumerate(((.08, .22), (-.035, .035), (-.15, .15))):
                self.twist[ids, axis] = torch.empty(len(ids), device=self.device).uniform_(*bounds)
            self.low_target[ids] = self.low_bank[torch.randint(len(self.low_bank), (len(ids),), device=self.device)]
            self.low_goal[ids] = self.low_bank[torch.randint(len(self.low_bank), (len(ids),), device=self.device)]
            self.high_target[ids] = self.high_bank[torch.randint(len(self.high_bank), (len(ids),), device=self.device)]
            self._write_script_commands(ids)

    def _write_script_commands(self, ids):
        stage = self.stage_name_ids[ids]
        command = torch.zeros(len(ids), 6, device=self.device)
        elapsed = self.elapsed[ids]
        for name in ('lift', 'tripod_walk', 'tripod_operation', 'high', 'rise'):
            mask = stage == STAGES.index(name)
            command[mask, 3:] = self.low_target[ids[mask]]
        for name in ('biped_settle', 'biped_walk', 'biped_operation', 'down'):
            mask = stage == STAGES.index(name)
            command[mask, 3:] = self.nominal
        command[stage == STAGES.index('high'), 5] = .32
        command[stage == STAGES.index('rise'), 5] = .36
        command[stage == STAGES.index('down'), 5] = .24
        for name, scale in (('quad_walk', 1.), ('tripod_walk', 1.), ('biped_walk', .6)):
            mask = stage == STAGES.index(name)
            command[mask, :3] = scale * self.twist[ids[mask]]
        # Interpolate between feasible FK targets, retaining the source's short
        # continuous retargeting behavior without unconstrained random goals.
        mask = stage == STAGES.index('tripod_operation')
        weight = .5 * (1. - torch.cos(elapsed[mask, None]))
        command[mask, 3:] = ((1. - weight) * self.low_target[ids[mask]]
                            + weight * self.low_goal[ids[mask]])
        mask = stage == STAGES.index('biped_operation')
        weight = .5 * (1. - torch.cos(elapsed[mask, None]))
        command[mask, 3:] = (1. - weight) * self.nominal + weight * self.high_target[ids[mask]]
        self._command[ids] = command

    def _update_command(self):
        if not self.cfg.automatic:
            return
        advancing = self._last_step < self._env.common_step_counter
        self._last_step[:] = self._env.common_step_counter
        self.elapsed += advancing * self._env.step_dt
        minimum = self.stage_seconds[self.scenario, self.stage]
        ready = self.elapsed >= minimum
        # Training teacher reference completion may extend its own episode,
        # without synchronizing other worlds or rewriting external intent.
        if 'twist' in self._env.cfg.commands:
            teacher = self._env.command_manager.get_term('twist')
            names = self.stage_name_ids
            ready &= ((names != STAGES.index('rise')) | (teacher.owner == 3))
            ready &= ((names != STAGES.index('down')) | (teacher.owner == 0))
            ready &= ((names != STAGES.index('return')) | ((teacher.owner == 0) & (teacher.phase == 0)))
        self.timed_out |= self.elapsed > minimum + 5.
        next_stage = self.stage + ready.long()
        self.finished |= next_stage >= self.stage_counts[self.scenario]
        self.stage = torch.minimum(next_stage, self.stage_counts[self.scenario] - 1)
        self.elapsed[ready] = 0.
        self._write_script_commands(torch.arange(self.num_envs, device=self.device))

    def _update_metrics(self):
        pass
