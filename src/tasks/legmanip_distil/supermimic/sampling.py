"""Six-task curriculum balanced by achieved real control frames.

Only intent is sampled here. Physical resets and the biped entry witness are
inherited from IntentCommand, including the action limiter's complete state.
"""
from dataclasses import dataclass
from functools import lru_cache

import torch

from ..intent import IntentCommand, IntentCommandCfg, SCENARIOS, STAGES

TASK_NAMES = ('quad_walk', 'biped_walk', 'tripod_operation',
              'biped_operation', 'combined', 'transition')
TARGET_FRACTIONS = (.20, .20, .20, .15, .15, .10)


@lru_cache(maxsize=8)
def _validated_high_paths(offsets, nominal, zero):
    """Keep hub-to-target Cartesian paths with sampled IK/collision witnesses.

    Cached across live and shadow environments. This certifies the nominal
    kinematic model at <= 1cm path samples, not the dynamic robot's clearance.
    Low paths already have this validation in build_foot_workspace().
    """
    import re
    import mujoco
    import numpy as np
    from scipy.optimize import least_squares
    from src.tasks.pedipulation.constants import DESIRED_ANGLES, JOINT_NAMES
    from src.tasks.pedipulation.robot import get_stand_spec
    from src.tasks.leg_manip.mdp.foot_workspace import REARED_ROTATION, MOVING_GEOMS
    model = get_stand_spec().compile()
    data = mujoco.MjData(model)
    addresses = [int(model.joint(name).qposadr[0]) for name in JOINT_NAMES]
    fr_joints = [model.joint(name).id for name in JOINT_NAMES[3:6]]
    limits = model.jnt_range[fr_joints]
    dofs = model.jnt_dofadr[fr_joints]
    site, base = model.site('FR').id, model.body('base_link').id
    moving = [model.geom(name).id for name in MOVING_GEOMS]
    obstacles = [i for i in range(model.ngeom) if re.fullmatch(
        r'(base_link|Head_upper|Head_lower|FL_.*)_collision', model.geom(i).name)]
    jacobian = np.zeros((3, model.nv))
    hub, zero = np.asarray(nominal), np.asarray(zero)
    kept = []
    for offset in offsets:
        data.qpos[addresses] = DESIRED_ANGLES
        solution = np.asarray(DESIRED_ANGLES[3:6]).copy()
        offset = np.asarray(offset)
        samples = max(9, int(np.ceil(np.linalg.norm(offset - hub) / .01)) + 1)
        valid = True
        for fraction in np.linspace(0., 1., samples):
            target = REARED_ROTATION.T @ (zero + hub * (1. - fraction) + offset * fraction)
            def residual(angles):
                data.qpos[addresses[3:6]] = angles
                mujoco.mj_forward(model, data)
                return data.site_xpos[site] - data.xpos[base] - target
            def jac(angles):
                residual(angles)
                mujoco.mj_jacSite(model, data, jacobian, None, site)
                return jacobian[:, dofs]
            result = least_squares(residual, solution, jac=jac,
                bounds=(limits[:, 0] + .05, limits[:, 1] - .05), max_nfev=40)
            if np.linalg.norm(result.fun) > 1.e-4:
                valid = False
                break
            solution = result.x
            residual(solution)
            if any(mujoco.mj_geomDistance(model, data, moved, obstacle, .02, None) < .01
                   for moved in moving for obstacle in obstacles):
                valid = False
                break
        if valid:
            kept.append(tuple(float(v) for v in offset))
    return tuple(kept)


@dataclass(kw_only=True)
class CoverageIntentCfg(IntentCommandCfg):
    coverage_enabled: bool = True
    curriculum_updates: int = 10000
    easy_fraction: float = .20

    def build(self, env):
        return CoverageIntent(self, env)


class CoverageIntent(IntentCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.curriculum_updates <= 0 or not 0 <= cfg.easy_fraction <= 1:
            raise ValueError('Require positive curriculum duration and easy fraction in [0, 1]')
        self.task_id = torch.zeros_like(self.scenario)
        self.training_progress = torch.zeros((), dtype=torch.long, device=self.device)
        self.target_fractions = torch.tensor(TARGET_FRACTIONS, device=self.device)
        self.allocated_episodes = torch.zeros(6, dtype=torch.long, device=self.device)
        self.actual_frames = torch.zeros_like(self.allocated_episodes)
        self.stage_frames = torch.zeros(len(STAGES), dtype=torch.long, device=self.device)
        self.owner_frames = torch.zeros(5, dtype=torch.long, device=self.device)
        self.failure_stage_counts = torch.zeros(6, len(STAGES), dtype=torch.long, device=self.device)
        self.velocity_frames = torch.zeros(7, dtype=torch.long, device=self.device)
        self.workspace_frames = torch.zeros(2, 2, 3, dtype=torch.long, device=self.device)
        self._recorded_step = torch.full((), -1, dtype=torch.long, device=self.device)
        self._failure_step = self._recorded_step.clone()
        self.task_elapsed = torch.zeros_like(self.elapsed)
        self.task_duration = torch.zeros_like(self.elapsed)
        self.command_elapsed = torch.zeros_like(self.elapsed)
        self.command_duration = torch.zeros_like(self.elapsed)
        self.easy_episode = torch.zeros_like(self.finished)
        self.foot_elapsed = torch.zeros_like(self.elapsed)
        self.foot_hold = torch.zeros_like(self.elapsed)
        self.foot_motion = torch.zeros_like(self.elapsed)
        self.foot_source = torch.zeros_like(self.twist)
        self.foot_goal = torch.zeros_like(self.twist)
        self.returning = torch.zeros_like(self.finished)
        self.r_return_episode = torch.zeros_like(self.finished)
        self.return_outcome_recorded = torch.zeros_like(self.finished)
        self.return_success_counts = torch.zeros(6, dtype=torch.long, device=self.device)
        self.return_timeout_counts = torch.zeros_like(self.return_success_counts)
        self.return_source = torch.zeros_like(self.twist)
        # Exclude the FSM's transition trigger boundary from each direct mode.
        # Every retained endpoint is a real FK/collision-checked bank witness.
        self.direct_low_bank = self.low_bank[self.low_bank[:, 2] < .35 - 1.e-5]
        self.direct_high_bank = self.high_bank[self.high_bank[:, 2] > .25 + 1.e-5]
        certified = _validated_high_paths(tuple(map(tuple, self.direct_high_bank.cpu().tolist())),
                                          tuple(self.nominal.cpu().tolist()), tuple(self.foot_zero.cpu().tolist()))
        self.direct_high_bank = torch.tensor(certified, device=self.device, dtype=torch.float32).reshape(-1, 3)
        if not len(self.direct_low_bank) or not len(self.direct_high_bank):
            raise ValueError('No feasible non-transition foot targets in workspace')
        for name in ('low', 'high'):
            bank = getattr(self, 'direct_' + name + '_bank')
            xy_lo, xy_hi = bank[:, :2].amin(0), bank[:, :2].amax(0)
            xy_scale = (xy_hi - xy_lo).clamp_min(1.e-6)
            edge = (((bank[:, :2] - xy_lo) / xy_scale * 2 - 1).abs().amax(-1) >= .65)
            cuts = torch.quantile(bank[:, 2], torch.tensor([1/3, 2/3], device=self.device))
            height = torch.bucketize(bank[:, 2].contiguous(), cuts)
            center = self.nominal if name == 'high' else torch.tensor([0., 0., .10], device=self.device)
            distance = (bank - center).norm(dim=-1)
            table = torch.zeros(5, 6, len(bank), dtype=torch.long, device=self.device)
            sizes = torch.zeros(5, 6, dtype=torch.long, device=self.device)
            for tier in range(5):
                threshold = torch.quantile(distance, .2 + .2 * tier)
                for e in range(2):
                    for z in range(3):
                        pool = ((edge == e) & (height == z) & (distance <= threshold + 1.e-6)).nonzero().flatten()
                        table[tier, e * 3 + z, :len(pool)] = pool
                        sizes[tier, e * 3 + z] = len(pool)
            setattr(self, '_' + name + '_strata', table)
            setattr(self, '_' + name + '_sizes', sizes)
            setattr(self, '_' + name + '_xy_lo', xy_lo)
            setattr(self, '_' + name + '_xy_scale', xy_scale)
            setattr(self, '_' + name + '_z_cuts', cuts)

    @property
    def coverage_active(self):
        return self.cfg.coverage_enabled and self.cfg.automatic and self.cfg.scenario == 'mixed'

    def set_training_progress(self, update: int):
        self.training_progress.fill_(max(0, int(update)))

    def allocation_weights(self):
        # Equal prior expected episode lengths are corrected for the longer
        # transition reference. Actual deficits then compensate early failures.
        expected = torch.full((6,), 15., device=self.device)
        expected[-1] = sum(seconds for _, seconds in SCENARIOS[2])
        prior = 1000.
        desired = self.target_fractions * (self.actual_frames.sum() + prior)
        achieved = self.actual_frames + self.target_fractions * prior
        compensation = (desired / achieved).clamp(.25, 4.)
        expected_frames = expected / self._env.step_dt
        prior_episodes = 10. + self.num_envs * self.target_fractions
        realized = ((self.actual_frames + expected_frames * prior_episodes) /
                    (self.allocated_episodes + prior_episodes))
        realized = torch.maximum(realized, expected_frames * .05)
        realized = torch.minimum(realized, expected_frames * 2.)
        weights = self.target_fractions / realized * compensation
        return weights / weights.sum()

    def _sample_workspace(self, ids, high):
        name = 'high' if high else 'low'
        bank = getattr(self, 'direct_' + name + '_bank')
        table, sizes = getattr(self, '_' + name + '_strata'), getattr(self, '_' + name + '_sizes')
        if not len(ids):
            return bank[:0]
        tier = (4 * self.training_progress.float() / self.cfg.curriculum_updates).clamp(0., 4.).long()
        tiers = torch.where(self.easy_episode[ids], 0, tier)
        available = sizes[tiers] > 0
        strata = torch.multinomial(available.float(), 1).squeeze(-1)
        position = (torch.rand(len(ids), device=self.device) * sizes[tiers, strata]).long()
        return bank[table[tiers, strata, position]]

    def _sample_twist(self, ids):
        progress = (self.training_progress.float() / self.cfg.curriculum_updates).clamp(0., 1.)
        extent = torch.where(self.easy_episode[ids], 0., progress)
        vx_lo, vx_hi = -.03 - .17 * extent, .15 + .45 * extent
        wz_hi = .20 + .80 * extent
        values = torch.zeros(len(ids), 3, device=self.device)
        values[:, 0] = vx_lo + torch.rand(len(ids), device=self.device) * (vx_hi - vx_lo)
        values[:, 2] = (torch.rand(len(ids), device=self.device) * 2 - 1) * wz_hi
        # Explicit stops, reverse, left, right, and straight commands. A true
        # biped stop retains its nonzero foot target, preserving the R latch.
        category = torch.randint(8, (len(ids),), device=self.device)
        values[category == 0] = 0.
        reverse = category == 1
        values[reverse, 0] = vx_lo[reverse] * (.25 + .75 * torch.rand(int(reverse.sum()), device=self.device))
        values[category == 2, 2] = values[category == 2, 2].abs()
        values[category == 3, 2] = -values[category == 3, 2].abs()
        values[category == 4, 2] = 0.
        inplace = (category == 6) | (category == 7)
        values[inplace, 0] = 0.
        magnitude = (.25 + .75 * torch.rand(len(ids), device=self.device)) * wz_hi
        values[category == 6, 2] = magnitude[category == 6]
        values[category == 7, 2] = -magnitude[category == 7]
        tripod = (self.task_id[ids] == 4) & ~self.starts_biped[ids]
        # The coordinator clamps active tripod motion to these physical limits.
        # Match raw intent to its supported supervision/reward target.
        limits = (.35, .15, .4)
        if 'twist' in self._env.cfg.commands:
            limits = getattr(self._env.cfg.commands['twist'], 'tripod_velocity_limits', limits)
        limits = torch.as_tensor(limits, device=self.device)
        values[tripod] = values[tripod].clamp(-limits, limits)
        self.twist[ids] = values
        self.command_elapsed[ids] = 0.
        self.command_duration[ids] = torch.empty(len(ids), device=self.device).uniform_(1., 4.)

    def _resample_command(self, ids):
        if not self.coverage_active:
            super()._resample_command(ids)
            self._label_legacy_tasks(ids)
            return
        self.task_id[ids] = torch.multinomial(self.allocation_weights(), len(ids), replacement=True)
        self.allocated_episodes += torch.bincount(self.task_id[ids], minlength=6)
        task = self.task_id[ids]
        starts = (task == 1) | (task == 3) | ((task == 4) & (torch.rand(len(ids), device=self.device) < .5))
        # Call the inherited reset independently for each entry family. This
        # retains its full robot/action/bank reset and leaves teacher reset order intact.
        original = self.cfg.scenario
        try:
            for scenario, selected in (('quad', ids[~starts & (task != 5)]),
                                       ('biped', ids[starts]), ('cycle', ids[task == 5])):
                if len(selected):
                    self.cfg.scenario = scenario
                    super()._resample_command(selected)
        finally:
            self.cfg.scenario = original
        self.twist[ids, 1] = 0.
        self.returning[ids] = False
        self.r_return_episode[ids] = self.starts_biped[ids] & (torch.rand(len(ids), device=self.device) < .5)
        self.return_outcome_recorded[ids] = False
        self.return_source[ids] = 0.
        self.task_elapsed[ids] = 0.
        self.task_duration[ids] = torch.empty(len(ids), device=self.device).uniform_(10., 20.)
        self.easy_episode[ids] = torch.rand(len(ids), device=self.device) < self.cfg.easy_fraction
        direct = ids[task != 5]
        self._sample_twist(direct)
        for high in (False, True):
            selected = direct[self.starts_biped[direct] == high]
            self.foot_source[selected] = self._sample_workspace(selected, high)
            self.foot_goal[selected] = self._sample_workspace(selected, high)
        self.foot_elapsed[direct] = 0.
        self.foot_hold[direct] = torch.empty(len(direct), device=self.device).uniform_(2., 5.)
        self.foot_motion[direct] = torch.empty(len(direct), device=self.device).uniform_(1., 2.)
        for task_id, scenario, stage in ((0, 0, 'quad_walk'), (1, 1, 'biped_walk'),
                                         (2, 0, 'tripod_operation'), (3, 1, 'biped_operation')):
            selected = ids[task == task_id]
            self.scenario[selected] = scenario
            self.stage[selected] = next(i for i, (name, _) in enumerate(SCENARIOS[scenario]) if name == stage)
        combined = ids[task == 4]
        for high in (False, True):
            selected = combined[self.starts_biped[combined] == high]
            scenario, name = (1, 'biped_walk') if high else (0, 'tripod_walk')
            self.scenario[selected] = scenario
            self.stage[selected] = next(i for i, (stage, _) in enumerate(SCENARIOS[scenario]) if stage == name)
        self._write_direct_commands(direct)

    def _label_legacy_tasks(self, ids):
        labels = {name: 5 for name in STAGES}
        labels.update(settle=0, quad_stand=0, quad_walk=0, biped_settle=1,
                      biped_walk=1, lift=2, tripod_operation=2,
                      biped_operation=3, tripod_walk=4)
        lookup = torch.tensor([labels[name] for name in STAGES], device=self.device)
        self.task_id[ids] = lookup[self.stage_name_ids[ids]]
        manual_biped = self.starts_biped[ids] & (self.stage_name_ids[ids] == STAGES.index('settle'))
        self.task_id[ids[manual_biped]] = 1

    def _write_direct_commands(self, ids):
        task = self.task_id[ids]
        command = torch.zeros(len(ids), 6, device=self.device)
        walking = (task == 0) | (task == 1) | (task == 4)
        command[walking, :3] = self.twist[ids[walking]]
        command[task == 1, 3:] = self.nominal
        operation = (task == 2) | (task == 3) | (task == 4)
        selected = ids[operation]
        fraction = ((self.foot_elapsed[selected] - self.foot_hold[selected]) /
                    self.foot_motion[selected]).clamp(0., 1.)
        hub = torch.where(self.starts_biped[selected, None], self.nominal, 0.)
        first = (fraction * 2).clamp(0., 1.)
        second = (fraction * 2 - 1).clamp(0., 1.)
        smooth_first = (first * first * (3. - 2. * first))[:, None]
        smooth_second = (second * second * (3. - 2. * second))[:, None]
        path = torch.where((fraction <= .5)[:, None],
            self.foot_source[selected].lerp(hub, smooth_first),
            hub.lerp(self.foot_goal[selected], smooth_second))
        command[operation, 3:] = path
        returning = self.returning[ids]
        returning_ids = ids[returning]
        fraction = ((self.task_elapsed[returning_ids] - (self.task_duration[returning_ids] - 3.)) / .75).clamp(0., 1.)
        weight = (fraction * fraction * (3. - 2. * fraction))[:, None]
        command[returning, :3] = 0.
        command[returning, 3:] = self.return_source[returning_ids] * (1. - weight)
        command[returning & self.starts_biped[ids]] = 0.  # Exact keyboard R shortcut.
        self._command[ids] = command

    def _update_command(self):
        if not self.coverage_active:
            super()._update_command()
            self._label_legacy_tasks(torch.arange(self.num_envs, device=self.device))
            return
        advancing = self._last_step < self._env.common_step_counter
        self._last_step[:] = self._env.common_step_counter
        delta = advancing * self._env.step_dt
        direct = self.task_id != 5
        self.task_elapsed += delta * direct
        self.elapsed += delta
        self.command_elapsed += delta * direct
        self.foot_elapsed += delta * direct
        scheduled_return = ((~self.starts_biped & ((self.task_id == 2) | (self.task_id == 4))) |
                            self.r_return_episode) & direct
        return_ready = torch.ones_like(self.finished)
        if 'twist' in self._env.cfg.commands:
            coordinator = self._env.command_manager.get_term('twist')
            return_ready = (coordinator.owner == 0) & (coordinator.phase == 0)
        end = direct & (self.task_elapsed >= self.task_duration - 1.e-5)
        successful_return = end & scheduled_return & return_ready & ~self.return_outcome_recorded
        overdue_return = (direct & scheduled_return & ~return_ready &
                          (self.task_elapsed > self.task_duration + 5.) & ~self.return_outcome_recorded)
        self.return_success_counts += torch.bincount(self.task_id[successful_return], minlength=6)
        self.return_timeout_counts += torch.bincount(self.task_id[overdue_return], minlength=6)
        self.return_outcome_recorded |= successful_return | overdue_return
        self.finished |= end & (~scheduled_return | return_ready)
        self.timed_out |= overdue_return
        ids = (direct & ~self.finished).nonzero().flatten()
        changes = ids[self.command_elapsed[ids] >= self.command_duration[ids]]
        self._sample_twist(changes)
        retarget = ids[self.foot_elapsed[ids] >= self.foot_hold[ids] + self.foot_motion[ids]]
        self.foot_source[retarget] = self.foot_goal[retarget]
        for high in (False, True):
            selected = retarget[self.starts_biped[retarget] == high]
            self.foot_goal[selected] = self._sample_workspace(selected, high)
        self.foot_elapsed[retarget] = 0.
        self.foot_hold[retarget] = torch.empty(len(retarget), device=self.device).uniform_(2., 5.)
        self.foot_motion[retarget] = torch.empty(len(retarget), device=self.device).uniform_(1., 2.)
        entering_return = (scheduled_return & ~self.returning &
            (self.task_elapsed >= self.task_duration - 3.)).nonzero().flatten()
        self.return_source[entering_return] = self.command[entering_return, 3:]
        self.returning[entering_return] = True
        for high in (False, True):
            selected = entering_return[self.starts_biped[entering_return] == high]
            scenario = int(high)
            self.scenario[selected] = scenario
            self.stage[selected] = next(i for i, (name, _) in enumerate(SCENARIOS[scenario]) if name == 'return')
        self._write_direct_commands(ids)
        # Preserve the exact legacy cycle reference order and per-world teacher gates.
        transition = ~direct
        minimum = self.stage_seconds[self.scenario, self.stage]
        ready = transition & (self.elapsed >= minimum)
        if 'twist' in self._env.cfg.commands:
            teacher = self._env.command_manager.get_term('twist')
            names = self.stage_name_ids
            ready &= ((names != STAGES.index('rise')) | (teacher.owner == 3))
            ready &= ((names != STAGES.index('down')) | (teacher.owner == 0))
            ready &= ((names != STAGES.index('return')) | ((teacher.owner == 0) & (teacher.phase == 0)))
        self.timed_out |= transition & (self.elapsed > minimum + 5.)
        next_stage = self.stage + ready.long()
        self.finished |= transition & (next_stage >= self.stage_counts[self.scenario])
        self.stage = torch.minimum(next_stage, self.stage_counts[self.scenario] - 1)
        self.elapsed[ready] = 0.
        self._write_script_commands(transition.nonzero().flatten())

    def record_frames(self):
        """Called by pre-reset capture, never from command/shadow updates."""
        step = self._env.common_step_counter
        if int(self._recorded_step) == step:
            return
        self._recorded_step.fill_(step)
        self.actual_frames += torch.bincount(self.task_id, minlength=6)
        self.stage_frames += torch.bincount(self.stage_name_ids, minlength=len(STAGES))
        if 'twist' in self._env.cfg.commands:
            owner = self._env.command_manager.get_term('twist').owner
        else:
            owner = self.starts_biped.long() * 3
        self.owner_frames += torch.bincount(owner, minlength=5)
        velocity = self.command[:, :3]
        categories = torch.stack(((velocity == 0).all(-1), velocity[:, 0] < 0,
                                  velocity[:, 0] > 0, velocity[:, 2] > 0, velocity[:, 2] < 0,
                                  (velocity[:, 0] == 0) & (velocity[:, 2] > 0),
                                  (velocity[:, 0] == 0) & (velocity[:, 2] < 0)))
        self.velocity_frames += categories.sum(-1)
        operating = (((self.task_id == 2) | (self.task_id == 3) | (self.task_id == 4))
                     & ~self.returning & (self.command[:, 3:] != 0).any(-1))
        for high in (False, True):
            ids = (operating & (self.starts_biped == high)).nonzero().flatten()
            name = 'high' if high else 'low'
            xy = (self.command[ids, 3:5] - getattr(self, '_' + name + '_xy_lo')) / getattr(self, '_' + name + '_xy_scale')
            edge = ((xy * 2 - 1).abs().amax(-1) >= .65).long()
            z = torch.bucketize(self.command[ids, 5].contiguous(), getattr(self, '_' + name + '_z_cuts'))
            self.workspace_frames[int(high)] += torch.bincount(edge * 3 + z, minlength=6).reshape(2, 3)

    def record_failure(self, failed):
        """Record a physical terminal stage before reset (time limits excluded)."""
        step = self._env.common_step_counter
        if int(self._failure_step) == step:
            return
        self._failure_step.fill_(step)
        ids = failed.nonzero().flatten()
        indices = self.task_id[ids] * len(STAGES) + self.stage_name_ids[ids]
        self.failure_stage_counts += torch.bincount(indices, minlength=6 * len(STAGES)).reshape(6, len(STAGES))

    def coverage_metrics(self):
        metrics = {'training_progress': float(self.training_progress)}
        allocated = self.allocated_episodes.float() / self.allocated_episodes.sum().clamp_min(1)
        achieved = self.actual_frames.float() / self.actual_frames.sum().clamp_min(1)
        weights = self.allocation_weights()
        total_frames = self.actual_frames.sum().clamp_min(1)
        for i, name in enumerate(STAGES):
            metrics['stage/' + name] = float(self.stage_frames[i] / total_frames)
        for i, name in enumerate(('quadruped', 'other', 'rise', 'biped', 'down')):
            metrics['owner/' + name] = float(self.owner_frames[i] / total_frames)
        for i, name in enumerate(TASK_NAMES):
            metrics['allocated/' + name] = float(allocated[i])
            metrics['achieved/' + name] = float(achieved[i])
            metrics['allocation_weight/' + name] = float(weights[i])
            metrics['frames/' + name] = float(self.actual_frames[i])
            metrics['return_success/' + name] = float(self.return_success_counts[i])
            metrics['return_timeout/' + name] = float(self.return_timeout_counts[i])
            for j, stage in enumerate(STAGES):
                metrics['failure_stage/' + name + '/' + stage] = float(self.failure_stage_counts[i, j])
        for i, name in enumerate(('stop', 'backward', 'forward', 'turn_left', 'turn_right', 'inplace_left', 'inplace_right')):
            metrics['command/' + name] = float(self.velocity_frames[i] / self.actual_frames.sum().clamp_min(1))
        for high, name in enumerate(('low', 'high')):
            total = self.workspace_frames[high].sum().clamp_min(1)
            for edge, region in enumerate(('interior', 'edge')):
                for z in range(3):
                    metrics['workspace/' + name + '/' + region + '/z' + str(z)] = float(self.workspace_frames[high, edge, z] / total)
        return metrics

    def state_dict(self):
        return {name: getattr(self, name).detach().clone() for name in (
            'training_progress', 'allocated_episodes', 'actual_frames',
            'failure_stage_counts', 'velocity_frames', 'workspace_frames',
            'return_success_counts', 'return_timeout_counts', 'stage_frames', 'owner_frames')}

    def load_state_dict(self, state):
        for name, value in state.items():
            if name not in self.state_dict():
                continue
            target = getattr(self, name)
            value = torch.as_tensor(value, device=self.device, dtype=target.dtype)
            if target.shape != value.shape:
                raise ValueError('Coverage sampler shape mismatch for ' + name)
            target.copy_(value)
        self._recorded_step.fill_(-1)
        self._failure_step.fill_(-1)
