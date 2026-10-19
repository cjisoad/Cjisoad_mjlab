"""Closed-loop evaluation, with physical metrics captured before automatic reset."""
import time
from pathlib import Path

import torch

from .intent import STAGES


def capture_sample(env):
    """Snapshot the manager's pre-reset frame without terminating the world.

    Action and reset flags are captured before automatic reset. Derived contact
    and kinematic fields are the values available at the termination-manager
    phase; the report labels this phase instead of implying an extra forward.
    """
    from src.tasks.teacher_common.anchor_frame import anchor_basis, to_anchor
    robot = env.scene['robot']
    intent = env.command_manager.get_term('intent')
    basis = anchor_basis(robot.data.root_link_quat_w, robot.data.gravity_vec_w)
    fr_id = robot.find_bodies('FR_foot')[0][0]
    fr_offset = to_anchor(basis, robot.data.body_link_pos_w[:, fr_id]
                          - robot.data.root_link_pos_w) - intent.foot_zero
    force = env.scene['feet_ground_contact'].data.force
    contacts = force.reshape(env.num_envs, 4, -1, 3).sum(2).norm(dim=-1) > 5.
    velocity = to_anchor(basis, robot.data.root_link_lin_vel_w)
    yaw_rate = (robot.data.root_link_ang_vel_w * basis[:, :, 2]).sum(-1)
    actual_twist = torch.cat((velocity[:, :2], yaw_rate[:, None]), dim=-1)
    env.legmanip_eval_sample = {
        'stage': intent.stage_name_ids.clone(),
        'elapsed': intent.elapsed.clone(),
        'completed': intent.finished.clone(),
        'timed_out': intent.timed_out.clone(),
        'gravity': robot.data.projected_gravity_b.clone(),
        'height': (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).clone(),
        'contacts': contacts.clone(),
        'foot_error': (fr_offset - intent.command[:, 3:]).norm(dim=-1),
        'velocity_error': (actual_twist - intent.command[:, :3]).abs().mean(-1),
        'applied_action': env.action_manager.get_term('joint_pos').raw_action.clone(),
    }
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def episode_result(record, dt):
    """Diagnostic task success requires continuous physical holds, not a timer."""
    reasons = []
    if record['physical_failure']:
        reasons.append('physical_failure')
    if not record['completed']:
        reasons.append('scenario_incomplete')
    if record['deadline']:
        reasons.append('evaluation_deadline')
    required = ['final_quad_hold']
    if record['scenario'] != 'quad':
        required.append('biped_hold')
    if record['scenario'] != 'biped':
        required.append('tripod_hold')
    reasons.extend(name + '_too_short' for name in required if record[name] < .5 - 1.e-6)
    return {
        'scenario': record['scenario'], 'success': not reasons, 'failure_reasons': reasons,
        'script_completed': record['completed'], 'physical_failure': record['physical_failure'],
        'duration_s': record['steps'] * dt, 'entry_trajectory': record['entry_trajectory'],
        'biped_hold_s': record['biped_hold'], 'tripod_hold_s': record['tripod_hold'],
        'final_quad_hold_s': record['final_quad_hold'],
        'mean_twist_abs_error': record['velocity_error_sum'] / max(record['steps'], 1),
        'mean_operation_foot_error_m': (record['foot_error_sum'] / record['foot_samples']
                                       if record['foot_samples'] else None),
        'mean_substep_to_final_target_difference_rad': record['substep_difference_sum'] / max(record['steps'], 1),
        'max_substep_to_final_target_difference_rad': record['substep_difference_max'],
    }


def new_record(scenario, trajectory):
    return dict(scenario=scenario, steps=0, completed=False, physical_failure=False,
                deadline=False, biped_hold=0., tripod_hold=0., final_quad_hold=0.,
                biped_run=0., tripod_run=0., final_quad_run=0.,
                velocity_error_sum=0., foot_error_sum=0., foot_samples=0,
                substep_difference_sum=0., substep_difference_max=0.,
                entry_trajectory=trajectory)


@torch.inference_mode()
def evaluate_policy(*, policy='student', checkpoint=None, scenario='all', episodes=16,
                    num_envs=16, max_steps=3000, device='cpu', split='validation', seed=123):
    """Use a disjoint entry split and the same scripted intent for each policy.

    teacher-hold recomputes the frozen teacher on its own visited states and
    applies only the final target, directly through student PD. Compare it with
    teacher to quantify the substep execution mismatch in closed loop.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import TerminationTermCfg
    from tensordict import TensorDict
    from .env_cfg import legmanip_distil_env_cfg
    from .deployment import validate_contract
    from .rl import load_student

    if policy not in ('student', 'teacher', 'teacher-hold'):
        raise ValueError('Unknown evaluation policy')
    if min(episodes, num_envs, max_steps) < 1 or split not in ('validation', 'final'):
        raise ValueError('Use positive evaluation counts and validation/final entry split')
    scenarios = ('quad', 'biped', 'cycle') if scenario == 'all' else (scenario,)
    if any(name not in ('quad', 'biped', 'cycle') for name in scenarios):
        raise ValueError('Unknown evaluation scenario')
    cfg = legmanip_distil_env_cfg(play=policy == 'student')
    cfg.scene.num_envs = min(num_envs, episodes)
    cfg.seed = seed
    cfg.commands['intent'].automatic = True
    cfg.commands['intent'].scenario = scenarios[0]
    cfg.commands['intent'].entry_split = split
    cfg.terminations = {'capture': TerminationTermCfg(func=capture_sample), **cfg.terminations}
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        metadata = None
        if policy == 'student':
            if checkpoint is None:
                raise ValueError('Student evaluation requires a checkpoint')
            actor, metadata = load_student(checkpoint, device=device, num_envs=env.num_envs)
            validate_contract(metadata, env)
        else:
            from .teacher import FSMTeacher
            actor = FSMTeacher(env)
        intent = env.command_manager.get_term('intent')
        action = env.action_manager.get_term('joint_pos')
        results = []
        timing_by_owner = {str(owner): {'frames': 0, 'sum_abs_rad': 0., 'max_abs_rad': 0.}
                           for owner in (0, 2, 3, 4)}
        started = time.perf_counter()

        def trajectory(index):
            selected = int(intent.entry_indices[index])
            return (int(intent.entry_bank.data['trajectory_id'][selected]) if selected >= 0 else None)

        for name in scenarios:
            intent.cfg.scenario = name
            obs, _ = env.reset(seed=seed)
            actor.reset()
            # Fixed batches keep teacher and teacher-hold initial states and
            # commands paired even when one execution mode falls earlier.
            completed = 0
            batch = 0
            while completed < episodes:
                if batch:
                    obs, _ = env.reset(seed=seed + batch)
                    actor.reset()
                count = min(env.num_envs, episodes - completed)
                active = [i < count for i in range(env.num_envs)]
                records = [new_record(name, trajectory(i)) for i in range(env.num_envs)]
                for step in range(max_steps):
                    actions = actor(TensorDict(obs, batch_size=[env.num_envs]))
                    difference = torch.zeros(env.num_envs, device=device)
                    maximum = difference.clone()
                    if policy != 'student':
                        actor.set_control_mask(torch.full((env.num_envs,), policy == 'teacher',
                                                         device=device, dtype=torch.bool))
                        deviation = (action.teacher_targets - action.teacher_targets[-1:]).abs()
                        difference = deviation.mean(dim=(0, 2))
                        maximum = deviation.amax(dim=(0, 2))
                        owners = env.command_manager.get_term('twist').owner
                        for i in range(count):
                            if active[i]:
                                stat = timing_by_owner[str(int(owners[i]))]
                                stat['frames'] += 1
                                stat['sum_abs_rad'] += float(difference[i])
                                stat['max_abs_rad'] = max(stat['max_abs_rad'], float(maximum[i]))
                    obs, _, terminated, truncated, _ = env.step(actions)
                    dones = terminated | truncated
                    actor.reset(dones)
                    sample = env.legmanip_eval_sample
                    for i in range(count):
                        if not active[i]:
                            continue
                        record = records[i]
                        record['steps'] += 1
                        record['velocity_error_sum'] += float(sample['velocity_error'][i])
                        record['substep_difference_sum'] += float(difference[i])
                        record['substep_difference_max'] = max(record['substep_difference_max'], float(maximum[i]))
                        stage = STAGES[int(sample['stage'][i])]
                        gravity, height = sample['gravity'][i], float(sample['height'][i])
                        contacts = sample['contacts'][i]
                        biped = (stage.startswith('biped') and float(gravity[0]) < -.85
                                 and abs(float(gravity[1])) < .25 and .4 < height < .65
                                 and bool(contacts[2:].all()))
                        tripod = (stage == 'tripod_operation' and float(sample['foot_error'][i]) < .12
                                  and bool(contacts[[0, 2, 3]].all()) and not bool(contacts[1]))
                        quadruped = (stage == 'return' and float(gravity[2]) < -.85
                                     and .18 < height < .45 and bool(contacts.all()))
                        for prefix, good in (('biped', biped), ('tripod', tripod), ('final_quad', quadruped)):
                            record[prefix + '_run'] = record[prefix + '_run'] + env.step_dt if good else 0.
                            record[prefix + '_hold'] = max(record[prefix + '_hold'], record[prefix + '_run'])
                        if 'operation' in stage:
                            record['foot_error_sum'] += float(sample['foot_error'][i])
                            record['foot_samples'] += 1
                        if bool(dones[i]) or step == max_steps - 1:
                            record['physical_failure'] = bool(terminated[i])
                            record['completed'] = bool(sample['completed'][i])
                            record['deadline'] = step == max_steps - 1 and not bool(dones[i])
                            results.append(episode_result(record, env.step_dt))
                            active[i] = False
                            completed += 1
                    if not any(active):
                        break
                batch += 1
        for stat in timing_by_owner.values():
            stat['mean_abs_rad'] = stat.pop('sum_abs_rad') / max(stat['frames'], 1)
        summary = {}
        for name in scenarios:
            rows = [row for row in results if row['scenario'] == name]
            summary[name] = dict(episodes=len(rows), successes=sum(row['success'] for row in rows),
                                 physical_failures=sum(row['physical_failure'] for row in rows),
                                 script_completions=sum(row['script_completed'] for row in rows))
        return dict(policy=policy, checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
                    completed_updates=metadata['completed_updates'] if metadata else None,
                    split=split, seed=seed, device=str(device), num_envs=env.num_envs,
                    max_steps=max_steps, wall_seconds=time.perf_counter() - started,
                    criteria=dict(continuous_hold_s=.5, biped_gravity_x_max=-.85,
                                  biped_height_m=[.4, .65], tripod_foot_error_max_m=.12,
                                  final_quad_gravity_z_max=-.85, final_quad_height_m=[.18, .45],
                                  contact_force_min_n=5.,
                                  capture_phase='termination_manager_pre_auto_reset'),
                    summary=summary, timing_by_owner=timing_by_owner, episodes=results)
    finally:
        env.close()
