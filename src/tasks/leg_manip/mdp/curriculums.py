"""Independent stance, switch, and walking gates in fixed-duration windows."""
import torch
from ..constants import STAGE_BIPED

CURRICULUM_GROUPS = ('quad_stance', 'biped_stance', 'quad_moving', 'biped_moving')

def manipulation_curriculum(env, env_ids):
  term = env.command_manager.get_term('twist')
  cfg = term.cfg
  if cfg.curriculum_enabled and term.stage < STAGE_BIPED:
    # All capability samples were streamed in their own stage/window. Episode
    # membership is also window-local; a long successful prefix cannot dilute
    # a terminal failure to one bad timestep among thousands of good samples.
    completed = term.stats[env_ids, 6] > 0
    ids = env_ids[completed]
    failures = term.stats[ids, 7] > 0
    term.curriculum_episodes += len(ids)
    term.curriculum_failures += int(failures.sum())
    if len(ids):
      term.group_survival[:, 0] += term.course_episode_groups[ids].sum(0)
      moving = ((term.command[ids, :3].norm(dim=-1) > .05)
                | (term.requested_velocity[ids].norm(dim=-1) > .05))
      context = term.mode[ids].long() + 2 * moving.long()
      for i in range(4):
        term.group_survival[i, 1] += (failures & (context == i)).sum()
    elapsed = env.common_step_counter - term.stage_started
    if elapsed >= cfg.min_stage_steps and term.curriculum_episodes >= cfg.min_curriculum_episodes:
      rows = term.curriculum_window
      sample_ok = rows[:, 0] >= cfg.min_group_samples
      ready_fraction = rows[:, 1] / rows[:, 0].clamp_min(1)
      error = rows[:, 2] / rows[:, 0].clamp_min(1)
      group_survival = 1. - term.group_survival[:, 1] / term.group_survival[:, 0].clamp_min(1)
      survival_ok = ((term.group_survival[:, 0] >= cfg.min_group_episodes) & (group_survival >= .8))
      group_ok = sample_ok & (ready_fraction >= .8) & (error < .15) & survival_ok
      required = 2 if term.stage == 0 else 4
      switches = term.switch_window
      switch_rate = switches[:, 1] / switches[:, 0].clamp_min(1)
      switch_ok = (switches[:, 0] >= cfg.min_switch_attempts) & (switch_rate >= .8)
      survival = 1. - term.curriculum_failures / max(term.curriculum_episodes, 1)
      passed = bool(group_ok[:required].all() & switch_ok.all()) and survival >= .8
      decision = {'decision_step': float(env.common_step_counter), 'decision_passed': float(passed),
        'decision_stage': float(term.stage), 'decision_velocity_level': float(term.velocity_level),
        'window_steps': float(elapsed), 'window_episodes': float(term.curriculum_episodes),
        'overall_survival': float(survival), 'overall_survival_ok': float(survival >= .8)}
      for i, group in enumerate(CURRICULUM_GROUPS):
        for field, value in (('samples', rows[i, 0]), ('ready_fraction', ready_fraction[i]),
                            ('velocity_error', error[i]), ('episodes', term.group_survival[i, 0]),
                            ('survival', group_survival[i]), ('survival_ok', survival_ok[i]),
                            ('samples_ok', sample_ok[i]), ('ok', group_ok[i])):
          decision[group + '_' + field] = float(value)
      for i, direction in enumerate(('rise', 'descent')):
        decision[direction + '_attempts'] = float(switches[i, 0])
        decision[direction + '_success'] = float(switch_rate[i])
        decision[direction + '_ok'] = float(switch_ok[i])
      term.last_decision = decision
      term.pass_streak = term.pass_streak + 1 if passed else 0
      if term.pass_streak >= cfg.passing_windows:
        if term.stage == 1 and term.velocity_level < 3:
          term.velocity_level += 1
        else:
          term.stage += 1
        term.pass_streak = 0
      # Every decision starts a fresh full-duration window, including failures.
      term.stage_started = env.common_step_counter
      term.curriculum_window.zero_()
      term.switch_window.zero_()
      term.group_survival.zero_()
      term.course_episode_groups.zero_()
      term.curriculum_episodes = 0
      term.curriculum_failures = 0
      term.interrupt_switches(torch.arange(term.num_envs, device=term.device))
  return {key: torch.tensor(value, device=env.device, dtype=torch.float32) for key, value in {
    **term.last_decision, 'stage': term.stage, 'velocity_level': term.velocity_level,
    'pass_streak': term.pass_streak}.items()}
