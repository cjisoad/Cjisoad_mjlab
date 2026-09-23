"""Advance task difficulty after measured successes over completed episodes."""

import torch


def manipulation_curriculum(env, env_ids):
  term = env.command_manager.get_term('twist')
  cfg = term.cfg
  if cfg.curriculum_enabled and term.stage < 3:
    selected = term.stats[env_ids]
    completed = selected[:, 6] > 0
    term.curriculum_window += selected[completed].sum(0)
    term.curriculum_episodes += int(completed.sum())
    elapsed = env.common_step_counter - term.stage_started
    if elapsed >= cfg.min_stage_steps and term.curriculum_episodes >= cfg.min_curriculum_episodes:
      stats = term.curriculum_window
      survival = 1. - stats[7] / max(term.curriculum_episodes, 1)
      quad_ok = (stats[0] >= 100) & (stats[1] / stats[0].clamp_min(1.) < .25)
      qualified = quad_ok & (survival >= .8)
      if term.stage > 0:
        qualified &= ((stats[2] >= 100) & (stats[3] / stats[2].clamp_min(1.) < .05)
          & (stats[4] / stats[2].clamp_min(1.) < .1) & (stats[8] >= 10)
          & (stats[9] / stats[8].clamp_min(1.) >= .8))
      if term.stage == 2:
        qualified &= stats[5] / stats[2].clamp_min(1.) < .15
      if qualified:
        term.stage += 1
        term.stage_started = env.common_step_counter
      # Evaluate fresh windows so early exploration does not permanently dilute progress.
      term.curriculum_window.zero_()
      term.curriculum_episodes = 0
  return {'stage': torch.tensor(term.stage, device=env.device, dtype=torch.float32)}
