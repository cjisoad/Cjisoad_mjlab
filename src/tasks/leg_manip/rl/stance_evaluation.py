"""Independent mean-policy stage-zero trials, with fixed failure accounting."""
from contextlib import contextmanager
from copy import deepcopy
import random
import numpy as np
import torch
from ..mdp.commands import QUAD, HOLD
from ..mdp.stance_hold import StanceHoldHistory, stance_hold


class HoldTrials:
  def __init__(self, count, steps, device):
    self.steps = steps
    self.good = torch.zeros(count, device=device)
    self.failed = torch.zeros(count, dtype=torch.bool, device=device)

  def record(self, qualified, terminal):
    self.failed |= terminal
    self.good += qualified & ~self.failed

  def summary(self):
    return {'hold_fraction': float(self.good.sum() / (len(self.good) * self.steps)),
      'survival': float((~self.failed).float().mean()), 'trials': len(self.good),
      'planned_frames': len(self.good) * self.steps}


def apply_stance_evaluation(term, results, iteration):
  if term.stage != 0 or iteration <= term.last_stance_eval_iteration:
    return False
  passed = True
  decision = {'decision_stage': 0., 'decision_iteration': float(iteration)}
  for group in ('quad', 'biped'):
    row = results[group]
    valid = (row['trials'] >= 32 and row['planned_frames'] >= 1000
      and row['hold_fraction'] >= .7 and row['survival'] >= .85)
    passed &= valid
    for key, value in row.items():
      decision[group + '_' + key] = float(value)
    decision[group + '_ok'] = float(valid)
  decision['decision_passed'] = float(passed)
  term.last_decision = decision
  term.last_stance_eval_iteration = iteration
  term.pass_streak = term.pass_streak + 1 if passed else 0
  if term.pass_streak >= term.cfg.passing_windows:
    term.stage = 1
    term.pass_streak = 0
    term.stage_started = term._env.common_step_counter
    term.curriculum_window.zero_(); term.switch_window.zero_(); term.group_survival.zero_()
    term.course_episode_groups.zero_()
    term.curriculum_episodes = term.curriculum_failures = 0
    term.interrupt_switches(torch.arange(term.num_envs, device=term.device))
  return passed


@contextmanager
def isolated_rng(device):
  py_state, np_state = random.getstate(), np.random.get_state()
  # Environment seeding uses torch.manual_seed, which seeds every CUDA device
  # even when the evaluation itself runs on CPU or an implicit current device.
  devices = list(range(torch.cuda.device_count()))
  try:
    with torch.random.fork_rng(devices=devices):
      yield
  finally:
    random.setstate(py_state); np.random.set_state(np_state)


def evaluate_stances(actor, training_cfg, device, seed):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  cfg = deepcopy(training_cfg)
  cfg.seed = seed
  cfg.scene.num_envs = 64
  cfg.episode_length_s = 1e9
  cfg.commands['twist'].initial_stage = 0
  cfg.commands['twist'].curriculum_enabled = False
  cfg.commands['twist'].debug_vis = False
  cfg.commands['twist'].resampling_time_range = (1000., 1000.)
  cfg.curriculum = {}
  cfg.events.pop('push_robot', None)
  cfg.observations['actor'].terms['policy'].params['add_noise'] = False
  was_training = actor.training
  with isolated_rng(device):
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
      wrapped = RslRlVecEnvWrapper(env, clip_actions=10.)
      actor.eval()
      with torch.no_grad():
        obs, _ = env.reset()
        term = env.command_manager.get_term('twist')
        ids = torch.arange(64, device=device)
        biped_ids = ids[32:]
        # Fixed schedule:2s quad warmup,5s transition allowance,1s settle,5s scoring.
        dt = env.step_dt
        raise_step = round(2. / dt)
        measure_step = round(8. / dt)
        steps = round(5. / dt)
        trials = HoldTrials(64, steps, device)
        history = StanceHoldHistory(64, dt, device)
        term.set_command(ids, velocity=(0., 0., 0.), target_offset=(0., 0., 0.))
        term.manual.zero_(); term.time_left.fill_(1000.)
        obs = wrapped.get_observations()
        for step in range(measure_step + steps):
          if step == raise_step:
            term.set_command(biped_ids, target_offset=term.nominal_biped_offset)
            term.manual.zero_(); term.time_left.fill_(1000.)
          actions = actor(obs, stochastic_output=False)
          obs, _, dones, _ = wrapped.step(actions)
          # Sticky failure includes failures during arrival; successful-only filtering is forbidden.
          trials.failed |= dones.bool()
          history.update(term.contacts, term.robot.data.root_link_lin_vel_w, term.robot.data.root_link_ang_vel_w)
          if step >= measure_step:
            quad = stance_hold(term, history, biped=False) & (term.phase == QUAD) & ~term.mode
            biped = stance_hold(term, history, biped=True) & (term.phase == HOLD) & term.mode
            qualified = torch.cat((quad[:32], biped[32:]))
            trials.record(qualified, dones.bool())
        results = {}
        for group, group_ids in (('quad', ids[:32]), ('biped', ids[32:])):
          results[group] = {'hold_fraction': float(trials.good[group_ids].sum() / (32 * steps)),
            'survival': float((~trials.failed[group_ids]).float().mean()),
            'trials': 32, 'planned_frames': 32 * steps}
        return results
    finally:
      actor.train(was_training)
      env.close()
