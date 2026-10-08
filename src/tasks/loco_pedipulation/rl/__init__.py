"""Standard PPO with the velocity runner's policy export support."""

from src.tasks.velocity.config.go2.rl_cfg import unitree_go2_ppo_runner_cfg
from src.tasks.velocity.rl import VelocityOnPolicyRunner
from .metrics import ManipulationLogger


def loco_pedipulation_ppo_runner_cfg():
  cfg = unitree_go2_ppo_runner_cfg()
  cfg.actor.obs_normalization = False
  cfg.critic.obs_normalization = False
  cfg.experiment_name = 'loco_pedipulation'
  cfg.logger = 'tensorboard'
  cfg.upload_model = False
  cfg.max_iterations = 13000
  cfg.clip_actions = 10.
  return cfg


class LocoPedipulationOnPolicyRunner(VelocityOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
    super().__init__(env, train_cfg, log_dir, device)
    term = self.env.unwrapped.command_manager.get_term('twist')
    self.logger = ManipulationLogger(self.logger, term.training_metrics, self.is_distributed)

  def _curriculum_state(self):
    term = self.env.unwrapped.command_manager.get_term('twist')
    return {'stage': term.stage, 'stage_started': term.stage_started,
            'window': term.curriculum_window.clone(), 'episodes': term.curriculum_episodes,
            'landing_counts': self.logger.landing_totals.clone()}

  def save(self, path, infos=None):
    super().save(path, {**(infos or {}), 'loco_pedipulation': self._curriculum_state()})

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    term = self.env.unwrapped.command_manager.get_term('twist')
    state = (infos or {}).get('loco_pedipulation')
    if state and term.cfg.curriculum_enabled:
      term.stage = state['stage']
      term.stage_started = state['stage_started']
      term.curriculum_window.copy_(state['window'])
      term.curriculum_episodes = state['episodes']
      if 'landing_counts' in state:
        self.logger.restore_counts(state['landing_counts'])
    return infos
