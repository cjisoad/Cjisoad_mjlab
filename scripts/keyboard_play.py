"""Native MuJoCo keyboard playback of a Pedipulation checkpoint."""
import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time

from keyboard_control import KeyboardController

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'logs/rsl_rl/pedipulation/2026-09-22_16-56-16_fr_target/model_13800.pt'
TRAINING_SOURCE = ROOT / 'outputs/wandb_a3vowvqj/source'


def manual_command_cfg(original, controller):
  from src.tasks.pedipulation.mdp.commands import PedipulationCommand, PedipulationCommandCfg

  class ManualCommand(PedipulationCommand):
    def _resample_command(self, env_ids):
      self._command[env_ids] = self._command.new_tensor(controller.command)

    def _update_command(self):
      quat = self._env.scene['robot'].data.root_link_quat_w
      self._anchor_frame.update(quat, self._env.step_dt, self._env.common_step_counter)
      self._command[:] = self._command.new_tensor(controller.command)

  @dataclass(kw_only=True)
  class ManualCommandCfg(PedipulationCommandCfg):
    def build(self, env):
      return ManualCommand(self, env)

  return ManualCommandCfg(**vars(original))


def run_viewer(env, policy, controller, keyboard, status_file):
  import mujoco
  from mjlab.viewer import NativeMujocoViewer

  class KeyboardViewer(NativeMujocoViewer):
    def __init__(self):
      super().__init__(env, policy)
      self.last_input = time.monotonic()
      self.last_status = 0.

    def _safe_key_callback(self, key):
      # MuJoCo invokes this on its render thread; only queue viewer actions.
      if key == 259:
        self.request_reset()
      elif key not in {ord(k) for k in 'WASDQER'} | {262, 263, 264, 265}:
        super()._safe_key_callback(key)

    def reset_environment(self):
      controller.reset()
      super().reset_environment()

    def tick(self):
      now = time.monotonic()
      if now - self.last_input >= .01:
        keys = keyboard.read()
        controller.update(keys, now - self.last_input,
                          focused=keyboard.focused, paused=self._is_paused)
        self.last_input = now
      return super().tick()

    def _set_status_overlay(self, viewer):
      status = self.get_status()
      command = controller.command
      active = any(command[3:])
      term = env.unwrapped.command_manager.get_term('stand')
      error = None
      if active:
        robot = env.unwrapped.scene['robot']
        foot_id = robot.find_sites('FR')[0][0]
        error = (robot.data.site_pos_w[0, foot_id] - term.foot_target_pos_w[0]).norm().item()
      labels = 'State\nStep\nInput\nvx (m/s)\nwz (rad/s)\nFR target\ndx (m)\ndy (m)\ndz (m)\nFR error (m)'
      values = (f"{'PAUSED' if status.paused else 'RUNNING'}\n{status.step_count}\n"
                f"{'ACTIVE' if keyboard.focused else 'INACTIVE'}\n"
                f'{command[0]:+.3f}\n{command[2]:+.3f}\n'
                f"{'ON' if active else 'OFF'}\n"
                f'{command[3]:+.3f}\n{command[4]:+.3f}\n{command[5]:+.3f}\n'
                + (f'{error:.3f}' if error is not None else '-'))
      viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT.value, labels, values))
      now = time.monotonic()
      if status_file and now - self.last_status >= .2:
        snapshot = {'pid': os.getpid(), 'step': status.step_count, 'paused': status.paused,
                    'focused': keyboard.focused, 'command': command,
                    'observed_command': term.command[0].tolist(),
                    'fr_error_m': error, 'last_error': status.last_error}
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(snapshot, indent=2))
        temporary.replace(status_file)
        self.last_status = now

  KeyboardViewer().run()


def smoke(env, policy, controller, steps):
  import torch
  term = env.unwrapped.command_manager.get_term('stand')
  with torch.inference_mode():
    for step in range(steps):
      keys = {'w', 'a', 'up', 'q'} if 150 <= step < 170 else set()
      controller.update(keys, env.unwrapped.step_dt)
      obs = env.get_observations()
      expected = torch.tensor(controller.command, device=env.device).expand(env.num_envs, -1)
      torch.testing.assert_close(term.command, expected)
      action = policy(obs)
      assert torch.isfinite(action).all()
      obs, reward, _, _ = env.step(action)
      assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
      torch.testing.assert_close(term.command, expected)
    assert any(controller.command[3:]), 'Target should survive the 10-second resampling'
    controller.reset()
    env.reset()
    torch.testing.assert_close(term.command, torch.zeros_like(term.command))
  print(json.dumps({'smoke': 'passed', 'steps': steps, 'reset': 'passed',
                    'command_shape': list(term.command.shape)}, indent=2), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint-file', type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument('--source-root', type=Path, help='Repository/source snapshot matching the checkpoint')
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--status-file', type=Path, help='Optional live JSON for diagnostics')
  parser.add_argument('--smoke', action='store_true', help='Run a bounded headless command/rollout check')
  parser.add_argument('--steps', type=int, default=550, help='Headless smoke steps, at least 501')
  args = parser.parse_args()
  checkpoint = args.checkpoint_file.expanduser().resolve()
  source = (args.source_root.expanduser().resolve() if args.source_root else
            TRAINING_SOURCE if checkpoint == DEFAULT_CHECKPOINT else ROOT)
  if not checkpoint.is_file():
    parser.error(f'Checkpoint not found: {checkpoint}')
  if not (source / 'src/tasks/pedipulation/mdp/commands.py').is_file():
    parser.error(f'Pedipulation source not found: {source}; provide --source-root')
  if args.smoke and args.steps < 501:
    parser.error('--steps must be at least 501 to cross command resampling')
  sys.path.insert(0, str(source))
  os.environ.setdefault('OMP_NUM_THREADS', '1')
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from src.tasks.pedipulation.env_cfg import pedipulation_env_cfg
  from src.tasks.pedipulation.rl import PedipulationOnPolicyRunner, pedipulation_ppo_runner_cfg

  keyboard, env = None, None
  try:
    if not args.smoke:
      from keyboard_x11 import X11Keyboard
      keyboard = X11Keyboard()
    configure_torch_backends()
    torch.set_num_threads(1)
    controller = KeyboardController()
    cfg = pedipulation_env_cfg(play=True)
    cfg.seed = args.seed
    cfg.commands['stand'] = manual_command_cfg(cfg.commands['stand'], controller)
    cfg.viewer.azimuth, cfg.viewer.elevation = 135, -15
    print(f'[INFO] Checkpoint: {checkpoint}\n[INFO] Source: {source}', flush=True)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.)
    term = env.command_manager.get_term('stand')
    controller.min_z_offset = max(-.10, term.foot_workspace.min_height - term._foot_zero_anchor[2].item())
    runner = PedipulationOnPolicyRunner(wrapped, asdict(pedipulation_ppo_runner_cfg()), device=args.device)
    runner.load(str(checkpoint), load_cfg={'actor': True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    if args.smoke:
      smoke(wrapped, policy, controller, args.steps)
    else:
      if args.status_file:
        args.status_file = args.status_file.expanduser().resolve()
        args.status_file.parent.mkdir(parents=True, exist_ok=True)
      run_viewer(wrapped, policy, controller, keyboard, args.status_file)
  finally:
    if keyboard:
      keyboard.close()
    if env:
      env.close()


if __name__ == '__main__':
  main()
