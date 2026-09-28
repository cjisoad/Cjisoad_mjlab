"""Native MuJoCo keyboard playback for a Loco Pedipulation checkpoint."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from keyboard_control import KeyboardController

ROOT = Path(__file__).resolve().parents[1]
PHASE_NAMES = ('QUAD', 'PREPARE', 'REACH', 'HOLD', 'LOWER', 'RECOVER')


class KeyboardCommandBridge:
  """Send the shared six-value keyboard command to Loco Pedipulation."""

  def __init__(self, term, num_envs):
    self.term = term
    self.env_ids = tuple(range(num_envs))
    self._previous_target = None

  def update(self, controller):
    values = tuple(float(value) for value in controller.command)
    velocity, offset = values[:3], values[3:]
    target = offset if any(offset) else (0., 0., 0.)
    changed = target != self._previous_target
    self.term.set_command(self.env_ids, velocity=velocity,
                          target_offset=target if changed else None)
    self._previous_target = target

  def release(self):
    self.term.set_command(self.env_ids, velocity=(0., 0., 0.),
                          target_offset=(0., 0., 0.))
    self.term.release_manual(self.env_ids)
    self._previous_target = None


def run_viewer(env, policy, controller, bridge, keyboard, status_file):
  import mujoco
  from mjlab.viewer import NativeMujocoViewer

  class KeyboardViewer(NativeMujocoViewer):
    def __init__(self):
      super().__init__(env, policy)
      self.last_input = time.monotonic()
      self.last_status = 0.

    def _safe_key_callback(self, key):
      if key == 259:
        self.request_reset()
      elif key not in {ord(k) for k in 'WASDQER'} | {262, 263, 264, 265}:
        super()._safe_key_callback(key)

    def reset_environment(self):
      controller.reset()
      bridge.release()
      super().reset_environment()

    def tick(self):
      now = time.monotonic()
      if now - self.last_input >= .01:
        controller.update(keyboard.read(), now - self.last_input,
                          focused=keyboard.focused, paused=self._is_paused)
        bridge.update(controller)
        self.last_input = now
      return super().tick()

    def _set_status_overlay(self, viewer):
      status, term = self.get_status(), bridge.term
      command = controller.command
      enabled = any(command[3:])
      phase = PHASE_NAMES[int(term.phase[0])] if len(term.phase) else '-'
      labels = 'State\nStep\nInput\nvx (m/s)\nvy (m/s)\nwz (rad/s)\nFR target\ndx (m)\ndy (m)\ndz delta (m)\nFR phase'
      values = (f"{'PAUSED' if status.paused else 'RUNNING'}\n{status.step_count}\n"
                f"{'ACTIVE' if keyboard.focused else 'INACTIVE'}\n"
                f'{command[0]:+.3f}\n{command[1]:+.3f}\n{command[2]:+.3f}\n'
                f"{'ON' if enabled else 'OFF'}\n{command[3]:+.3f}\n{command[4]:+.3f}\n"
                f'{command[5]:+.3f}\n{phase}')
      viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT.value, labels, values))
      now = time.monotonic()
      if status_file and now - self.last_status >= .2:
        snapshot = {'pid': os.getpid(), 'step': status.step_count, 'paused': status.paused,
                    'focused': keyboard.focused, 'command': command, 'fr_enabled': enabled,
                    'phase': phase, 'observed_command': term.command[0].tolist(),
                    'last_error': status.last_error}
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(snapshot, indent=2))
        temporary.replace(status_file)
        self.last_status = now

  KeyboardViewer().run()


def smoke(env, policy, controller, bridge, steps):
  import torch

  term = env.unwrapped.command_manager.get_term('twist')
  with torch.inference_mode():
    for step in range(steps):
      keys = {'w', 'right', 'q'} if 150 <= step < 170 else set()
      controller.update(keys, env.unwrapped.step_dt)
      bridge.update(controller)
      obs = env.get_observations()
      action = policy(obs)
      assert torch.isfinite(action).all()
      obs, reward, _, _ = env.step(action)
      assert torch.isfinite(reward).all() and all(torch.isfinite(value).all() for value in obs.values())
    assert term.manual.all(), 'Keyboard command must override automatic resampling'
    assert term.enabled.all(), 'Right-front target must remain active after held-key release'
    controller.reset()
    bridge.release()
    env.reset()
    assert not term.manual.any() and not term.enabled.any()
    torch.testing.assert_close(term.command, torch.zeros_like(term.command))
  print(json.dumps({'smoke': 'passed', 'steps': steps, 'reset': 'passed',
                    'command_shape': list(term.command.shape)}, indent=2), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint-file', type=Path, required=True,
                      help='48-input Loco Pedipulation checkpoint trained with this interface')
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--status-file', type=Path, help='Optional live JSON for diagnostics')
  parser.add_argument('--smoke', action='store_true', help='Run a bounded headless command/rollout check')
  parser.add_argument('--steps', type=int, default=550, help='Headless smoke steps, at least 501')
  args = parser.parse_args()
  checkpoint = args.checkpoint_file.expanduser().resolve()
  if not checkpoint.is_file():
    parser.error(f'Checkpoint not found: {checkpoint}')
  if args.smoke and args.steps < 501:
    parser.error('--steps must be at least 501 to cross command resampling')

  os.environ.setdefault('OMP_NUM_THREADS', '1')
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from src.tasks.loco_pedipulation.env_cfg import loco_pedipulation_env_cfg
  from src.tasks.loco_pedipulation.rl import LocoPedipulationOnPolicyRunner, loco_pedipulation_ppo_runner_cfg

  keyboard = env = None
  try:
    if not args.smoke:
      from keyboard_x11 import X11Keyboard
      keyboard = X11Keyboard()
    configure_torch_backends()
    torch.set_num_threads(1)
    controller = KeyboardController()
    cfg = loco_pedipulation_env_cfg(play=True)
    cfg.seed = args.seed
    print(f'[INFO] Checkpoint: {checkpoint}', flush=True)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=10.)
    bridge = KeyboardCommandBridge(env.command_manager.get_term('twist'), env.num_envs)
    runner = LocoPedipulationOnPolicyRunner(wrapped, asdict(loco_pedipulation_ppo_runner_cfg()), device=args.device)
    runner.load(str(checkpoint), load_cfg={'actor': True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    if args.smoke:
      smoke(wrapped, policy, controller, bridge, args.steps)
    else:
      if args.status_file:
        args.status_file = args.status_file.expanduser().resolve()
        args.status_file.parent.mkdir(parents=True, exist_ok=True)
      run_viewer(wrapped, policy, controller, bridge, keyboard, args.status_file)
  finally:
    if keyboard:
      keyboard.close()
    if env:
      env.close()


if __name__ == '__main__':
  main()
