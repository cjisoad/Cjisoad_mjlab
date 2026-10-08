"""Keyboard-controlled native MuJoCo playback for aligned teacher policies."""
import argparse
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import time


@dataclass(frozen=True)
class TeacherWorkspace:
  mode: str
  boxes: tuple
  targets: tuple = ()

  def contains(self, offset):
    return any(all(low - 1e-9 <= value <= high + 1e-9
                   for value, (low, high) in zip(offset, box)) for box in self.boxes)

  def clip(self, offset):
    # Preserve the requested height before clamping XY. This lets Q/E cross
    # the overlap even when the current XY lies only in the lower segment.
    z = max(min(box[2][0] for box in self.boxes),
            min(max(box[2][1] for box in self.boxes), offset[2]))
    candidates = []
    for box in self.boxes:
      if box[2][0] - 1e-9 <= z <= box[2][1] + 1e-9:
        candidates.append(tuple(max(low, min(high, value))
                                for value, (low, high) in zip((*offset[:2], z), box)))
    return min(candidates, key=lambda point: sum((a - b)**2 for a, b in zip(point, offset)))

  def project(self, offset):
    projected = self.clip(offset)
    if self.targets:
      return min(self.targets, key=lambda point: sum((a - b)**2 for a, b in zip(point, projected)))
    return projected


def workspace_for_mode(mode):
  from src.tasks.leg_manip.constants import DZ_MIN, TRIPOD_HIGH, BIPED_LOW, BIPED_HIGH, BIPED_X_RANGE, BIPED_Y_RANGE
  from src.tasks.leg_manip.mdp.foot_workspace import LOW
  from src.tasks.teacher_common.workspace import (
    LOCO_TEACHER_TRIPOD_X_RANGE, LOCO_TEACHER_TRIPOD_Y_RANGE,
    loco_teacher_foot_workspace)
  quad = (LOCO_TEACHER_TRIPOD_X_RANGE, LOCO_TEACHER_TRIPOD_Y_RANGE, (DZ_MIN, TRIPOD_HIGH))
  biped = (BIPED_X_RANGE, BIPED_Y_RANGE, (BIPED_LOW, BIPED_HIGH))
  boxes = {'quadruped': (quad,), 'biped': (biped,), 'union': (quad, biped)}
  if mode == 'quadruped':
    bank = loco_teacher_foot_workspace()
    targets = tuple(map(tuple, bank.offsets[bank.segment == LOW]))
    return TeacherWorkspace(mode, boxes[mode], targets)
  return TeacherWorkspace(mode, boxes[mode])


class TeacherKeyboardController:
  """Continuous absolute offsets from the common quadruped FR zero."""
  def __init__(self, workspace, *, idle_target=(0., 0., 0.),
               vx_range=(-.2, .6), wz_range=(-1., 1.), target_rate=.25):
    self.workspace = workspace
    self.idle_target = tuple(idle_target)
    self.vx_range, self.wz_range = vx_range, wz_range
    self.target_rate = target_rate
    self.reset()

  def reset(self):
    self.velocity = [0., 0., 0.]
    self.target = self.idle_target
    self.requested_target = self.idle_target
    self.target_active = False

  @property
  def command(self):
    return (*self.velocity, *self.target)

  def update(self, keys, dt, *, focused=True, paused=False):
    if not math.isfinite(dt) or dt < 0:
      raise ValueError('Input timestep must be finite and nonnegative')
    dt = min(dt, .1)
    if not focused or paused:
      self.velocity = [0., 0., 0.]
      return

    def direction(positive, negative):
      return int(positive in keys) - int(negative in keys)

    for index, sign, rate, bounds in (
        (0, direction('w', 's'), 1., self.vx_range),
        (2, direction('a', 'd'), 2., self.wz_range)):
      previous = self.velocity[index]
      if not sign or previous * sign < 0:
        previous = 0.
      self.velocity[index] = max(bounds[0], min(bounds[1], previous + sign * rate * dt)) if sign else 0.
    if 'r' in keys:
      self.target = self.requested_target = self.idle_target
      self.target_active = False
      return
    movement = (direction('right', 'left'), direction('up', 'down'), direction('q', 'e'))
    if any(movement):
      origin = self.requested_target if self.target_active else self.idle_target
      self.requested_target = self.workspace.clip(
        tuple(value + sign * self.target_rate * dt for value, sign in zip(origin, movement)))
      self.target = self.workspace.project(self.requested_target)
      self.target_active = True


def send_keyboard_command(env, env_ids, controller):
  command = controller.command
  env.command_manager.get_term('twist').set_command(
    env_ids, velocity=command[:3], target_offset=command[3:])
  # mjlab caches observations after each physics step. Manual input changes
  # between steps must be visible to the next policy query without advancing
  # observation history or resetting its buffers.
  env.observation_manager._obs_buffer = None


def keyboard_env_cfg(task, num_envs=1):
  from src.tasks.teacher_common.env_cfg import (
    pedipulation_teacher_env_cfg, loco_pedipulation_teacher_env_cfg)
  cfg = (pedipulation_teacher_env_cfg(play=True) if task == 'pedipulation_t'
         else loco_pedipulation_teacher_env_cfg(play=True))
  cfg.seed = 42
  cfg.scene.num_envs = num_envs
  # The loco source uses a one-frame history buffer. Its cached single frame
  # ignores commands changed between physics steps. No stacking is needed for
  # these feed-forward teachers: use fresh samples with identical dimensions.
  for group in cfg.observations.values():
    if group is not None and group.history_length == 1:
      group.history_length = 0
  return cfg


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('task', choices=('pedipulation_t', 'loco_pedipulation_t'))
  parser.add_argument('--checkpoint-file', type=Path, required=True)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--num-envs', type=int, default=1)
  parser.add_argument('--workspace', choices=('quadruped', 'biped', 'union'),
                      help='FR target range; defaults to the loaded teacher range')
  args = parser.parse_args(argv)
  if args.num_envs < 1:
    parser.error('--num-envs must be positive')
  return args


def main():
  args = parse_args()
  checkpoint = args.checkpoint_file.expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

  os.environ.setdefault('OMP_NUM_THREADS', '1')
  import torch
  import mujoco
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.viewer import NativeMujocoViewer
  from src.tasks.teacher_common.rl import TeacherOnPolicyRunner, teacher_ppo_runner_cfg

  try:
    from keyboard_x11 import X11Keyboard
  except Exception as exc:
    raise RuntimeError('Keyboard playback requires an active X11 desktop session') from exc

  configure_torch_backends()
  torch.set_num_threads(1)
  device = args.device
  cfg = keyboard_env_cfg(args.task, args.num_envs)
  mode = args.workspace or ('biped' if args.task == 'pedipulation_t' else 'quadruped')
  workspace = workspace_for_mode(mode)
  env = keyboard = None
  try:
    keyboard = X11Keyboard()
    env = ManagerBasedRlEnv(cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=10.)
    term = env.command_manager.get_term('twist')
    nominal = (term.nominal_biped_offset.detach().cpu().tolist()
               if args.task == 'pedipulation_t' else [0., 0., 0.])
    controller = TeacherKeyboardController(
      workspace, idle_target=nominal, vx_range=term.cfg.lin_vel_x,
      wz_range=getattr(term.cfg, 'ang_vel_z', (-1., 1.)))
    ids = torch.arange(env.num_envs, device=device)
    runner = TeacherOnPolicyRunner(wrapped, asdict(teacher_ppo_runner_cfg(args.task)), device=device)
    runner.load(str(checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)

    class KeyboardViewer(NativeMujocoViewer):
      def __init__(self):
        super().__init__(wrapped, policy)
        self.last_input = time.monotonic()

      def _safe_key_callback(self, key):
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
          controller.update(keyboard.read(), now - self.last_input,
                            focused=keyboard.focused, paused=self._is_paused)
          self.last_input = now
        # Reassert manual ownership after automatic episode resets, too.
        send_keyboard_command(env, ids, controller)
        return super().tick()

      def _set_status_overlay(self, viewer):
        status = self.get_status()
        vx, _, wz, dx, dy, dz = controller.command
        labels = 'State\nStep\nKeys\nWorkspace\nFR target\nvx\nwz\nFR dx (m)\nFR dy (m)\nFR dz (m)'
        values = (f"{'PAUSED' if status.paused else 'RUNNING'}\n{status.step_count}\n"
                  f"{'ACTIVE' if keyboard.focused else 'INACTIVE'}\n"
                  f"{mode}\n{'ON' if controller.target_active else 'DEFAULT'}\n"
                  f'{vx:+.2f}\n{wz:+.2f}\n{dx:+.2f}\n{dy:+.2f}\n{dz:+.2f}')
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                          mujoco.mjtGridPos.mjGRID_TOPLEFT.value, labels, values))

    print(f'[INFO] Task: {args.task}\n[INFO] Checkpoint: {checkpoint}', flush=True)
    print(f'[INFO] Workspace: {mode}; XYZ boxes (m): {workspace.boxes}', flush=True)
    print('[INFO] FR offsets are absolute from the shared quadruped zero; R restores the loaded teacher default.', flush=True)
    print('[INFO] Focus the MuJoCo window. WASD moves/turns; arrows adjust FR XY; Q/E adjust height; R clears target; Backspace resets; Space pauses.', flush=True)
    KeyboardViewer().run()
  finally:
    if keyboard:
      keyboard.close()
    if env:
      env.close()


if __name__ == '__main__':
  main()
