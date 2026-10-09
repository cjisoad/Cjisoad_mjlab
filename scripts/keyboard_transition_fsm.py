"""Keyboard and bounded scripted playback of the actual actor75 teacher chain."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def load_chain(args, *, render_mode=None):
  os.environ.setdefault('OMP_NUM_THREADS', '1')
  os.environ.setdefault('MPLCONFIGDIR', '/tmp/transition_t_mpl')
  os.environ.setdefault('XDG_CACHE_HOME', '/tmp/transition_t_cache')
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from src.tasks.transition_t.keyboard import keyboard_transition_env_cfg, AlignedKeyboardTeacher
  from src.tasks.transition_t.rl import TransitionOnPolicyRunner, transition_ppo_runner_cfg
  from src.tasks.pedipulation_bridge_t.experts import DEFAULT_BIPED_CHECKPOINT
  paths = {name: Path(path).expanduser().resolve() for name, path in (
    ('tripod', args.checkpoint_file), ('transition_t', args.transition_checkpoint_file),
    ('biped', args.biped_checkpoint_file or DEFAULT_BIPED_CHECKPOINT))}
  for name, path in paths.items():
    if not path.is_file():
      raise FileNotFoundError(f'{name} checkpoint not found: {path}')
  configure_torch_backends()
  torch.set_num_threads(1)
  cfg = keyboard_transition_env_cfg(args.num_envs)
  cfg.seed = 42
  if render_mode:
    cfg.viewer.width, cfg.viewer.height = 960, 720
  env = ManagerBasedRlEnv(cfg, device=args.device, render_mode=render_mode)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=10.)
    runner = TransitionOnPolicyRunner(wrapped, asdict(transition_ppo_runner_cfg()), device=args.device)
    runner.load(str(paths['transition_t']), load_cfg={'actor': True}, map_location=args.device,
      allow_legacy_target_limiter=getattr(args, 'allow_legacy_transition_limiter', False))
    term = env.command_manager.get_term('twist')
    term.configure_trigger(args.transition_margin)
    action = env.action_manager.get_term('joint_pos')
    action.source_teacher = AlignedKeyboardTeacher(paths['tripod'], 'loco_pedipulation_t', args.device)
    action.biped_teacher = AlignedKeyboardTeacher(paths['biped'], 'pedipulation_t', args.device)
    # Loading restores the training counter. Reset playback clocks together;
    # log elapsed playback time rather than the training run's accumulated time.
    term.session_start_step = env.common_step_counter
    wrapped.reset()
    return env, wrapped, runner.get_inference_policy(device=args.device), paths
  except Exception:
    env.close()
    raise


def keyboard_inputs(env):
  from keyboard_teacher_fsm import KeyboardInputs
  from src.tasks.teacher_common.env_cfg import loco_pedipulation_teacher_env_cfg, pedipulation_teacher_env_cfg
  from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace
  # The command intentionally uses the established WARM/BRIDGE/VERIFY state IDs.
  return KeyboardInputs(env.command_manager.get_term('twist'),
    loco_pedipulation_teacher_env_cfg(play=True).commands['twist'],
    pedipulation_teacher_env_cfg(play=True).commands['twist'],
    build_foot_workspace().nominal_biped_offset)


def print_events(term, events=None):
  while term.keyboard_events:
    event = term.keyboard_events.pop(0)
    if events is not None:
      events.append(event)
    print('KEYBOARD_HANDOFF '+json.dumps(event), flush=True)


def run(args):
  import mujoco
  from mjlab.viewer import NativeMujocoViewer
  from keyboard_x11 import X11Keyboard
  keyboard = env = None
  try:
    keyboard = X11Keyboard()
    env, wrapped, policy, paths = load_chain(args)
    term = env.command_manager.get_term('twist')
    inputs = keyboard_inputs(env)

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
        super().reset_environment()
        inputs.sync()
        self.last_input = time.monotonic()

      def _execute_step(self):
        if term.keyboard_timed_out.any():
          self.pause()
          return False
        ok = super()._execute_step()
        inputs.sync()
        print_events(term)
        if term.keyboard_timed_out.any():
          self.pause()
          return False
        return ok

      def tick(self):
        now = time.monotonic()
        if now-self.last_input >= .01:
          inputs.update(keyboard.read(), now-self.last_input,
            focused=keyboard.focused, paused=self._is_paused)
          self.last_input = now
        return super().tick()

      def _set_status_overlay(self, viewer):
        status = self.get_status()
        state = term.status(self.env_idx)
        labels = ('Simulation\nActive policy\nTransition model\nInput\nRequested FR dz\nExecuted FR dz\n'
          'Trigger FR dz\nTrigger hold\nReference progress\nStanding handoff hold\n'
          'Strict endpoint hold\nWorld feet RMS (diagnostic)\nFR error (diagnostic)\nUnmet standing gates\nReset')
        values = (f"{'PAUSED' if status.paused else 'RUNNING'}\n{state['state']}\n"
          f"transition_t/{paths['transition_t'].name}\n{'ACTIVE' if keyboard.focused else 'INACTIVE'}\n"
          f"{state['requested_dz']:.3f}m\n{state['executed_dz']:.3f}m\n{state['trigger_height']:.3f}m\n"
          f"{state['trigger_hold']:.2f}/0.10s\n{state['progress']:.0%}\n{state['standing_hold']:.2f}/1.00s\n"
          f"{'PASSED' if state['strict_held'] else format(state['strict_hold'], '.2f')+'/1.00s'}\n"
          f"{state['feet_rms']:.3f}m / 0.060m\n{state['fr_error']:.3f}m\n"
          f"{', '.join(state['missing']) or 'none'}\nBackspace")
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
          mujoco.mjtGridPos.mjGRID_TOPLEFT.value, labels, values))

    print('[INFO] TRIPOD -> TRANSITION (BeyondMimic actor75) -> BIPED', flush=True)
    for name, path in paths.items():
      print(f'[INFO] {name}: {path}', flush=True)
    limiter = env.action_manager.get_term('joint_pos').target_limiter
    print('[INFO] Target limiter: '+json.dumps(limiter.contract()), flush=True)
    print(f'[INFO] Trigger FR dz={term.trigger_height:.3f}m after 0.10s tracking hold.', flush=True)
    print('[INFO] Physical state and action history persist across handoffs; the reference aligns XY/yaw only.', flush=True)
    print('[INFO] Standing contact/posture gates control handoff; world feet RMS is a separate strict endpoint diagnostic.', flush=True)
    print('[INFO] WASD: velocity; arrows: FR XY; Q/E: height; R: default; Backspace: reset; Space: pause.', flush=True)
    print('[INFO] Transition input is frozen; a 10s transition timeout pauses without forcing biped handoff.', flush=True)
    KeyboardViewer().run()
  finally:
    if keyboard:
      keyboard.close()
    if env:
      env.close()


def run_scripted(args, output, *, seconds=18., video=None):
  """Exercise the same keyboard/FSM with a one-second settle then held Q."""
  import math
  import torch
  from src.tasks.transition_t.keyboard import TRIPOD, BIPED
  env, wrapped, policy, paths = load_chain(args, render_mode='rgb_array' if video else None)
  writer = None
  try:
    term = env.command_manager.get_term('twist')
    motion = env.command_manager.get_term('motion')
    action = env.action_manager.get_term('joint_pos')
    inputs = keyboard_inputs(env)
    events, continuity, samples = [], [], []
    original_start = motion.start_from_current_state

    def audited_start(ids):
      tensors = (env.sim.data.qpos, env.sim.data.qvel, action.raw_action,
        action.previous_action, action.delay_steps, action.motor_offsets,
        action.target_limiter.position, action.target_limiter.velocity)
      before = [value.clone() for value in tensors]
      original_start(ids)
      record = {name: float((old-value.clone()).abs().max()) for name, old, value in zip(
        ('qpos', 'qvel', 'raw_action', 'previous_action', 'delay_steps', 'motor_offsets',
         'limiter_position', 'limiter_velocity'), before, tensors)}
      continuity.append(record)
      assert all(value == 0. for value in record.values()), record

    motion.start_from_current_state = audited_start
    obs = wrapped.get_observations()
    biped_seconds = torch.zeros(env.num_envs, device=args.device, dtype=torch.float64)
    failure = False
    if video:
      import imageio.v2 as imageio
      Path(video).parent.mkdir(parents=True, exist_ok=True)
      writer = imageio.get_writer(video, fps=25, codec='libx264', quality=8, ffmpeg_log_level='error')
    with torch.inference_mode():
      for step in range(math.ceil(seconds/env.step_dt)):
        keys = {'q'} if step*env.step_dt >= 1. else set()
        # The controller ignores target edits during transition. Stop Q in BIPED.
        if not (term.owner == TRIPOD).any():
          keys = set()
        inputs.update(keys, env.step_dt, focused=True, paused=False)
        if writer and step % 2 == 0:
          from PIL import Image, ImageDraw
          import numpy as np
          picture = Image.fromarray(env.render())
          draw = ImageDraw.Draw(picture)
          draw.rectangle((8, 8, 952, 64), fill=(12, 12, 12))
          draw.text((18, 18), f"Actual teacher chain | {term.status()['state']} | t={step*env.step_dt:.2f}s | transition_t/{paths['transition_t'].name}", fill='white')
          writer.append_data(np.asarray(picture))
        executed_owner = term.owner.clone()
        obs, _, done, _ = wrapped.step(policy(obs))
        inputs.sync()
        print_events(term, events)
        biped_seconds += env.step_dt*((executed_owner == BIPED) & ~done.bool())
        if step % 10 == 0:
          samples.append(dict(simulation_s=(step+1)*env.step_dt, **term.status()))
        if done.any():
          failure = True
          break
        if term.keyboard_timed_out.any() or (biped_seconds >= 3.-1e-9).all():
          break
    report = dict(device=args.device, num_envs=env.num_envs, simulated_s=(step+1)*env.step_dt,
      target_limiter=action.target_limiter.contract(),
      legacy_limiter_experiment=getattr(args, 'allow_legacy_transition_limiter', False),
      paths={name: str(path) for name, path in paths.items()}, trigger_height=term.trigger_height,
      motion_sha256=motion.reference.sha256, physical_failure_or_tracking_reset=failure,
      biped_seconds=biped_seconds.cpu().tolist(), final_status=[term.status(i) for i in range(env.num_envs)],
      continuity=continuity, events=events, samples=samples)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('SCRIPTED_HANDOFF_RESULT '+json.dumps({key: value for key, value in report.items() if key not in ('events', 'samples')}), flush=True)
    return report
  finally:
    if writer:
      writer.close()
    env.close()
