"""Native keyboard viewer for the three-teacher rise state machine."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


class KeyboardInputs:
  """Keep one manual controller per world, synchronized with physical resets."""
  def __init__(self,term,source_cfg,biped_cfg,biped_idle):
    from keyboard_teacher import TeacherKeyboardController,workspace_for_mode
    self.term=term
    self.source_cfg,self.biped_cfg=source_cfg,biped_cfg
    self.biped_idle=tuple(biped_idle)
    self.workspace_for_mode=workspace_for_mode
    self.controllers=[TeacherKeyboardController(workspace_for_mode('union')) for _ in range(term.num_envs)]
    self.revisions=[-1]*term.num_envs
    self.sync()

  def sync(self):
    from src.tasks.pedipulation_bridge_t.commands import WARM,VERIFY
    revisions=self.term.keyboard_revision.cpu().tolist()
    owners=self.term.owner.cpu().tolist()
    for world,(revision,state) in enumerate(zip(revisions,owners,strict=True)):
      if revision==self.revisions[world]: continue
      controller=self.controllers[world]
      biped=state==VERIFY
      cfg=self.biped_cfg if biped else self.source_cfg
      controller.workspace=self.workspace_for_mode('biped' if biped else 'union')
      controller.idle_target=self.biped_idle if biped else (0.,0.,0.)
      controller.vx_range=cfg.lin_vel_x
      controller.wz_range=getattr(cfg,'ang_vel_z',(-1.,1.))
      controller.reset()
      if state!=WARM:
        controller.target=tuple(self.term.keyboard_command[world,3:].cpu().tolist())
        controller.requested_target=controller.target
        controller.target_active=True
      self.revisions[world]=revision

  def update(self,keys,dt,*,focused,paused):
    from src.tasks.pedipulation_bridge_t.commands import WARM,VERIFY
    self.sync()
    for world,controller in enumerate(self.controllers):
      if int(self.term.owner[world]) in (WARM,VERIFY):
        controller.update(keys,dt,focused=focused,paused=paused)
    self.term.submit(list(range(len(self.controllers))),
      [controller.command for controller in self.controllers],
      [controller.target_active for controller in self.controllers],
      input_enabled=focused and not paused)
    self.term._env.observation_manager._obs_buffer=None


def run(args):
  bridge=args.bridge_checkpoint_file or ROOT/'outputs/bridge_tracking_v3_recovered_20261008/model_900.pt'
  from src.tasks.pedipulation_bridge_t.experts import DEFAULT_BIPED_CHECKPOINT
  biped=args.biped_checkpoint_file or DEFAULT_BIPED_CHECKPOINT
  paths={name:Path(path).expanduser().resolve() for name,path in (
    ('tripod',args.checkpoint_file),('transition',bridge),('biped',biped))}
  for name,path in paths.items():
    if not path.is_file(): raise FileNotFoundError(f'{name} checkpoint not found: {path}')
  os.environ.setdefault('OMP_NUM_THREADS','1')
  os.environ.setdefault('MPLCONFIGDIR','/tmp/bridge-mpl')
  os.environ.setdefault('XDG_CACHE_HOME','/tmp/bridge-cache')
  import torch
  import mujoco
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.viewer import NativeMujocoViewer
  from src.tasks.pedipulation_bridge_t.keyboard import KeyboardTeacher,keyboard_bridge_env_cfg
  from src.tasks.pedipulation_bridge_t.rl import BridgeOnPolicyRunner,bridge_ppo_runner_cfg
  from src.tasks.teacher_common.env_cfg import loco_pedipulation_teacher_env_cfg,pedipulation_teacher_env_cfg
  from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace
  from keyboard_x11 import X11Keyboard

  configure_torch_backends()
  torch.set_num_threads(1)
  keyboard=env=None
  try:
    keyboard=X11Keyboard()
    env=ManagerBasedRlEnv(keyboard_bridge_env_cfg(args.num_envs),device=args.device)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=10.)
    term=env.command_manager.get_term('twist')
    term.configure_trigger(args.transition_margin)
    runner=BridgeOnPolicyRunner(wrapped,asdict(bridge_ppo_runner_cfg()),device=args.device)
    # Validate the bridge's own frozen training pair, physics and reference.
    # The keyboard source actor is loaded separately and may have51 or54 inputs.
    runner.load(str(paths['transition']),map_location=args.device)
    action=env.action_manager.get_term('joint_pos')
    action.source_teacher=KeyboardTeacher(paths['tripod'],'loco_pedipulation_t',args.device)
    action.biped_teacher=KeyboardTeacher(paths['biped'],'pedipulation_t',args.device)
    policy=runner.get_inference_policy(device=args.device)
    inputs=KeyboardInputs(term,
      loco_pedipulation_teacher_env_cfg(play=True).commands['twist'],
      pedipulation_teacher_env_cfg(play=True).commands['twist'],
      build_foot_workspace().nominal_biped_offset)

    class KeyboardViewer(NativeMujocoViewer):
      def __init__(self):
        super().__init__(wrapped,policy)
        self.last_input=time.monotonic()

      def _safe_key_callback(self,key):
        if key==259: self.request_reset()
        elif key not in {ord(k) for k in 'WASDQER'}|{262,263,264,265}:
          super()._safe_key_callback(key)

      def reset_environment(self):
        super().reset_environment()
        inputs.sync()
        self.last_input=time.monotonic()

      def _execute_step(self):
        if term.keyboard_timed_out.any():
          self.pause()
          return False
        ok=super()._execute_step()
        inputs.sync()
        while term.keyboard_events:
          print('KEYBOARD_HANDOFF '+json.dumps(term.keyboard_events.pop(0)),flush=True)
        if term.keyboard_timed_out.any():
          self.pause()
          return False
        return ok

      def tick(self):
        now=time.monotonic()
        if now-self.last_input>=.01:
          inputs.update(keyboard.read(),now-self.last_input,
            focused=keyboard.focused,paused=self._is_paused)
          self.last_input=now
        return super().tick()

      def _set_status_overlay(self,viewer):
        status=self.get_status()
        state=term.status(self.env_idx)
        missing=', '.join(state['missing']) or 'none'
        labels=(f'Simulation\nWorld{self.env_idx} policy\nInput\nRequested FR dz\nExecuted FR dz\n'
          'Trigger FR dz\nTrigger hold\nRise progress\nStanding hold\nFR error (diagnostic)\nUnmet gates\nReset')
        values=(f"{'PAUSED' if status.paused else 'RUNNING'}\n{state['state']}\n"
          f"{'ACTIVE' if keyboard.focused else 'INACTIVE'}\n"
          f"{state['requested_dz']:.3f}m\n{state['executed_dz']:.3f}m\n"
          f"{state['trigger_height']:.3f}m\n{state['trigger_hold']:.2f}/0.10s\n"
          f"{state['progress']:.0%}\n{state['standing_hold']:.2f}/1.00s\n"
          f"{state['fr_error']:.3f}m\n{missing}\nBackspace")
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
          mujoco.mjtGridPos.mjGRID_TOPLEFT.value,labels,values))

    print('[INFO] Keyboard state machine: TRIPOD -> TRANSITION -> BIPED',flush=True)
    for name,path in paths.items(): print(f'[INFO] {name}: {path}',flush=True)
    print(f'[INFO] Trigger FR dz={term.trigger_height:.3f}m, tracking hold0.10s; standing hold1.00s.',flush=True)
    print('[INFO] WASD: velocity; arrows: FR XY; Q/E: height; R: current teacher default; Backspace: tripod reset; Space: pause.',flush=True)
    print('[INFO] Transition owns the command until standing. A10s timeout pauses with unmet gates displayed.',flush=True)
    print('[INFO] Biped handoff uses standing gates; FR tracking error is diagnostic only.',flush=True)
    KeyboardViewer().run()
  finally:
    if keyboard: keyboard.close()
    if env: env.close()
