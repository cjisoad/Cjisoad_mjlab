"""Bounded commands with a live scene and autonomous FSM transitions."""
from __future__ import annotations

import math
import queue
import threading
import time
import uuid

import numpy as np
import torch

from src.tasks.transition_t import terminations
from .elevator_adapter import ElevatorAdapter
from .observation.controls import BUTTON_NAMES

QUADRUPED, RISE, BIPED, DOWN = 0, 2, 3, 4
OWNER_NAMES = {QUADRUPED: 'QUADRUPED', RISE: 'RISE', BIPED: 'BIPED', DOWN: 'DOWN'}


def failure_checks(owner_id, terminations_module):
    checks = {'physical_failure': terminations_module.physical_failure}
    if owner_id in (RISE, DOWN):
        checks.update(anchor_pos=terminations_module.tracking_height_failure,
                      anchor_ori=terminations_module.tracking_orientation_failure)
    return checks


def validate_command(command):
    if not isinstance(command, dict) or set(command) != {'cmd_v', 'cmd_p', 't'}:
        raise ValueError('expected exactly cmd_v, cmd_p, and t')
    velocity = np.asarray(command['cmd_v'], dtype=np.float64)
    position = np.asarray(command['cmd_p'], dtype=np.float64)
    duration = float(command['t'])
    if velocity.shape != (3,) or position.shape != (3,):
        raise ValueError('cmd_v and cmd_p must each have three values')
    if not np.isfinite(velocity).all() or not np.isfinite(position).all():
        raise ValueError('cmd_v and cmd_p must be finite')
    if not math.isfinite(duration) or not 0 < duration <= 1:
        raise ValueError('t must be in (0, 1] seconds')
    return velocity.tolist(), position.tolist(), duration


def duration_steps(seconds, step_dt):
    if not (math.isfinite(seconds) and math.isfinite(step_dt) and step_dt > 0):
        raise ValueError('duration and control step must be finite')
    steps = int(math.floor(seconds / step_dt + .5 + 1e-10))
    if steps < 1:
        raise ValueError('duration must be at least one control step')
    return min(steps, max(1, int(math.floor(1. / step_dt + 1e-10))))


class AgenticRuntime:
    def __init__(self, args, *, observation_service=None):
        import importlib
        import sys
        from pathlib import Path
        script_dir = Path(__file__).resolve().parents[2] / 'scripts'
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        fsm = importlib.import_module('keyboard_transition_fsm')
        from src.agentic_robot.scene import configure_elevator
        self.fsm_module = fsm
        self.env, self.wrapped, self.action_term, self.term, rise, down, paths = fsm.load_full_chain(
            args, configure_env=configure_elevator)
        self.paths = paths
        self.policy = fsm.FullChainPolicy(self.term, rise, down)
        self.lock = threading.RLock()
        self.episode_id = str(uuid.uuid4())
        self.done_reason = None
        self.native_viewer = None
        self.action_interrupt = threading.Event()
        self.background_thread = None
        self._background_stop = threading.Event()
        self._scene_inputs = queue.SimpleQueue()
        self.adapter = ElevatorAdapter(self.env.sim)
        self.adapter.reset()
        self.observation = observation_service
        from .observation.controls import ControlRegistry
        from .observation.state import StateReader
        model = self.env.sim.mj_model
        self.state_reader = StateReader(model,
            __import__('mujoco').mj_name2id(model, __import__('mujoco').mjtObj.mjOBJ_BODY, 'robot/base_link'),
            __import__('mujoco').mj_name2id(model, __import__('mujoco').mjtObj.mjOBJ_SITE, 'robot/FR'),
            self.term.zero.detach().cpu().numpy())
        self.control_registry = ControlRegistry(model)
        if self.observation:
            self.observation.reset(self.episode_id, self.adapter.data,
                                   fsm=self._fsm_state(command_accepted=None))
        self.original_sim_step = self.env.sim.step

        def integrated_step():
            self.adapter.before_step()
            self.original_sim_step()
            self.adapter.after_step()
            if self.observation:
                self.observation.sample(self.adapter.data,
                    fsm=self._fsm_state(command_accepted=None))

        self.env.sim.step = integrated_step
        self.obs = self.wrapped.get_observations()

    def _fsm_state(self, *, command_accepted):
        owner_id = int(self.term.owner[0].item())
        return {'owner': OWNER_NAMES.get(owner_id, f'UNKNOWN_{owner_id}'),
                'owner_id': owner_id, 'command_accepted': command_accepted,
                'support_phase': int(self.term.phase[0].item()),
                'support_blend': float(self.term.blend[0].item())}

    def _publish_state(self):
        """Optional snapshot hook; ordinary synchronous operation has no work."""

    def _publish_current(self):
        if self.native_viewer is not None:
            self.native_viewer.publish(self.adapter.data)
        self._publish_state()

    def _policy_step(self):
        """One teacher/FSM step, shared by requests and autonomous transitions."""
        self._drain_scene_inputs()
        policy_action = self.policy(self.obs)
        self.obs, *_ = self.wrapped.step(policy_action)
        self.adapter.sync()
        owner_id = int(self.term.owner[0].item())
        reasons = [name for name, fn in failure_checks(owner_id, terminations).items()
                   if bool(fn(self.env)[0].item())]
        if reasons:
            self.done_reason = ','.join(reasons)
        self._publish_current()

    def _idle_step(self):
        """Caller holds the runtime lock; manual owners freeze only the robot."""
        self._drain_scene_inputs()
        owner = int(self.term.owner[0].item())
        if owner in (RISE, DOWN) and not self.done_reason and not self.action_interrupt.is_set():
            self._policy_step()
            return
        # Bypass env.step: command timers, policy history and robot actions
        # stay held while elevator/buttons/doors continue physical integration.
        for _ in range(self.env.cfg.decimation):
            self.adapter.step_scene(self.original_sim_step)
            self.env.scene.update(dt=self.env.physics_dt)
            if self.observation is not None:
                self.observation.sample(self.adapter.data,
                    fsm=self._fsm_state(command_accepted=None))
        self._publish_current()

    def queue_button_press(self, button):
        """Nonblocking GUI input; the physics owner applies the plunger force."""
        if button not in BUTTON_NAMES:
            raise ValueError(f'unknown button: {button}')
        self._scene_inputs.put(button)

    def _drain_scene_inputs(self):
        while True:
            try:
                button = self._scene_inputs.get_nowait()
            except queue.Empty:
                return
            self.adapter.controller.mouse_press(button)

    def start_background(self):
        """Start after observation/viewer setup; all physics uses one lock."""
        with self.lock:
            if getattr(self, 'background_thread', None) is not None:
                return
            self._background_stop = threading.Event()

            def run():
                with torch.inference_mode():
                    while not self._background_stop.is_set():
                        started = time.monotonic()
                        # Do not queue idle work behind an active command.
                        if self.lock.acquire(blocking=False):
                            try:
                                if not self._background_stop.is_set():
                                    self._idle_step()
                            except Exception as exc:
                                self.done_reason = f'background_error:{exc}'
                                self._background_stop.set()
                                import traceback
                                traceback.print_exc()
                            finally:
                                self.lock.release()
                        self._background_stop.wait(max(0., self.env.step_dt - (time.monotonic() - started)))

            self.background_thread = threading.Thread(target=run,
                name='elevator-idle', daemon=True)
            self.background_thread.start()

    def stop_background(self):
        worker = getattr(self, 'background_thread', None)
        if worker is not None:
            self._background_stop.set()
            worker.join()
            self.background_thread = None

    def observe(self, consumer_id='default', cursor=None, max_frames=2, replay_latest=False):
        with self.lock:
            self.adapter.sync()
            if self.observation is None:
                return {'schema_version': 1, 'episode_id': self.episode_id,
                        'snapshot_id': str(uuid.uuid4()),
                        'sim_time': float(self.adapter.data.time),
                        'robot': self.state_reader.read(self.adapter.data,
                            self._fsm_state(command_accepted=None)),
                        'controls': self.control_registry.read(self.adapter.data), 'frames': [],
                        'next_cursor': cursor, 'dropped_frame_count': 0,
                        'image_status': 'unavailable'}
            return self.observation.observe(self.adapter.data,
                fsm=self._fsm_state(command_accepted=None), consumer_id=consumer_id,
                cursor=cursor, max_frames=max_frames, replay_latest=replay_latest)

    def _observe_current(self):
        if self.observation is None:
            return {'schema_version': 1, 'episode_id': self.episode_id,
                    'sim_time': float(self.adapter.data.time),
                    'robot': self.state_reader.read(self.adapter.data,
                        self._fsm_state(command_accepted=None)),
                    'controls': self.control_registry.read(self.adapter.data), 'frames': [],
                    'image_status': 'unavailable'}
        return self.observation.observe(self.adapter.data,
            fsm=self._fsm_state(command_accepted=None), consumer_id='default', max_frames=2)

    def acknowledge(self, consumer_id, cursor):
        if self.observation is None:
            raise RuntimeError('observation service is unavailable')
        with self.lock:
            return self.observation.acknowledge(consumer_id, cursor)

    def resource(self, resource_id):
        if self.observation is None:
            raise KeyError('observation resources unavailable')
        return self.observation.resource(resource_id)

    def scene_image(self, camera=None):
        if self.observation is None:
            raise RuntimeError('observation service is unavailable')
        with self.lock:
            self.adapter.sync()
            return self.observation.render_scene(self.adapter.data, camera)

    def click_scene(self, camera, relx, rely):
        if self.observation is None:
            raise RuntimeError('observation service is unavailable')
        with self.lock:
            self.adapter.sync()
            button = self.observation.pick_scene_button(self.adapter.data, camera, relx, rely)
            if button is None:
                return {'button': None}
            return {'button': button, **self.press_button(button)}

    def press_button(self, button):
        if button not in BUTTON_NAMES:
            raise ValueError(f'unknown button: {button}')
        with self.lock:
            self.adapter.controller.mouse_press(button)
            return {'button': button, **self.action({
                'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': .4})}

    def action(self, command):
        velocity, position, duration = validate_command(command)
        with self.lock, torch.inference_mode():
            if self.done_reason:
                raise RuntimeError(f'episode stopped: {self.done_reason}')
            steps = duration_steps(duration, self.env.step_dt)
            owner_before = int(self.term.owner[0].item())
            accepted = owner_before in (QUADRUPED, BIPED)
            self.term.submit([0], [velocity + position], [any(abs(v) > 1e-9 for v in position)],
                             input_enabled=True)
            # Idle scene motion can change contacts under the held robot.
            self.env.sim.sense()
            self.obs = self.wrapped.get_observations()
            events_start = len(self.adapter.events)
            executed_steps = 0
            for _ in range(steps):
                if self.action_interrupt.is_set():
                    break
                self._policy_step()
                executed_steps += 1
                if self.done_reason:
                    break
            elapsed = float(self.adapter.data.time)
            status = self._fsm_state(command_accepted=accepted)
            if self.observation:
                packet = self.observation.observe(self.adapter.data, fsm=status,
                    consumer_id='default', max_frames=2)
            else:
                packet = self._observe_current()
            return {'episode_id': self.episode_id,
                    'sim_time': elapsed,
                    'executed_duration': executed_steps * self.env.step_dt,
                    'requested_duration': duration,
                    'cmd_v': velocity, 'cmd_p': position,
                    'steps': executed_steps,
                    'command_accepted': accepted,
                    'owner_before': OWNER_NAMES[owner_before],
                    'owner_after': status['owner'], 'stop_reason': self.done_reason or
                        ('cancelled' if self.action_interrupt.is_set() else None),
                    'events': self.adapter.events[events_start:],
                    'elevator': self.adapter.evaluation_state(), 'observation': packet}

    def reset(self):
        # Sensor buffers may have been allocated by inference-mode policy
        # steps; reset them in the same mode as request/background execution.
        with self.lock, torch.inference_mode():
            self.wrapped.reset()
            self.adapter.reset()
            self._scene_inputs = queue.SimpleQueue()
            self.policy.reset()
            self.episode_id = str(uuid.uuid4())
            self.done_reason = None
            self.obs = self.wrapped.get_observations()
            if self.native_viewer is not None:
                self.native_viewer.publish(self.adapter.data)
            if self.observation:
                self.observation.reset(self.episode_id, self.adapter.data,
                    fsm=self._fsm_state(command_accepted=None))
            self.action_interrupt.clear()
            self._publish_state()
            return self.observe()

    def close(self):
        self.stop_background()
        if self.native_viewer is not None:
            self.native_viewer.close()
            self.native_viewer = None
        if self.observation:
            self.observation.close()
        self.env.close()
