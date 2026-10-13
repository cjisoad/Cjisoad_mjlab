"""One serial executor consumes plans produced by a persistent model worker."""
from copy import deepcopy
import math
import threading
import time
import uuid

import numpy as np

from ..runtime import validate_command, duration_steps
from ..observation.state import world_to_command

ACTIVE = {'waiting_for_plan', 'executing', 'stopping'}


class AsyncPipeline:
    def __init__(self, runtime):
        self.runtime = runtime
        self.condition = threading.Condition(threading.RLock())
        self.thread = None
        self.state = 'idle'
        self.task = None
        self.pending = None
        self.current = None
        self.index = 0
        self.last_request_index = -1
        self.executed_actions = 0
        self.events = []
        self.reason = None

    @property
    def active(self):
        with self.condition:
            return self.state in ACTIVE

    def _record(self, kind, **values):
        self.events.append({'type': kind, 'wall_ns': time.time_ns(), **values})
        self.events = self.events[-1000:]

    def start(self, value):
        if set(value) != {'objective', 'max_actions', 'position_bounds'}:
            raise ValueError('start requires objective, max_actions, position_bounds')
        if not isinstance(value['objective'], str) or not 0 < len(value['objective']) <= 4000:
            raise ValueError('objective must be a nonempty short string')
        count = value['max_actions']
        if type(count) is not int or not 1 <= count <= 500:
            raise ValueError('max_actions must be an integer in [1, 500]')
        bounds = np.asarray(value['position_bounds'], dtype=float)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError('position_bounds requires three finite [min,max] pairs')
        guard = getattr(self.runtime, 'lock', None)
        if guard is not None and not guard.acquire(blocking=False):
            raise ValueError('physical action in progress; wait before starting pipeline')
        try:
            return self._start_locked(value)
        finally:
            if guard is not None:
                guard.release()

    def _start_locked(self, value):
        with self.condition:
            if self.active or (self.thread and self.thread.is_alive()):
                raise ValueError('pipeline already owns execution')
            self.task = {'task_id': uuid.uuid4().hex, **deepcopy(value)}
            self.pending = self.current = None
            self.index = self.executed_actions = 0
            self.last_request_index = -1
            self.reason = None
            self.events = []
            self.runtime.action_interrupt.clear()
            self.state = 'waiting_for_plan'
            self._record('task_started', task_id=self.task['task_id'])
            self.thread = threading.Thread(target=self._run, name='robot-plan-executor', daemon=True)
            self.thread.start()
            return self.status()

    def observe(self, value):
        if set(value) - {'after_snapshot_id', 'after_frame_seq', 'wait_s'}:
            raise ValueError('unexpected planning observation field')
        after = value.get('after_snapshot_id')
        after_frame = value.get('after_frame_seq')
        wait = float(value.get('wait_s', 0))
        if (after is not None and (type(after) is not int or after < 0)) or not math.isfinite(wait) or not 0 <= wait <= 15:
            raise ValueError('invalid snapshot version or wait_s (0..15)')
        if after_frame is not None and (type(after_frame) is not int or after_frame < -1):
            raise ValueError('after_frame_seq must be an integer >= -1')
        with self.condition:
            if not self.active or self.state == 'stopping':
                raise ValueError('no active planning task')
            task = deepcopy(self.task)
        request = self.runtime.snapshots.request(task['task_id'], after, wait, after_frame)
        with self.condition:
            if not self.active or self.state == 'stopping' or self.task['task_id'] != task['task_id']:
                raise ValueError('planning task changed or replaced while waiting for observation')
            request.pop('acked')
            request['task'] = task
            request['execution'] = self.status()
            self._record('observation_delivered', request_id=request['request_id'],
                source_snapshot_id=request['observation']['snapshot_id'],
                source_sim_time=request['observation']['sim_time'])
        return request

    def acknowledge(self, value):
        if set(value) != {'request_id', 'frame_ids'}:
            raise ValueError('ack requires request_id and frame_ids')
        self.runtime.snapshots.acknowledge(value['request_id'], value['frame_ids'])
        return {'acknowledged': True}

    def _validate_current(self, plan, snapshot):
        source = plan['source']
        if snapshot['episode_id'] != source['episode_id']:
            raise ValueError('episode changed')
        if snapshot['sim_time'] >= plan['expires_sim_time'] - 1e-8:
            raise ValueError('plan expired in simulation time')
        age = (time.time_ns() - plan['delivered_wall_ns']) / 1e9
        if age > 120:
            raise ValueError('plan expired in wall time')
        if snapshot['robot']['fsm']['owner'] != plan['expected_owner']:
            raise ValueError('FSM owner changed')
        position = np.asarray(snapshot['robot']['position_w'])
        bounds = np.asarray(self.task['position_bounds'])
        if not np.all((position >= bounds[:, 0]) & (position <= bounds[:, 1])):
            raise ValueError('robot outside task position bounds')
        if np.linalg.norm(position - np.asarray(source['robot']['position_w'])) > 2.:
            raise ValueError('robot moved too far from source observation')
        a = np.asarray(snapshot['robot']['anchor']['R_world_from_anchor'])[:, 0]
        b = np.asarray(source['robot']['anchor']['R_world_from_anchor'])[:, 0]
        if np.dot(a, b) < math.cos(.5):
            raise ValueError('anchor heading changed too far')

    def submit(self, value):
        required = {'request_id', 'expected_owner', 'expires_sim_time', 'terminal', 'commands'}
        if set(value) != required or type(value['terminal']) is not bool:
            raise ValueError('plan requires request_id, expected_owner, expires_sim_time, terminal, commands')
        request = self.runtime.snapshots.get_request(value['request_id'])
        if not request['acked']:
            raise ValueError('acknowledge delivered images before submitting plan')
        commands = value['commands']
        if not isinstance(commands, list) or not 1 <= len(commands) <= 5:
            raise ValueError('plan must contain one to five commands')
        for command in commands:
            if set(command) == {'cmd_v', 'foot_target_w', 't'}:
                p = np.asarray(command['foot_target_w'], dtype=float)
                if p.shape != (3,) or not np.isfinite(p).all():
                    raise ValueError('foot_target_w must be three finite values')
                checked = {**command, 'cmd_p': [0, 0, 0]}
                del checked['foot_target_w']
            else:
                checked = command
            velocity, position, duration = validate_command(checked)
            duration_steps(duration, self.runtime.env.step_dt)
            if any(abs(p) > 1e-9 for p in position):
                raise ValueError('use foot_target_w for nonzero foot targets; old anchor offsets are invalid')
            if abs(velocity[0]) > .6 or abs(velocity[1]) > .6 or abs(velocity[2]) > 1.:
                raise ValueError('experimental velocity limit exceeded')
        if value['terminal'] and any(abs(v) > 1e-9 for v in commands[-1]['cmd_v']):
            raise ValueError('terminal plan must end with zero velocity')
        expiry = float(value['expires_sim_time'])
        source = request['observation']
        if not math.isfinite(expiry) or expiry > source['sim_time'] + 6.:
            raise ValueError('expiry may extend at most six simulation seconds from source')
        accepted = {**deepcopy(value), 'source': source,
            'request_index': request['request_index'], 'delivered_wall_ns': request['delivered_wall_ns']}
        with self.condition:
            if not self.active or self.state == 'stopping' or request['task_id'] != self.task['task_id']:
                raise ValueError('task stopped or replaced')
            if request['request_index'] <= self.last_request_index:
                raise ValueError('older or superseded planning request')
            self._validate_current(accepted, self.runtime.snapshots.read())
            self.last_request_index = request['request_index']
            self.pending = accepted
            self._record('plan_accepted', request_id=value['request_id'], commands=len(commands),
                observation_to_plan_s=(time.time_ns() - request['delivered_wall_ns']) / 1e9,
                source_snapshot_id=source['snapshot_id'])
            self.condition.notify_all()
            return {'accepted': True, 'request_id': value['request_id']}

    def on_snapshot(self, snapshot):
        # Called by the physics owner; cancellation never waits for action lock.
        with self.condition:
            if self.state == 'executing':
                p = np.asarray(snapshot['robot']['position_w'])
                bounds = np.asarray(self.task['position_bounds'])
                if not np.all((p >= bounds[:, 0]) & (p <= bounds[:, 1])):
                    self.reason = 'position_bounds'
                    self.state = 'stopping'
                    self.runtime.action_interrupt.set()
                    self.condition.notify_all()

    def _run(self):
        try:
            while True:
                with self.condition:
                    if self.state == 'stopping':
                        self.state = 'stopped'
                        break
                    if self.pending is not None:
                        self.current, self.pending = self.pending, None
                        self.index = 0
                    if self.current is None:
                        self.state = 'waiting_for_plan'
                        self.condition.notify_all()
                        self.condition.wait()
                        continue
                    snapshot = self.runtime.snapshots.read()
                    try:
                        self._validate_current(self.current, snapshot)
                    except ValueError as exc:
                        self._record('plan_invalidated', reason=str(exc))
                        self.current = None
                        continue
                    command = deepcopy(self.current['commands'][self.index])
                    actual_duration = duration_steps(float(command['t']), self.runtime.env.step_dt) * self.runtime.env.step_dt
                    if snapshot['sim_time'] + actual_duration > self.current['expires_sim_time'] + 1e-8:
                        self._record('plan_invalidated', reason='command would outlive plan expiry')
                        self.current = None
                        continue
                    if 'foot_target_w' in command:
                        command['cmd_p'] = world_to_command(command.pop('foot_target_w'),
                            snapshot['robot']['anchor']).tolist()
                    self.state = 'executing'
                    request_id = self.current['request_id']
                    self._record('action_dispatched', request_id=request_id, step=self.index + 1,
                        command=command, source_snapshot_id=snapshot['snapshot_id'])
                result = self.runtime.action(command)
                with self.condition:
                    self.executed_actions += 1
                    self._record('action_returned', request_id=request_id,
                        executed_duration=result['executed_duration'],
                        stop_reason=result['stop_reason'], events=result['events'])
                    if self.state == 'stopping' or result['stop_reason'] == 'cancelled':
                        self.state = 'stopped'
                        break
                    if result['stop_reason']:
                        self.reason = result['stop_reason']
                        self.state = 'failed'
                        break
                    if not result['command_accepted'] and self.current['expected_owner'] not in ('RISE', 'DOWN'):
                        self.reason = 'command_not_accepted'
                        self.state = 'failed'
                        break
                    self.index += 1
                    finished = self.index == len(self.current['commands'])
                    if self.executed_actions >= self.task['max_actions'] or (finished and self.current['terminal'] and self.pending is None):
                        self.reason = 'max_actions' if self.executed_actions >= self.task['max_actions'] else 'terminal_plan'
                        self.state = 'completed'
                        break
                    if finished:
                        self.current = None
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                # An action may have executed: never replay on transport/error.
                self.reason = f'{type(exc).__name__}: {exc}'
                self.state = 'failed'
        finally:
            with self.condition:
                self.runtime.action_interrupt.clear()
                self.pending = self.current = None
                self._record('executor_finished', state=self.state, reason=self.reason)
                self.condition.notify_all()

    def status(self):
        with self.condition:
            return {'state': self.state, 'task': deepcopy(self.task), 'reason': self.reason,
                'executed_actions': self.executed_actions,
                'remaining_current_commands': 0 if self.current is None else len(self.current['commands']) - self.index,
                'pending_request_id': None if self.pending is None else self.pending['request_id'],
                'events': deepcopy(self.events)}

    def stop(self):
        with self.condition:
            if self.active:
                self.reason = 'requested_stop'
                self.state = 'stopping'
                self.pending = None
                self.runtime.action_interrupt.set()
                self.condition.notify_all()
            return self.status()

    def close(self):
        self.stop()
        if self.thread is not None:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError('executor did not stop; do not close its simulation underneath it')
