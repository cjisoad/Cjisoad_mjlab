"""Bounded, action-only runtime controls never trust model output blindly."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.agentic_robot.runtime import AgenticRuntime, validate_command, duration_steps, failure_checks


def test_validate_command_requires_finite_velocity_position_and_bounded_time():
    result = validate_command({'cmd_v': [0.1, 0, 0], 'cmd_p': [0, .02, .08], 't': .21})
    assert result == ([.1, 0., 0.], [0., .02, .08], .21)
    for invalid in (
        {'cmd_v': [np.nan, 0, 0], 'cmd_p': [0, 0, 0], 't': .1},
        {'cmd_v': [0, 0], 'cmd_p': [0, 0, 0], 't': .1},
        {'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': 1.01},
        {'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': 0},
    ):
        with pytest.raises(ValueError):
            validate_command(invalid)


@pytest.mark.parametrize(('seconds', 'dt', 'steps'), [(.21, .02, 11), (1., .005, 200), (.1, .1, 1)])
def test_duration_rounds_to_nearest_control_boundary_without_exceeding_one_second(seconds, dt, steps):
    assert duration_steps(seconds, dt) == steps


def test_duration_rejects_policy_step_smaller_than_one_policy_interval():
    with pytest.raises(ValueError, match='at least one control step'):
        duration_steps(.001, .02)


@pytest.mark.parametrize(('owner', 'expected'), [
    (0, {'physical_failure'}),
    (2, {'physical_failure', 'anchor_pos', 'anchor_ori'}),
    (3, {'physical_failure'}),
    (4, {'physical_failure', 'anchor_pos', 'anchor_ori'}),
])
def test_tracking_failures_only_apply_while_transition_policy_owns_robot(owner, expected):
    from src.tasks.transition_t import terminations

    checks = failure_checks(owner, terminations)
    assert set(checks) == expected


def idle_runtime(owner):
    """Use deterministic policy endpoints without loading teacher checkpoints."""
    import threading
    import queue
    import torch
    runtime = object.__new__(AgenticRuntime)
    runtime.lock = threading.RLock()
    runtime.term = SimpleNamespace(owner=torch.tensor([owner]))
    runtime.done_reason = None
    runtime.native_viewer = None
    runtime.observation = None
    runtime.action_interrupt = threading.Event()
    runtime._scene_inputs = queue.SimpleQueue()
    runtime.env = SimpleNamespace(step_dt=.02, cfg=SimpleNamespace(decimation=4),
        physics_dt=.005, scene=SimpleNamespace(update=lambda **kw: None))
    runtime.adapter = SimpleNamespace(data=SimpleNamespace(time=0.), sync=lambda: None)
    runtime._publish_state = lambda: None
    runtime.scene_steps = 0
    runtime.policy_steps = 0
    runtime.original_sim_step = lambda: None

    def step_scene(step):
        runtime.scene_steps += 1
        runtime.adapter.data.time += .005
    runtime.adapter.step_scene = step_scene

    def policy_step():
        runtime.policy_steps += 1
        runtime.adapter.data.time += .02
        if runtime.policy_steps == 3:
            runtime.term.owner[0] = 3 if owner == 2 else 0
    runtime._policy_step = policy_step
    return runtime


@pytest.mark.parametrize(('owner', 'endpoint'), [(2, 3), (4, 0)])
def test_idle_fsm_finishes_then_holds_robot_and_keeps_scene_running(owner, endpoint):
    runtime = idle_runtime(owner)
    for _ in range(5):
        runtime._idle_step()
    assert runtime.term.owner.item() == endpoint
    assert runtime.policy_steps == 3
    assert runtime.scene_steps == 8
    assert runtime.adapter.data.time == pytest.approx(.1)


@pytest.mark.parametrize('stopped', ['failure', 'cancelled'])
def test_stopped_transition_holds_robot_but_scene_keeps_running(stopped):
    runtime = idle_runtime(2)
    if stopped == 'failure':
        runtime.done_reason = 'physical_failure'
    else:
        runtime.action_interrupt.set()
    runtime._idle_step()
    assert runtime.policy_steps == 0
    assert runtime.scene_steps == 4


def test_background_worker_stops_before_environment_teardown():
    import threading
    runtime = idle_runtime(0)
    advanced = threading.Event()
    original_idle = runtime._idle_step
    def idle():
        original_idle()
        advanced.set()
    runtime._idle_step = idle
    with runtime.lock:
        runtime.start_background()
        worker = runtime.background_thread
        runtime.start_background()
        assert runtime.background_thread is worker
        # Requests hold this same lock, so the worker must skip physics.
        assert not advanced.wait(.05)
        assert runtime.scene_steps == 0
    assert advanced.wait(2.)
    def close_env():
        assert not worker.is_alive()
    runtime.env.close = close_env
    runtime.close()
    assert not worker.is_alive()


def test_autonomous_transition_uses_original_policy_and_stops_on_failure(monkeypatch):
    import torch
    from src.tasks.transition_t import terminations
    runtime = idle_runtime(2)
    del runtime._policy_step
    runtime.obs = 'previous observation'
    runtime.policy = lambda obs: 'policy action' if obs == runtime.obs else None
    steps = []
    def step(action):
        steps.append(action)
        return 'next observation', None, None, None
    runtime.wrapped = SimpleNamespace(step=step)
    monkeypatch.setattr(terminations, 'physical_failure', lambda env: torch.tensor([True]))
    monkeypatch.setattr(terminations, 'tracking_height_failure', lambda env: torch.tensor([False]))
    monkeypatch.setattr(terminations, 'tracking_orientation_failure', lambda env: torch.tensor([False]))
    runtime._idle_step()
    assert steps == ['policy action']
    assert runtime.obs == 'next observation'
    assert runtime.done_reason == 'physical_failure'
    runtime._idle_step()
    assert steps == ['policy action']
    assert runtime.scene_steps == 4


def test_reset_can_clear_sensor_buffers_created_by_inference_steps():
    import torch
    runtime = idle_runtime(0)
    with torch.inference_mode():
        buffer = torch.ones(1)
    runtime.wrapped = SimpleNamespace(reset=lambda: buffer.zero_(), get_observations=lambda: {})
    runtime.adapter.reset = lambda: None
    runtime.policy = SimpleNamespace(reset=lambda: None)
    runtime.observe = lambda: {'reset': True}
    runtime.action_interrupt.set()
    assert runtime.reset() == {'reset': True}
    assert buffer.item() == 0.
    assert not runtime.action_interrupt.is_set()


def test_native_button_queue_is_consumed_by_scene_without_submitting_robot_action():
    runtime = idle_runtime(0)
    pressed = []
    runtime.adapter.controller = SimpleNamespace(mouse_press=pressed.append)
    runtime.queue_button_press('light_f1')
    runtime.queue_button_press('call_up_f1')
    assert pressed == []
    runtime._idle_step()
    assert pressed == ['light_f1', 'call_up_f1']
    assert runtime.policy_steps == 0
    assert runtime.scene_steps == 4
    runtime._idle_step()
    assert pressed == ['light_f1', 'call_up_f1']
    with pytest.raises(ValueError):
        runtime.queue_button_press('unknown')
