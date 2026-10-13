"""Asynchronous planning never gains a second physics owner."""
from copy import deepcopy
import threading
import time
from types import SimpleNamespace

import pytest


def packet(seq=0, sim_time=0., x=0., frames=None):
    return {'episode_id': 'episode', 'snapshot_id': seq, 'sim_time': sim_time,
        'robot': {'position_w': [x, 0., .3], 'fr_position_w': [.2, -.1, .0],
            'orientation_wxyz': [1., 0., 0., 0.],
            'anchor': {'origin_w': [x, 0., .3],
                'R_world_from_anchor': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                'fr_zero_anchor': [.2, -.1, -.3]},
            'fsm': {'owner': 'QUADRUPED'}},
        'controls': [], 'frames': frames or [], 'elevator': {'lamps': {'1': True}}}


class FakeRuntime:
    def __init__(self):
        from src.agentic_robot.experimental.snapshots import SnapshotStore
        self.snapshots = SnapshotStore()
        self.env = SimpleNamespace(step_dt=.02)
        self.current = packet()
        self.snapshots.publish(self.current)
        self.action_interrupt = threading.Event()
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls = []
        self.overlap = 0
        self.max_overlap = 0

    def action(self, command):
        self.overlap += 1
        self.max_overlap = max(self.max_overlap, self.overlap)
        self.started.set()
        assert self.release.wait(3)
        self.calls.append(deepcopy(command))
        if not self.action_interrupt.is_set():
            self.current['sim_time'] += command['t']
            self.current['robot']['position_w'][0] += command['cmd_v'][0] * command['t']
            self.current['robot']['anchor']['origin_w'][0] = self.current['robot']['position_w'][0]
        self.snapshots.publish(self.current)
        self.overlap -= 1
        return {'command_accepted': True, 'owner_after': 'QUADRUPED',
            'stop_reason': 'cancelled' if self.action_interrupt.is_set() else None,
            'events': [], 'executed_duration': command['t'], 'observation': self.current}


@pytest.fixture
def running():
    from src.agentic_robot.experimental.pipeline import AsyncPipeline
    runtime = FakeRuntime()
    pipeline = AsyncPipeline(runtime)
    pipeline.start({'objective': 'bounded test', 'max_actions': 10,
        'position_bounds': [[-1, 2], [-1, 1], [.1, .6]]})
    yield runtime, pipeline
    runtime.release.set()
    pipeline.close()


def observe_and_ack(pipeline):
    request = pipeline.observe({})
    pipeline.acknowledge({'request_id': request['request_id'],
        'frame_ids': [f['frame_id'] for f in request['observation']['frames']]})
    return request


def plan(request, commands=None, terminal=False):
    return {'request_id': request['request_id'], 'expected_owner': 'QUADRUPED',
        'expires_sim_time': request['observation']['sim_time'] + 6.,
        'terminal': terminal, 'commands': commands or [
            {'cmd_v': [.1, 0, 0], 'cmd_p': [0, 0, 0], 't': .1}]}


def wait_state(pipeline, states):
    deadline = time.monotonic() + 3
    with pipeline.condition:
        while pipeline.status()['state'] not in states:
            remaining = deadline - time.monotonic()
            assert remaining > 0, pipeline.status()
            pipeline.condition.wait(remaining)
    return pipeline.status()


def test_model_can_read_and_replan_while_executor_is_inside_action(running):
    runtime, pipeline = running
    runtime.release.clear()
    first = observe_and_ack(pipeline)
    pipeline.submit(plan(first, commands=[
        {'cmd_v': [.1, 0, 0], 'cmd_p': [0, 0, 0], 't': .1}] * 5))
    assert runtime.started.wait(2)
    runtime.current['sim_time'] = .04
    runtime.snapshots.publish(runtime.current)
    before = time.monotonic()
    second = observe_and_ack(pipeline)
    assert time.monotonic() - before < .2
    assert second['observation']['sim_time'] == .04
    # Replacement is accepted without waiting for the executing action.
    pipeline.submit(plan(second, commands=[
        {'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': .1}], terminal=True))
    runtime.release.set()
    status = wait_state(pipeline, {'completed'})
    assert len(runtime.calls) == 2
    assert runtime.max_overlap == 1
    assert status['executed_actions'] == 2


def test_expired_or_older_plan_never_replaces_newer_plan(running):
    runtime, pipeline = running
    old = observe_and_ack(pipeline)
    new = observe_and_ack(pipeline)
    runtime.release.clear()
    pipeline.submit(plan(new))
    assert runtime.started.wait(2)
    with pytest.raises(ValueError, match='older|superseded'):
        pipeline.submit(plan(old))
    runtime.current['sim_time'] = 7.
    runtime.snapshots.publish(runtime.current)
    third = observe_and_ack(pipeline)
    bad = plan(third)
    bad['expires_sim_time'] = 6.
    with pytest.raises(ValueError, match='expired'):
        pipeline.submit(bad)


def test_plan_requires_actual_image_ack_and_resources_are_pinned(running):
    runtime, pipeline = running
    frame = {'frame_id': 'episode:1', 'frame_seq': 1,
        'capture_sim_time': 0., 'resources': {'rgb': {'resource_id': 'a' * 32,
            'uri': '/resources/' + 'a' * 32, 'mime_type': 'image/png'}}}
    runtime.snapshots.publish(packet(frames=[frame]), {'a' * 32: (b'image', 'image/png')})
    request = pipeline.observe({})
    with pytest.raises(ValueError, match='acknowledge'):
        pipeline.submit(plan(request))
    runtime.snapshots.publish(packet(seq=2, sim_time=1.))
    assert runtime.snapshots.resource('a' * 32) == (b'image', 'image/png')
    with pytest.raises(ValueError, match='frame'):
        pipeline.acknowledge({'request_id': request['request_id'], 'frame_ids': []})
    pipeline.acknowledge({'request_id': request['request_id'], 'frame_ids': ['episode:1']})
    runtime.release.clear()
    pipeline.submit(plan(request))


def test_stop_interrupts_without_waiting_for_current_action(running):
    runtime, pipeline = running
    runtime.release.clear()
    request = observe_and_ack(pipeline)
    pipeline.submit(plan(request, commands=plan(request)['commands'] * 5))
    assert runtime.started.wait(2)
    before = time.monotonic()
    pipeline.stop()
    assert time.monotonic() - before < .2
    assert runtime.action_interrupt.is_set()
    runtime.release.set()
    wait_state(pipeline, {'stopped'})
    assert len(runtime.calls) == 1


def test_executor_converts_world_foot_target_using_latest_anchor(running):
    runtime, pipeline = running
    request = observe_and_ack(pipeline)
    runtime.current['robot']['anchor']['origin_w'][0] = .5
    runtime.snapshots.publish(runtime.current)
    pipeline.submit(plan(request, commands=[
        {'cmd_v': [0, 0, 0], 'foot_target_w': [.7, -.1, 0.], 't': .1}], terminal=True))
    wait_state(pipeline, {'completed'})
    assert runtime.calls[0]['cmd_p'] == pytest.approx([0, 0, 0])


def test_bounds_failure_truncates_queue(running):
    runtime, pipeline = running
    request = observe_and_ack(pipeline)
    runtime.current['robot']['position_w'][0] = 2.1
    runtime.snapshots.publish(runtime.current)
    with pytest.raises(ValueError, match='bounds'):
        pipeline.submit(plan(request))
    assert not runtime.calls


def test_idle_without_plan_never_advances_physics(running):
    runtime, pipeline = running
    assert pipeline.status()['state'] == 'waiting_for_plan'
    assert runtime.current['sim_time'] == 0
    assert not runtime.calls


def test_failed_action_is_not_replayed(running):
    runtime, pipeline = running
    request = observe_and_ack(pipeline)
    def fail_after_dispatch(command):
        runtime.calls.append(command)
        raise RuntimeError('execution uncertain')
    runtime.action = fail_after_dispatch
    pipeline.submit(plan(request))
    status = wait_state(pipeline, {'failed'})
    assert 'execution uncertain' in status['reason']
    assert len(runtime.calls) == 1


def test_start_does_not_wait_for_an_existing_physical_action():
    from src.agentic_robot.experimental.pipeline import AsyncPipeline
    runtime = FakeRuntime()
    runtime.lock = threading.RLock()
    holding, release = threading.Event(), threading.Event()
    def hold():
        with runtime.lock:
            holding.set()
            release.wait(3)
    thread = threading.Thread(target=hold)
    thread.start()
    assert holding.wait(1)
    pipeline = AsyncPipeline(runtime)
    try:
        with pytest.raises(ValueError, match='physical action in progress'):
            pipeline.start({'objective': 'test', 'max_actions': 1,
                'position_bounds': [[-1, 1], [-1, 1], [.1, .6]]})
        assert not pipeline.active
    finally:
        release.set()
        thread.join()
        pipeline.close()


def test_command_cannot_run_past_plan_expiry(running):
    runtime, pipeline = running
    request = observe_and_ack(pipeline)
    candidate = plan(request)
    candidate['expires_sim_time'] = .05
    pipeline.submit(candidate)
    status = wait_state(pipeline, {'waiting_for_plan'})
    # Wait for the executor to consume and reject the candidate, not the initial wait.
    deadline = time.monotonic() + 2
    with pipeline.condition:
        while not any(e['type'] == 'plan_invalidated' for e in pipeline.events):
            remaining = deadline - time.monotonic()
            assert remaining > 0
            pipeline.condition.wait(remaining)
    assert not runtime.calls


def test_quantized_control_duration_cannot_run_past_expiry(running):
    runtime, pipeline = running
    request = observe_and_ack(pipeline)
    candidate = plan(request, commands=[
        {'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': .035}])
    candidate['expires_sim_time'] = .035
    pipeline.submit(candidate)
    deadline = time.monotonic() + 2
    with pipeline.condition:
        while not any(e['type'] in ('plan_invalidated', 'action_returned') for e in pipeline.events):
            remaining = deadline - time.monotonic()
            assert remaining > 0
            pipeline.condition.wait(remaining)
    assert not runtime.calls


def test_observation_wait_cannot_return_a_replaced_task(running):
    runtime, pipeline = running
    old_task = pipeline.task['task_id']
    original = runtime.snapshots.request
    def switch_task_during_wait(task_id, *args):
        pipeline.stop()
        pipeline.thread.join(2)
        pipeline.start({'objective': 'new task', 'max_actions': 2,
            'position_bounds': [[-1, 2], [-1, 1], [.1, .6]]})
        return original(task_id, *args)
    runtime.snapshots.request = switch_task_during_wait
    with pytest.raises(ValueError, match='task.*changed|task.*replaced'):
        pipeline.observe({'wait_s': 1})
    assert pipeline.task['task_id'] != old_task


def test_button_ownership_is_checked_before_any_scene_mutation():
    from src.agentic_robot.experimental.runtime import ExperimentalRuntime
    runtime = object.__new__(ExperimentalRuntime)
    runtime.pipeline = SimpleNamespace(active=False, thread=None)
    mutations = []
    runtime.adapter = SimpleNamespace(controller=SimpleNamespace(mouse_press=mutations.append))
    class PipelineStartsBeforeLock:
        def __enter__(self):
            runtime.pipeline.active = True
        def __exit__(self, *args):
            pass
    runtime.lock = PipelineStartsBeforeLock()
    with pytest.raises(RuntimeError, match='pipeline owns actions'):
        runtime.press_button('light_f1')
    assert mutations == []


def test_planner_can_wait_for_new_rgbd_while_structured_snapshots_update(running):
    runtime, pipeline = running
    frame = {'frame_id': 'episode:0', 'frame_seq': 0, 'capture_sim_time': 0., 'resources': {}}
    runtime.snapshots.publish(packet(frames=[frame]))
    returned, result = threading.Event(), []
    def get_new_image():
        result.append(pipeline.observe({'after_frame_seq': 0, 'wait_s': 2}))
        returned.set()
    thread = threading.Thread(target=get_new_image)
    thread.start()
    runtime.snapshots.publish(packet(sim_time=.1, frames=[frame]))
    assert not returned.wait(.05)
    newer = {**frame, 'frame_id': 'episode:1', 'frame_seq': 1, 'capture_sim_time': .5}
    runtime.snapshots.publish(packet(sim_time=.52, frames=[frame, newer]))
    assert returned.wait(1)
    thread.join()
    assert result[0]['observation']['frames'][-1]['frame_seq'] == 1


@pytest.mark.parametrize('commands', [
    [{'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': 1.1}],
    [{'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, .4], 't': .1}],
    [{'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': .1}] * 6,
    [{'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': .001}],
])
def test_invalid_or_stale_anchor_commands_are_rejected(running, commands):
    _, pipeline = running
    request = observe_and_ack(pipeline)
    with pytest.raises(ValueError):
        pipeline.submit(plan(request, commands=commands))


def test_repository_skill_client_resolves_moved_checkout_from_discovery_link(monkeypatch, tmp_path):
    import runpy
    import shutil
    from pathlib import Path

    root = tmp_path / 'moved-checkout'
    helper = root / 'skills/agentic-elevator-async-control/scripts/client.py'
    helper.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / helper.relative_to(root)
    shutil.copy2(source, helper)
    client = root / 'scripts/agentic_pipeline_client.py'
    client.parent.mkdir()
    client.write_text('# transport entry point\n')
    discovery = root / '.agents/skills/agentic-elevator-async-control'
    discovery.parent.mkdir(parents=True)
    discovery.symlink_to('../../skills/agentic-elevator-async-control', target_is_directory=True)
    monkeypatch.delenv('UNITREE_AGENTIC_REPO', raising=False)
    monkeypatch.chdir(tmp_path)
    captured = []
    execute = runpy.run_path
    monkeypatch.setattr(runpy, 'run_path', lambda path, **kw: captured.append(Path(path)))
    execute(str(discovery / 'scripts/client.py'), run_name='skill_portability_probe')
    assert captured == [client]
