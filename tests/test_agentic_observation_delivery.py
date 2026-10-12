import copy

import mujoco
import pytest

from test_agentic_observation_state import make_model


def service_fixture(tmp_path):
    from src.agentic_robot.observation import ObservationService
    model, data = make_model()
    service = ObservationService(model, tmp_path, fr_zero_anchor=[0, 0, 0], capture_images=False, max_buffer_frames=4, max_consumers=2)
    service.reset('first', data, fsm={})
    return service, model, data


def sample_at(service, model, data, time):
    data.time = time
    mujoco.mj_forward(model, data)
    service.sample(data, fsm={})


def test_delivery_retry_ack_two_one_zero_and_read_only(tmp_path):
    service, model, data = service_fixture(tmp_path)
    sample_at(service, model, data, .5)
    first = service.observe(data, fsm={}, consumer_id='agent')
    assert [f['capture_sim_time'] for f in first['frames']] == [0, .5]
    assert first['frames'][0]['image_status'] == 'disabled'
    qpos_before = data.qpos.copy()
    assert service.observe(data, fsm={}, consumer_id='agent')['frames'] == first['frames']
    with pytest.raises(ValueError, match='acknowledge'):
        service.observe(data, fsm={}, consumer_id='agent', cursor=first['next_cursor'])
    service.acknowledge('agent', first['next_cursor'])
    empty = service.observe(data, fsm={}, consumer_id='agent')
    assert empty['frames'] == [] and empty['image_status'] == 'no_new_frame'
    sample_at(service, model, data, 1.)
    one = service.observe(data, fsm={}, consumer_id='agent')
    assert [f['capture_sim_time'] for f in one['frames']] == [1.]
    assert data.time == 1. and (data.qpos == qpos_before).all()
    assert first['sim_time'] == .5
    service.close()


def test_slow_consumers_drop_count_and_reset_invalidates_cursor(tmp_path):
    service, model, data = service_fixture(tmp_path)
    for t in [.5, 1., 1.5, 2., 2.5]:
        sample_at(service, model, data, t)
    packet = service.observe(data, fsm={}, consumer_id='slow')
    assert [f['capture_sim_time'] for f in packet['frames']] == [2., 2.5]
    assert packet['dropped_frame_count'] == 4
    old_cursor = packet['next_cursor']
    data.time = 0
    service.reset('second', data, fsm={})
    with pytest.raises(ValueError, match='episode'):
        service.observe(data, fsm={}, cursor=old_cursor)
    with pytest.raises(ValueError, match='episode'):
        service.acknowledge('slow', old_cursor)
    assert [f['capture_sim_time'] for f in service.observe(data, fsm={})['frames']] == [0]


def test_ack_rejects_undelivered_and_malformed_cursor_and_bounds_consumers(tmp_path):
    service, model, data = service_fixture(tmp_path)
    packet = service.observe(data, fsm={}, consumer_id='one')
    bad = copy.deepcopy(packet['next_cursor'])
    bad['frame_seq'] += 10
    with pytest.raises(ValueError, match='delivered'):
        service.acknowledge('one', bad)
    with pytest.raises(ValueError):
        service.acknowledge('one', {'episode_id': 'first', 'frame_seq': 0.5})
    service.observe(data, fsm={}, consumer_id='two')
    with pytest.raises(ValueError, match='consumer'):
        service.observe(data, fsm={}, consumer_id='three')
    with pytest.raises(KeyError):
        service.resource('../../etc/passwd')


def test_scheduler_keeps_half_second_grid_with_float_roundoff(tmp_path):
    service, model, data = service_fixture(tmp_path)
    for i in range(1, 51):
        sample_at(service, model, data, i * .02 - 1e-15)
    packet = service.observe(data, fsm={})
    assert [f['capture_sim_time'] for f in packet['frames']] == pytest.approx([.5, 1.])
    assert packet['dropped_frame_count'] == 1
    data.time = .4
    with pytest.raises(ValueError, match='backwards'):
        service.sample(data, fsm={})


def test_returned_packets_cannot_mutate_buffer_or_fsm(tmp_path):
    service, model, data = service_fixture(tmp_path)
    packet = service.observe(data, fsm={})
    packet['frames'][0]['robot']['position_w'][0] = 999
    assert service.observe(data, fsm={})['frames'][0]['robot']['position_w'][0] == 0
