"""A fresh agent can read the retained camera view after all frames were ACKed."""
import copy

import pytest

from test_agentic_observation_delivery import service_fixture, sample_at
from test_agentic_server import endpoint, post


def test_replay_latest_is_read_only_and_does_not_create_pending_delivery(tmp_path):
    service, model, data = service_fixture(tmp_path)
    sample_at(service, model, data, .5)
    initial = service.observe(data, fsm={})
    service.acknowledge('default', initial['next_cursor'])
    before = copy.deepcopy(service.buffer.consumers)
    qpos_before = data.qpos.copy()
    for _ in range(5):
        replay = service.observe(data, fsm={}, replay_latest=True)
        assert [frame['capture_sim_time'] for frame in replay['frames']] == [0, .5]
        assert replay['frames_replayed'] is True and replay['ack_required'] is False
        assert replay['next_cursor'] == initial['next_cursor']
        assert service.buffer.consumers == before
        assert data.time == .5 and (data.qpos == qpos_before).all()
    replay['frames'][0]['robot']['position_w'][0] = 999
    assert service.observe(data, fsm={}, replay_latest=True)['frames'][0]['robot']['position_w'][0] == 0
    service.close()


def test_replay_does_not_skip_unconfirmed_frames_or_override_normal_delivery(tmp_path):
    service, model, data = service_fixture(tmp_path)
    initial = service.observe(data, fsm={}, replay_latest=True)
    assert initial['frames_replayed'] is False and initial['ack_required'] is True
    sample_at(service, model, data, .5)
    pending = service.observe(data, fsm={}, replay_latest=True)
    assert pending['frames'] == initial['frames']
    service.acknowledge('default', pending['next_cursor'])
    fresh = service.observe(data, fsm={}, replay_latest=True)
    assert [frame['capture_sim_time'] for frame in fresh['frames']] == [.5]
    assert fresh['frames_replayed'] is False and fresh['ack_required'] is True
    service.close()


def test_replay_respects_zero_frame_request_and_rejects_non_boolean_flag(tmp_path):
    service, model, data = service_fixture(tmp_path)
    packet = service.observe(data, fsm={})
    service.acknowledge('default', packet['next_cursor'])
    assert service.observe(data, fsm={}, max_frames=0, replay_latest=True)['frames'] == []
    with pytest.raises(ValueError, match='boolean'):
        service.observe(data, fsm={}, replay_latest='yes')
    service.close()


def test_http_observe_forwards_replay_flag_and_rejects_invalid_type(endpoint):
    import json
    from urllib.error import HTTPError
    base, runtime = endpoint
    json.load(post(base, '/observe', {'replay_latest': True}))
    assert runtime.calls[-1] == ('observe', {
        'consumer_id': 'default', 'cursor': None, 'max_frames': 2, 'replay_latest': True})
    with pytest.raises(HTTPError) as error:
        post(base, '/observe', {'replay_latest': 'true'})
    assert error.value.code == 400
