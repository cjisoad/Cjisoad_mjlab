"""HTTP tools expose bounded actions, observation cursors and safe resources."""
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.agentic_robot.server import make_server


class Runtime:
    def __init__(self):
        self.calls = []
    def action(self, value):
        self.calls.append(('action', value))
        return {'executed_duration': value['t']}
    def observe(self, **kwargs):
        self.calls.append(('observe', kwargs))
        return {'frames': [], 'next_cursor': {'episode_id': 'e', 'frame_seq': -1}}
    def acknowledge(self, consumer_id, cursor):
        self.calls.append(('ack', consumer_id, cursor))
    def reset(self):
        self.calls.append(('reset',))
        return {'episode_id': 'next'}
    def resource(self, resource_id):
        if resource_id != 'a' * 32:
            raise KeyError(resource_id)
        return b'frame-resource', 'image/png'
    def scene_image(self, camera):
        self.calls.append(('scene_image', camera))
        return b'scene-frame'
    def press_button(self, button):
        self.calls.append(('press_button', button))
        return {'button': button}
    def click_scene(self, camera, x, y):
        self.calls.append(('click_scene', camera, x, y))
        return {'button': 'call_up_f1'}


@pytest.fixture
def endpoint():
    runtime = Runtime()
    server = make_server(runtime, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}', runtime
    server.shutdown()
    server.server_close()
    thread.join()


def post(base, path, value):
    req = Request(base + path, data=json.dumps(value).encode(), headers={'Content-Type': 'application/json'})
    return urlopen(req)


def test_action_observe_ack_reset_and_resource(endpoint):
    base, runtime = endpoint
    action = {'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': 1.}
    assert json.load(post(base, '/action', action))['executed_duration'] == 1.
    packet = json.load(post(base, '/observe', {'consumer_id': 'a', 'max_frames': 1}))
    assert packet['next_cursor']['frame_seq'] == -1
    assert json.load(post(base, '/ack', {'consumer_id': 'a', 'cursor': packet['next_cursor']}))['acknowledged']
    assert json.load(post(base, '/reset', {}))['episode_id'] == 'next'
    with urlopen(base + '/resources/' + 'a' * 32) as response:
        assert response.headers['Content-Type'] == 'image/png'
        assert response.read() == b'frame-resource'
    assert [call[0] for call in runtime.calls] == ['action', 'observe', 'ack', 'reset']


def test_root_serves_keyboard_and_observation_ui(endpoint):
    base, _ = endpoint
    with urlopen(base + '/') as response:
        page = response.read().decode()
        assert response.headers['Content-Type'].startswith('text/html')
        assert 'Go2 / Three-Floor Elevator' in page
        assert '/action' in page and '/observe' in page
        assert '/scene-click' in page and "post('/button'" in page


def test_scene_image_accepts_draggable_camera_parameters(endpoint):
    base, runtime = endpoint
    with urlopen(base + '/scene.png?azimuth=42&elevation=-18&distance=12') as response:
        assert response.headers['Content-Type'] == 'image/png'
        assert response.read() == b'scene-frame'
    assert runtime.calls[-1] == ('scene_image', {'azimuth': 42., 'elevation': -18., 'distance': 12.})


def test_mouse_button_press_and_scene_pick(endpoint):
    base, runtime = endpoint
    assert json.load(post(base, '/button', {'button': 'call_up_f1'})) == {'button': 'call_up_f1'}
    assert json.load(post(base, '/scene-click', {
        'camera': {'azimuth': 42}, 'x': .25, 'y': .75})) == {'button': 'call_up_f1'}
    assert runtime.calls == [
        ('press_button', 'call_up_f1'),
        ('click_scene', {'azimuth': 42}, .25, .75),
    ]


def test_scene_click_rejects_out_of_viewport_coordinates(endpoint):
    base, _ = endpoint
    with pytest.raises(HTTPError) as error:
        post(base, '/scene-click', {'camera': {}, 'x': 1.1, 'y': .5})
    assert error.value.code == 400


def test_unexpected_fields_unknown_path_and_unsafe_bind_rejected(endpoint):
    base, _ = endpoint
    with pytest.raises(HTTPError) as error:
        post(base, '/observe', {'debug': True})
    assert error.value.code == 400
    with pytest.raises(HTTPError) as error:
        urlopen(base + '/resources/' + 'b' * 32)
    assert error.value.code == 404
    with pytest.raises(ValueError, match='loopback'):
        make_server(Runtime(), host='0.0.0.0', port=0)
