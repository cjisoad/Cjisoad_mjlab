"""Loopback-only HTTP bridge for bounded actions and observation resources."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, urlsplit


def make_server(runtime, host='127.0.0.1', port=8766, *, web_ui=True):
    if host not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('agent control service must bind to loopback')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _fmt, *_args):
            pass

        def _reply(self, status, value, content_type='application/json'):
            payload = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def _body(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError as exc:
                raise ValueError('invalid content length') from exc
            if not 0 < length <= 16384:
                raise ValueError('body must be between 1 and 16384 bytes')
            try:
                return json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError('body must contain valid JSON') from exc

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                if not web_ui:
                    return self._reply(404, {'error': 'web viewer is disabled'})
                page = Path(__file__).with_name('ui.html').read_bytes()
                return self._reply(200, page, 'text/html; charset=utf-8')
            if path == '/scene.png':
                if not web_ui:
                    return self._reply(404, {'error': 'web viewer is disabled'})
                try:
                    values = parse_qs(urlsplit(self.path).query, strict_parsing=True)
                    allowed = {'azimuth', 'elevation', 'distance', 'x', 'y', 'z'}
                    if values.keys() - allowed or any(len(value) != 1 for value in values.values()):
                        raise ValueError('invalid camera query')
                    camera = {key: float(value[0]) for key, value in values.items()}
                    return self._reply(200, runtime.scene_image(camera), 'image/png')
                except (ValueError, RuntimeError) as exc:
                    return self._reply(400, {'error': str(exc)})
            if path == '/health':
                return self._reply(200, {'ok': True})
            pipeline = getattr(runtime, 'pipeline', None)
            if path == '/pipeline/status' and pipeline is not None:
                return self._reply(200, pipeline.status())
            match = re.fullmatch(r'/pipeline/resources/([a-f0-9]{32})', path)
            if match and pipeline is not None:
                try:
                    payload, mime = runtime.snapshots.resource(match.group(1))
                    return self._reply(200, payload, mime)
                except KeyError:
                    return self._reply(404, {'error': 'snapshot resource not found'})
            match = re.fullmatch(r'/resources/([a-f0-9]{32})', path)
            if match:
                try:
                    payload, mime = runtime.resource(match.group(1))
                    return self._reply(200, payload, mime)
                except KeyError:
                    return self._reply(404, {'error': 'resource not found'})
            return self._reply(404, {'error': 'not found'})

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and urlsplit(origin).netloc != self.headers.get('Host'):
                return self._reply(403, {'error': 'origin mismatch'})
            try:
                value = self._body()
                if not isinstance(value, dict):
                    raise ValueError('expected a JSON object')
                path = urlsplit(self.path).path
                pipeline = getattr(runtime, 'pipeline', None)
                if path.startswith('/pipeline/') and pipeline is not None:
                    routes = {'/pipeline/start': pipeline.start,
                        '/pipeline/observe': pipeline.observe, '/pipeline/ack': pipeline.acknowledge,
                        '/pipeline/plan': pipeline.submit}
                    if path == '/pipeline/stop':
                        if value:
                            raise ValueError('stop body must be empty')
                        return self._reply(200, pipeline.stop())
                    if path in routes:
                        return self._reply(200, routes[path](value))
                if pipeline is not None and pipeline.active and path in ('/action', '/reset', '/button', '/scene-click'):
                    return self._reply(409, {'error': 'pipeline owns execution; stop it before direct control'})
                if path == '/action':
                    return self._reply(200, runtime.action(value))
                if path == '/button':
                    if set(value) != {'button'} or not isinstance(value['button'], str):
                        raise ValueError('button requires exactly a button name')
                    return self._reply(200, runtime.press_button(value['button']))
                if path == '/scene-click':
                    if not web_ui:
                        return self._reply(404, {'error': 'web viewer is disabled'})
                    if set(value) != {'camera', 'x', 'y'} or not isinstance(value['camera'], dict):
                        raise ValueError('scene-click requires camera, x, and y')
                    x, y = float(value['x']), float(value['y'])
                    if not all(map(math.isfinite, (x, y))) or not (0 <= x <= 1 and 0 <= y <= 1):
                        raise ValueError('scene click must be inside the viewport')
                    return self._reply(200, runtime.click_scene(value['camera'], x, y))
                if path == '/observe':
                    keys = set(value)
                    if keys - {'consumer_id', 'cursor', 'max_frames', 'replay_latest'}:
                        raise ValueError('unexpected observe field')
                    replay = value.get('replay_latest', False)
                    if type(replay) is not bool:
                        raise ValueError('replay_latest must be a boolean')
                    return self._reply(200, runtime.observe(
                        consumer_id=value.get('consumer_id', 'default'),
                        cursor=value.get('cursor'), max_frames=value.get('max_frames', 2),
                        **({'replay_latest': True} if replay else {})))
                if path == '/ack':
                    if set(value) != {'consumer_id', 'cursor'}:
                        raise ValueError('ack requires consumer_id and cursor')
                    runtime.acknowledge(value['consumer_id'], value['cursor'])
                    return self._reply(200, {'acknowledged': True})
                if path == '/reset':
                    if value:
                        raise ValueError('reset body must be empty')
                    return self._reply(200, runtime.reset())
                return self._reply(404, {'error': 'not found'})
            except (ValueError, TypeError, RuntimeError, KeyError) as exc:
                return self._reply(400, {'error': str(exc)})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve(runtime, *, host='127.0.0.1', port=8766, viewer='web'):
    if viewer not in ('native', 'web', 'none'):
        raise ValueError('viewer must be native, web, or none')
    server = make_server(runtime, host, port, web_ui=viewer == 'web')
    thread = None
    print(f'Agentic robot tools: http://{host}:{server.server_port}', flush=True)
    try:
        if viewer == 'native':
            from .native_viewer import NativeSceneViewer
            runtime.native_viewer = NativeSceneViewer(runtime.env.sim.mj_model, runtime.adapter.data,
                on_button=runtime.queue_button_press)
            runtime.native_viewer.open()
            runtime.start_background()
            thread = threading.Thread(target=server.serve_forever,
                kwargs={'poll_interval': .1}, daemon=True)
            thread.start()
            print('Viewer: native MuJoCo window; click a button to press, drag to orbit/pan, scroll to zoom', flush=True)
            while runtime.native_viewer.tick():
                time.sleep(1. / 30.)
        else:
            print(f'Viewer: {viewer}', flush=True)
            runtime.start_background()
            server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        if viewer != 'native' or (thread is not None and thread.is_alive()):
            server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join()
        runtime.close()
