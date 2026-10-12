"""Loopback-only HTTP bridge for bounded actions and observation resources."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
from urllib.parse import urlsplit


def make_server(runtime, host='127.0.0.1', port=8766):
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
            if path == '/health':
                return self._reply(200, {'ok': True})
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
                if path == '/action':
                    return self._reply(200, runtime.action(value))
                if path == '/observe':
                    keys = set(value)
                    if keys - {'consumer_id', 'cursor', 'max_frames'}:
                        raise ValueError('unexpected observe field')
                    return self._reply(200, runtime.observe(
                        consumer_id=value.get('consumer_id', 'default'),
                        cursor=value.get('cursor'), max_frames=value.get('max_frames', 2)))
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


def serve(runtime, *, host='127.0.0.1', port=8766):
    server = make_server(runtime, host, port)
    print(f'Agentic robot tools: http://{host}:{server.server_port}', flush=True)
    try:
        server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        runtime.close()
