"""Transport helper for a persistent planner; never generates or retries actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request(base, path, body=None):
    payload = None if body is None else json.dumps(body, allow_nan=False).encode()
    req = Request(base + path, data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=25) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode()}') from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8766')
    parser.add_argument('--output', type=Path, default=Path('/tmp/elevator_async_planner'))
    sub = parser.add_subparsers(dest='operation', required=True)
    start = sub.add_parser('start')
    start.add_argument('--objective', required=True)
    start.add_argument('--max-actions', type=int, default=20)
    start.add_argument('--bounds', required=True, help='JSON three [min,max] coordinate pairs')
    observe = sub.add_parser('observe')
    observe.add_argument('--after', type=int)
    observe.add_argument('--after-frame', type=int, help='wait for a newer RGBD frame sequence')
    observe.add_argument('--wait', type=float, default=0)
    ack = sub.add_parser('ack')
    ack.add_argument('request_file', type=Path)
    submit = sub.add_parser('submit')
    submit.add_argument('plan_file', type=Path)
    sub.add_parser('status')
    sub.add_parser('stop')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.operation == 'start':
        value = request(args.base, '/pipeline/start', {'objective': args.objective,
            'max_actions': args.max_actions, 'position_bounds': json.loads(args.bounds)})
    elif args.operation == 'observe':
        body = {'wait_s': args.wait}
        if args.after is not None:
            body['after_snapshot_id'] = args.after
        if args.after_frame is not None:
            body['after_frame_seq'] = args.after_frame
        value = request(args.base, '/pipeline/observe', body)
        dest = args.output / f"request_{value['request_id']}.json"
        for frame in value['observation']['frames']:
            for kind in ('rgb', 'depth_visualization', 'depth'):
                if kind in frame['resources']:
                    resource = frame['resources'][kind]
                    suffix = '.npy' if kind == 'depth' else '.png'
                    path = args.output / f"{frame['frame_id']}_{kind}{suffix}"
                    with urlopen(args.base + resource['uri'], timeout=10) as response:
                        path.write_bytes(response.read())
                    resource['local_path'] = str(path)
        dest.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        value['request_file'] = str(dest)
        # These are only downloaded. The model must view images before ack.
    elif args.operation == 'ack':
        observation = json.loads(args.request_file.read_text())
        value = request(args.base, '/pipeline/ack', {'request_id': observation['request_id'],
            'frame_ids': [frame['frame_id'] for frame in observation['observation']['frames']]})
    elif args.operation == 'submit':
        value = request(args.base, '/pipeline/plan', json.loads(args.plan_file.read_text()))
    else:
        value = request(args.base, '/pipeline/' + args.operation,
            {} if args.operation == 'stop' else None)
    args.output.joinpath('last_response.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
