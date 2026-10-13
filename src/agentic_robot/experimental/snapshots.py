"""Immutable observations readable without the action owner's lock."""
from collections import OrderedDict
from copy import deepcopy
import threading
import time
import uuid


class SnapshotStore:
    def __init__(self):
        self.condition = threading.Condition(threading.RLock())
        self.latest = None
        self.version = 0
        self.requests = OrderedDict()
        self.resources = {}
        self.request_index = 0

    def publish(self, packet, resources=None):
        value = deepcopy(packet)
        with self.condition:
            self.version += 1
            value['snapshot_id'] = self.version
            value['published_wall_ns'] = time.time_ns()
            for frame in value.get('frames', []):
                for resource in frame['resources'].values():
                    resource['uri'] = '/pipeline/resources/' + resource['resource_id']
            self.latest = value
            self.resources.update(resources or {})
            self._prune_resources()
            self.condition.notify_all()

    def _prune_resources(self):
        packets = [self.latest] + [r['observation'] for r in self.requests.values()]
        keep = {resource['resource_id'] for packet in packets if packet
            for frame in packet.get('frames', []) for resource in frame['resources'].values()}
        self.resources = {key: value for key, value in self.resources.items() if key in keep}

    def read(self):
        with self.condition:
            if self.latest is None:
                raise RuntimeError('no published observation')
            return deepcopy(self.latest)

    def request(self, task_id, after_snapshot_id=None, wait_s=0, after_frame_seq=None):
        deadline = time.monotonic() + wait_s
        with self.condition:
            def needs_update():
                old_state = after_snapshot_id is not None and self.version <= after_snapshot_id
                frames = self.latest.get('frames', []) if self.latest else []
                old_image = after_frame_seq is not None and (
                    not frames or frames[-1]['frame_seq'] <= after_frame_seq)
                return old_state or old_image
            while needs_update():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            self.request_index += 1
            request = {'request_id': uuid.uuid4().hex, 'request_index': self.request_index,
                'task_id': task_id, 'observation': self.read(),
                'delivered_wall_ns': time.time_ns(), 'acked': False}
            self.requests[request['request_id']] = request
            # One persistent planner; bounded leases also tolerate retries.
            while len(self.requests) > 8:
                self.requests.popitem(last=False)
            self._prune_resources()
            return deepcopy(request)

    def get_request(self, request_id):
        with self.condition:
            if request_id not in self.requests:
                raise ValueError('unknown or evicted planning request')
            return deepcopy(self.requests[request_id])

    def acknowledge(self, request_id, frame_ids):
        with self.condition:
            request = self.get_request(request_id)
            expected = [f['frame_id'] for f in request['observation'].get('frames', [])]
            if frame_ids != expected:
                raise ValueError('acknowledge exactly the delivered frame IDs')
            self.requests[request_id]['acked'] = True

    def resource(self, resource_id):
        with self.condition:
            if resource_id not in self.resources:
                raise KeyError('snapshot resource not found')
            return self.resources[resource_id]
