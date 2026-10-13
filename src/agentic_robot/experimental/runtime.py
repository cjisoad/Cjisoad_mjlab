"""Opt-in snapshot publication and one exclusive plan execution owner."""
from copy import deepcopy
import threading

from ..runtime import AgenticRuntime
from .snapshots import SnapshotStore
from .pipeline import AsyncPipeline


class ExperimentalRuntime(AgenticRuntime):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.snapshots = SnapshotStore()
        self.pipeline = AsyncPipeline(self)
        self._frame_signature = None
        self._publish_state()

    def _publish_state(self):
        data = self.adapter.data
        packet = {'schema_version': '1.0', 'episode_id': self.episode_id,
            'sim_time': float(data.time),
            'robot': self.state_reader.read(data, self._fsm_state(command_accepted=None)),
            'controls': self.control_registry.read(data),
            'elevator': self.adapter.evaluation_state(), 'frames': [],
            'image_status': 'unavailable'}
        resources = {}
        if self.observation is not None:
            frames = [f for f in list(self.observation.buffer.frames)[-2:]
                if f['episode_id'] == self.episode_id]
            packet['frames'] = deepcopy(frames)
            packet['image_status'] = 'error' if self.observation.image_error else (
                'ok' if self.observation.capture_images else 'disabled')
            signature = tuple(f['frame_id'] for f in frames)
            if signature != self._frame_signature:
                for frame in frames:
                    for resource in frame['resources'].values():
                        resources[resource['resource_id']] = self.observation.resource(resource['resource_id'])
                self._frame_signature = signature
        self.snapshots.publish(packet, resources)
        self.pipeline.on_snapshot(packet)

    def action(self, command):
        with self.lock:
            if self.pipeline.active and threading.current_thread() is not self.pipeline.thread:
                raise RuntimeError('pipeline owns actions; stop it before direct control')
            return super().action(command)

    def reset(self):
        with self.lock:
            if self.pipeline.active:
                raise RuntimeError('stop pipeline before reset')
            return super().reset()

    def press_button(self, button):
        with self.lock:
            if self.pipeline.active:
                raise RuntimeError('pipeline owns actions; direct button testing is disabled')
            return super().press_button(button)

    def click_scene(self, camera, relx, rely):
        with self.lock:
            if self.pipeline.active:
                raise RuntimeError('pipeline owns actions; scene clicks are disabled')
            return super().click_scene(camera, relx, rely)

    def close(self):
        self.pipeline.close()
        super().close()
