"""Observation composition; the runtime remains the only physics owner."""
from copy import deepcopy
from pathlib import Path
import math

import mujoco

from .buffer import FrameBuffer
from .controls import BUTTON_NAMES, ControlRegistry
from .render_worker import RenderWorker
from .state import StateReader


class ObservationService:
    def __init__(self, model, output_dir: Path, *, robot_prefix='robot/', fr_zero_anchor,
                 width=640, height=480, capture_images=True, max_buffer_frames=64, max_consumers=32):
        self.model = model
        required = [(mujoco.mjtObj.mjOBJ_BODY, robot_prefix + 'base_link'),
                    (mujoco.mjtObj.mjOBJ_SITE, robot_prefix + 'FR'),
                    (mujoco.mjtObj.mjOBJ_CAMERA, 'robot_ego')]
        required += [(mujoco.mjtObj.mjOBJ_SITE, f'surface_{name}') for name in BUTTON_NAMES]
        required += [(mujoco.mjtObj.mjOBJ_JOINT, f'joint_{name}') for name in BUTTON_NAMES]
        ids = [mujoco.mj_name2id(model, kind, name) for kind, name in required]
        missing = [name for (_, name), object_id in zip(required, ids) if object_id < 0]
        if missing:
            raise ValueError('missing model objects: ' + ', '.join(missing))
        self.state = StateReader(model, ids[0], ids[1], fr_zero_anchor)
        self.controls = ControlRegistry(model)
        self.buffer = FrameBuffer(max_buffer_frames, max_consumers)
        self.capture_images = capture_images
        self.width, self.height = width, height
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resources = {}
        self.capture = None
        self.scene_camera = None
        self.render_worker = RenderWorker() if capture_images else None
        self.episode_id = None
        self.snapshot_id = 0
        self.image_error = None

    def reset(self, episode_id: str, data, *, fsm):
        if not isinstance(episode_id, str) or not episode_id or episode_id == self.episode_id:
            raise ValueError('reset requires a new nonempty episode_id')
        self._delete_resources()
        self.episode_id = episode_id
        self.buffer.reset(episode_id)
        self.snapshot_id = 0
        self.image_error = None
        self.start_time = float(data.time)
        self.next_sample = 1
        self.last_sample_time = self.start_time
        self._capture(data, fsm, scheduled_sim_time=self.start_time)

    def sample(self, data, *, fsm):
        self._require_reset()
        now = float(data.time)
        if now < self.last_sample_time - 1e-9:
            raise ValueError('simulation time moved backwards; reset episode first')
        self.last_sample_time = now
        due = self.start_time + self.next_sample * .5
        if now + 1e-9 >= due:
            self._capture(data, fsm, scheduled_sim_time=due)
            # A large skipped interval cannot be reconstructed from one state.
            self.next_sample = max(self.next_sample + 1, math.floor((now - self.start_time + 1e-9) / .5) + 1)

    def _capture(self, data, fsm, *, scheduled_sim_time):
        frame = dict(episode_id=self.episode_id, capture_sim_time=float(data.time),
                     scheduled_sim_time=scheduled_sim_time, robot=self.state.read(data, fsm),
                     image_status='disabled', resources={})
        if self.capture_images:
            try:
                if self.capture is None:
                    self.capture = self.render_worker.call(self._new_rgbd_capture)
                metadata, blobs = self.render_worker.call(self.capture.capture, data)
                frame.update(metadata, image_status='ok')
                for name, (payload, mime, resource_metadata) in blobs.items():
                    import uuid
                    resource_id = uuid.uuid4().hex
                    path = self.output_dir / resource_id
                    path.write_bytes(payload)
                    self.resources[resource_id] = (path, mime)
                    frame['resources'][name] = dict(resource_metadata, resource_id=resource_id,
                                                   uri=f'/resources/{resource_id}', mime_type=mime)
                self.image_error = None
            except Exception as exc:
                frame['image_status'] = 'error'
                frame['image_error'] = f'{type(exc).__name__}: {exc}'
                self.image_error = frame['image_error']
        self.buffer.append(frame)
        self._cleanup_resources()

    def _new_rgbd_capture(self):
        from .rgbd import RGBDCapture
        return RGBDCapture(self.model, width=self.width, height=self.height)

    def _delete_resources(self):
        for path, _mime in self.resources.values():
            path.unlink(missing_ok=True)
        self.resources.clear()

    def _cleanup_resources(self):
        retained = {r['resource_id'] for frame in self.buffer.frames
                    for r in frame['resources'].values()}
        retained.update(r['resource_id'] for consumer in self.buffer.consumers.values()
                        if consumer['pending'] for frame in consumer['pending'][0]
                        for r in frame['resources'].values())
        for resource_id in self.resources.keys() - retained:
            path, _mime = self.resources.pop(resource_id)
            path.unlink(missing_ok=True)

    def _require_reset(self):
        if self.episode_id is None:
            raise ValueError('reset observation service before use')

    def observe(self, data, *, fsm, consumer_id='default', cursor=None, max_frames=2, replay_latest=False):
        self._require_reset()
        if type(replay_latest) is not bool:
            raise ValueError('replay_latest must be a boolean')
        frames, next_cursor, dropped = self.buffer.deliver(consumer_id, cursor, max_frames)
        replayed = bool(replay_latest and not frames and max_frames and self.buffer.frames)
        if replayed:
            # New model sessions need an initial view even when an earlier
            # session ACKed every frame. This neither advances the consumer nor
            # creates a new pending delivery; capture times remain unchanged.
            frames = deepcopy(list(self.buffer.frames)[-max_frames:])
        self.snapshot_id += 1
        status = 'error' if self.image_error else ('no_new_frame' if not frames else 'ok' if self.capture_images else 'disabled')
        return dict(schema_version='1.0', episode_id=self.episode_id, snapshot_id=self.snapshot_id,
                    sim_time=float(data.time), robot=self.state.read(data, fsm), controls=self.controls.read(data),
                    frames=deepcopy(frames), next_cursor=next_cursor, dropped_frame_count=dropped,
                    image_status=status, **({'image_error': self.image_error} if self.image_error else {}),
                    **({'frames_replayed': replayed, 'ack_required': bool(frames) and not replayed}
                       if replay_latest else {}))

    def acknowledge(self, consumer_id, cursor):
        self.buffer.acknowledge(consumer_id, cursor)
        self._cleanup_resources()

    def resource(self, resource_id):
        if not isinstance(resource_id, str) or resource_id not in self.resources:
            raise KeyError(resource_id)
        path, mime = self.resources[resource_id]
        return path.read_bytes(), mime

    def render_scene(self, data, camera=None):
        if self.render_worker is None:
            self.render_worker = RenderWorker()
        return self.render_worker.call(self._render_scene, data, camera)

    def _render_scene(self, data, camera=None):
        if self.scene_camera is None:
            from .scene_camera import SceneCamera
            self.scene_camera = SceneCamera(self.model)
        return self.scene_camera.render(data, camera)

    def pick_scene_button(self, data, camera, relx, rely):
        if self.render_worker is None:
            self.render_worker = RenderWorker()
        return self.render_worker.call(self._pick_scene_button, data, camera, relx, rely)

    def _pick_scene_button(self, data, camera, relx, rely):
        if self.scene_camera is None:
            from .scene_camera import SceneCamera
            self.scene_camera = SceneCamera(self.model)
        return self.scene_camera.pick_button(data, camera, relx, rely)

    def close(self):
        if self.render_worker is not None:
            self.render_worker.call(self._close_renderers)
            self.render_worker.close()
            self.render_worker = None

    def _close_renderers(self):
        if self.capture is not None:
            self.capture.close()
            self.capture = None
        if self.scene_camera is not None:
            self.scene_camera.close()
            self.scene_camera = None
