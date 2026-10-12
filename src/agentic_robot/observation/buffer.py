"""Bounded frame delivery with explicit per-consumer acknowledgement."""
from collections import deque
from copy import deepcopy


class FrameBuffer:
    def __init__(self, max_frames=64, max_consumers=32):
        if max_frames < 2 or max_consumers < 1:
            raise ValueError('buffer needs at least two frames and one consumer')
        self.frames = deque(maxlen=max_frames)
        self.max_consumers = max_consumers
        self.reset(None)

    def reset(self, episode_id):
        self.episode_id = episode_id
        self.frames.clear()
        self.consumers = {}
        self.seq = -1

    def append(self, frame):
        self.seq += 1
        frame['frame_seq'] = self.seq
        frame['frame_id'] = f'{self.episode_id}:{self.seq}'
        evicted = self.frames[0] if len(self.frames) == self.frames.maxlen else None
        self.frames.append(frame)
        return evicted

    def _consumer(self, consumer_id):
        if not isinstance(consumer_id, str) or not consumer_id or len(consumer_id) > 256:
            raise ValueError('invalid consumer_id')
        if consumer_id not in self.consumers:
            if len(self.consumers) >= self.max_consumers:
                raise ValueError('consumer limit reached')
            self.consumers[consumer_id] = dict(ack=-1, offered=None, pending=None)
        return self.consumers[consumer_id]

    def _validate_cursor(self, cursor):
        if not isinstance(cursor, dict) or cursor.get('episode_id') != self.episode_id:
            raise ValueError('cursor belongs to another episode')
        seq = cursor.get('frame_seq')
        if type(seq) is not int or seq < -1:
            raise ValueError('cursor frame_seq must be an integer >= -1')
        return seq

    def deliver(self, consumer_id, cursor=None, max_frames=2):
        if type(max_frames) is not int or not 0 <= max_frames <= 2:
            raise ValueError('max_frames must be 0, 1, or 2')
        seq = self._validate_cursor(cursor) if cursor is not None else None
        consumer = self._consumer(consumer_id)
        if seq is not None and seq != consumer['ack']:
            raise ValueError('advance the cursor with acknowledge before observe')
        # Freeze an unacknowledged batch so transport retries are stable even as
        # newer simulation frames arrive. Stored metadata is bounded to two frames.
        if consumer['pending'] is not None and max_frames:
            return deepcopy(consumer['pending'])
        candidates = [frame for frame in self.frames if frame['frame_seq'] > consumer['ack']]
        frames = candidates[-max_frames:] if max_frames else []
        dropped = max(0, frames[0]['frame_seq'] - consumer['ack'] - 1) if frames else 0
        last = frames[-1]['frame_seq'] if frames else consumer['ack']
        cursor_out = dict(episode_id=self.episode_id, frame_seq=last)
        result = (deepcopy(frames), cursor_out, dropped)
        if frames:
            consumer['offered'] = last
            consumer['pending'] = deepcopy(result)
        return result

    def acknowledge(self, consumer_id, cursor):
        seq = self._validate_cursor(cursor)
        consumer = self._consumer(consumer_id)
        if seq == consumer['ack']:
            return
        if seq != consumer['offered']:
            raise ValueError('cannot acknowledge frames that were not delivered')
        consumer.update(ack=seq, offered=None, pending=None)
