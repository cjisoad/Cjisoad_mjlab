"""ClearML lifecycle, artifact schedule and disabled optional dependency."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest


class FakeTask:
    def __init__(self):
        self.uploads = []
        self.closed = False
        self.failed = False
    def connect(self, data, **kwargs):
        self.configuration = data
    def upload_artifact(self, **kwargs):
        self.uploads.append((kwargs['name'], Path(kwargs['artifact_object']).read_bytes(), kwargs))
        return True
    def mark_failed(self, **kwargs):
        self.failed = True
    def close(self):
        self.closed = True


def test_disabled_clearml_requires_no_sdk(monkeypatch):
    from src.tasks.legmanip_distil.supermimic.clearml_logging import SuperMimicTracker
    from src.tasks.legmanip_distil.supermimic.config import ClearMLCfg
    monkeypatch.setitem(sys.modules, 'clearml', None)
    tracker = SuperMimicTracker(ClearMLCfg())
    tracker.close()
    assert tracker.task is None


def test_artifact_interval_final_export_and_no_duplicate(monkeypatch, tmp_path):
    from src.tasks.legmanip_distil.supermimic.clearml_logging import SuperMimicTracker
    from src.tasks.legmanip_distil.supermimic.config import ClearMLCfg
    task = FakeTask()
    options = {}
    def init(**kwargs):
        options.update(kwargs)
        return task
    monkeypatch.setitem(sys.modules, 'clearml', SimpleNamespace(Task=SimpleNamespace(init=init)))
    tracker = SuperMimicTracker(ClearMLCfg(enabled=True, checkpoint_interval=2))
    tracker.configuration({'recipe':'fresh'})
    student = tmp_path/'student.pt'
    student.write_bytes(b'student2')
    checkpoint = tmp_path/'model_2.pt'
    checkpoint.write_bytes(b'full-state')
    tracker.checkpoint(checkpoint,1)
    assert not task.uploads
    tracker.checkpoint(checkpoint,2)
    tracker.checkpoint(checkpoint,2,force=True)
    assert [u[0] for u in task.uploads] == ['checkpoint_2','student_2']
    assert all(u[2]['wait_on_upload'] for u in task.uploads)
    assert options['output_uri'] is True
    assert options['auto_connect_frameworks']['pytorch'] is False
    assert options['auto_connect_frameworks']['tensorboard'] is True
    assert options['reuse_last_task_id'] is False
    student.write_bytes(b'student3')
    tracker.checkpoint(checkpoint,3,force=True)
    assert task.uploads[-1][1] == b'student3'
    tracker.close(failed=True)
    assert task.closed and task.failed


def test_invalid_interval_is_rejected_before_connecting():
    from src.tasks.legmanip_distil.supermimic.clearml_logging import SuperMimicTracker
    from src.tasks.legmanip_distil.supermimic.config import ClearMLCfg
    with pytest.raises(ValueError):
        SuperMimicTracker(ClearMLCfg(enabled=True,checkpoint_interval=0))
