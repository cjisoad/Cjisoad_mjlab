"""Optional ClearML cloud tracking, with explicit checkpoint uploads."""
from pathlib import Path
import warnings
from .config import ClearMLCfg


class SuperMimicTracker:
    def __init__(self, cfg, task_name='SuperMimic fresh'):
        self.cfg = ClearMLCfg(**cfg) if isinstance(cfg, dict) else cfg
        self.task = None
        self.last_uploaded_step = None
        if self.cfg.checkpoint_interval < 1:
            raise ValueError('ClearML checkpoint interval must be positive')
        if self.cfg.offline and not self.cfg.enabled:
            raise ValueError('ClearML offline mode requires enabled=True')
        if not self.cfg.enabled:
            return
        try:
            from clearml import Task
        except ImportError as error:
            raise ImportError('Install requirements-clearml.txt to enable ClearML') from error
        if self.cfg.offline:
            Task.set_offline(offline_mode=True)
        self.task = Task.init(
            project_name=self.cfg.project,
            task_name=self.cfg.task_name or task_name or 'SuperMimic fresh',
            tags=['Go2', 'SuperMimic', 'four-teacher'],
            reuse_last_task_id=False,
            output_uri=self.cfg.output_uri if self.cfg.output_uri is not None else True,
            auto_connect_arg_parser=False,
            # Upload full training state explicitly; tensorboard supplies all
            # existing metrics without duplicate hooks on every torch.save.
            auto_connect_frameworks={'pytorch': False, 'tensorboard': True},
            auto_connect_streams=True,
            auto_resource_monitoring=True,
        )

    def configuration(self, data):
        if self.task is not None:
            self.task.connect(data, name='Training recipe')

    def checkpoint(self, path, step, force=False):
        if self.task is None or self.last_uploaded_step == step:
            return
        if not force and step % self.cfg.checkpoint_interval:
            return
        path = Path(path)
        try:
            for name, artifact in ((f'checkpoint_{step}', path),
                                   (f'student_{step}', path.with_name('student.pt'))):
                ok = self.task.upload_artifact(
                    name=name, artifact_object=artifact,
                    metadata={'completed_updates': step},
                    # student.pt is overwritten at the next save. Wait until
                    # the upload has captured this exact version.
                    wait_on_upload=True, retries=2)
                if not ok:
                    raise RuntimeError(f'ClearML artifact upload failed: {name}')
            self.last_uploaded_step = step
        except Exception as error:
            # A connectivity failure must not discard hours of local training.
            # The next checkpoint and final save will retry the latest state.
            warnings.warn(f'ClearML upload incomplete; local checkpoint retained at {path}: {error}',
                          RuntimeWarning, stacklevel=2)

    def close(self, failed=False):
        if self.task is not None:
            try:
                if failed:
                    self.task.mark_failed(status_reason='Training interrupted or failed')
            finally:
                self.task.close()
