"""Optional ClearML reporting without embedding credentials in run metadata."""
import math
from pathlib import Path


class ClearMLTracker:
  def __init__(self, *, enabled=False, project='Go2/transition_t', task_name='V3 three-stage scratch',
      output_uri=None, offline=False, checkpoint_interval=1000, task=None, model_factory=None):
    self.task, self.model, self.model_factory = task, None, model_factory
    self.checkpoint_interval = checkpoint_interval
    if enabled:
      from clearml import Task, OutputModel
      if offline:
        Task.set_offline(offline_mode=True)
      self.task = Task.init(project_name=project, task_name=task_name,
        tags=['V3', 'three-stage', 'from-scratch'], reuse_last_task_id=False,
        output_uri=output_uri if output_uri is not None else True, auto_connect_arg_parser=False,
        auto_connect_frameworks={'tensorboard': True}, auto_connect_streams=False,
        auto_resource_monitoring=True)
      self.model_factory = OutputModel
    if self.task is not None:
      self.logger = self.task.get_logger()

  def configuration(self, data):
    if self.task is not None:
      self.task.connect(data, name='Training recipe')

  def artifact(self, path):
    if self.task is not None:
      path = Path(path)
      self.task.upload_artifact(name=path.stem, artifact_object=path)

  def scalar(self, title, series, value, step):
    if self.task is not None and isinstance(value, (int, float)) and math.isfinite(value):
      self.logger.report_scalar(title=title, series=series, value=value, iteration=step)

  def evaluation(self, report, path, step):
    title = f'Evaluation/{report["initialization"]}/stage{report["level"]}'
    attempts = report['attempts']
    self.scalar(title, 'success_rate', report['successes']/attempts, step)
    self.scalar(title, 'full_motion_completion_rate', report['full_motion_completions']/attempts, step)
    self.scalar(title, 'distinct_trajectories', report.get('distinct_trajectories', 0), step)
    for axis in ('actual_speed', 'fr_error'):
      for i, entry in enumerate(report.get('by_'+axis, [])):
        if entry['attempts']:
          self.scalar(title+'/'+axis, str(i), entry['successes']/entry['attempts'], step)
    diagnostic_names = set().union(*(r.get('diagnostics', {}) for r in report.get('records', [])))
    for name in sorted(diagnostic_names):
      values = [r['diagnostics'][name] for r in report.get('records', [])
        if isinstance(r.get('diagnostics', {}).get(name), (int, float))]
      if values:
        self.scalar(title+'/diagnostics', name, sum(values)/len(values), step)
    self.artifact(path)

  def course(self, event, path):
    for key, value in event['course'].items():
      self.scalar('Curriculum', key, int(value), event['updates_completed'])
    self.scalar('Curriculum', 'promoted', int(event['promoted']), event['updates_completed'])
    if event['promoted']:
      self.artifact(path)

  def checkpoint(self, path, step, *, force=False):
    if self.task is None or (not force and step % self.checkpoint_interval):
      return
    if self.model is None:
      # ClearML's offline DummyModel in some SDK versions does not implement
      # the OutputModel lock properties. Artifact upload remains portable.
      if getattr(self.task, 'is_offline', lambda: False)():
        self.artifact(path)
        return
      self.model = self.model_factory(task=self.task, name='V3 transition latest', framework='PyTorch')
    self.model.update_weights(weights_filename=str(path), auto_delete_file=False, iteration=step)

  def close(self):
    if self.task is not None:
      self.task.close()
