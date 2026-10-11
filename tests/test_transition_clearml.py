from pathlib import Path
from types import SimpleNamespace


class FakeTask:
  def __init__(self):
    self.scalars = []; self.artifacts = []; self.models = []
  def get_logger(self):
    return SimpleNamespace(report_scalar=lambda **kw: self.scalars.append(kw))
  def upload_artifact(self, name, artifact_object, **kw):
    self.artifacts.append((name, artifact_object))


def test_clearml_disabled_needs_no_sdk_and_evaluation_has_bins(tmp_path):
  from src.tasks.transition_t.clearml_logging import ClearMLTracker
  assert ClearMLTracker(enabled=False).task is None
  task = FakeTask()
  tracker = ClearMLTracker(task=task)
  report = dict(level=1, initialization='entries', attempts=64, successes=48,
    full_motion_completions=60, by_actual_speed=[dict(lower=0,upper=.1,attempts=10,successes=8)],
    records=[dict(trajectory_id=123)])
  path = tmp_path/'eval.json'; path.write_text('{}')
  tracker.evaluation(report, path, 250)
  assert any(s['title'] == 'Evaluation/entries/stage1' and s['series'] == 'success_rate'
             and s['value'] == .75 for s in task.scalars)
  assert task.artifacts == [('eval', path)]


def test_course_report_and_checkpoint_upload(tmp_path):
  from src.tasks.transition_t.clearml_logging import ClearMLTracker
  task = FakeTask(); uploaded = []
  tracker = ClearMLTracker(task=task, model_factory=lambda **kw: SimpleNamespace(
    update_weights=lambda **v: uploaded.append(v)))
  tracker.course(dict(updates_completed=500, promoted=True,
    course=dict(level=1, streak=0, evaluations=2, accepted=False)), tmp_path/'course.jsonl')
  tracker.checkpoint(tmp_path/'model_499.pt', 500, force=True)
  assert uploaded[0]['weights_filename'].endswith('model_499.pt')
  assert any(s['series'] == 'level' and s['value'] == 1 for s in task.scalars)


def test_offline_checkpoint_uses_artifact_without_dummy_model(tmp_path):
  from src.tasks.transition_t.clearml_logging import ClearMLTracker
  task = FakeTask(); task.is_offline = lambda: True
  tracker = ClearMLTracker(task=task, model_factory=lambda **kw: (_ for _ in ()).throw(AssertionError('offline model')))
  checkpoint = tmp_path/'model_249.pt'
  tracker.checkpoint(checkpoint,250,force=True)
  assert task.artifacts == [('model_249',checkpoint)]
