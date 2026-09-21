"""Import a staged task alongside the installed target repository for CPU tests."""

from pathlib import Path


def import_stand():
  import src.tasks
  tasks_path = str(Path(__file__).resolve().parents[1] / 'src' / 'tasks')
  if tasks_path not in src.tasks.__path__:
    src.tasks.__path__.append(tasks_path)
  import src.tasks.stand
  return src.tasks.stand
