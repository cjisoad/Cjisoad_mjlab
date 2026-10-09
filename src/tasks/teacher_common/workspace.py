"""Verified FR target bank for the quadruped locomotion teacher."""

from .training_workspace import DEFAULT_TRAINING_CACHE, load_training_workspace

# Legacy rectangular limits for the union keyboard mode. The quadruped teacher
# and quadruped keyboard load their respective geometry-derived caches.
LOCO_TEACHER_TRIPOD_X_RANGE = (-.30, .30)
LOCO_TEACHER_TRIPOD_Y_RANGE = (-.12, .12)


def loco_teacher_foot_workspace(path=DEFAULT_TRAINING_CACHE):
  return load_training_workspace(path)
