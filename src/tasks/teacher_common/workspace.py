"""Verified FR target bank for the quadruped locomotion teacher."""

from src.tasks.leg_manip.constants import DZ_MIN, TRIPOD_HIGH
from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace

# A dense MuJoCo FK scan with Cartesian-path IK, joint-limit margins, and
# collision clearance reaches about +/- .35 m in X and +/- .15 m in Y. Retain
# 5 cm of margin for the teacher's randomized dynamics and tracking error.
LOCO_TEACHER_TRIPOD_X_RANGE = (-.30, .30)
LOCO_TEACHER_TRIPOD_Y_RANGE = (-.12, .12)


def loco_teacher_foot_workspace():
  return build_foot_workspace(
    lift_range=(DZ_MIN, TRIPOD_HIGH),
    max_offset=(LOCO_TEACHER_TRIPOD_X_RANGE[1], LOCO_TEACHER_TRIPOD_Y_RANGE[1]),
    grid_points=13)
