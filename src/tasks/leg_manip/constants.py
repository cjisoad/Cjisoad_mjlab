"""FR target height-band constants for the fused loco/biped task, in meters.

All thresholds compare against ``dz``, the commanded FR target vertical offset
from the quadruped standing FR site zero. The equivalent sole clearance above
the plane is ``dz + foot_radius`` at the nominal stance, so thresholds on ``dz``
are thresholds on operation height up to a constant.
"""

DZ_MIN = .05
TRIPOD_HIGH = .35
BIPED_LOW = TRIPOD_HIGH - .10
BIPED_HIGH = .72
STAGE_STANCE = 0
STAGE_WALK = 1
STAGE_OPERATION = 2
STAGE_BIPED = STAGE_OPERATION  # Public play/backward constant alias, not a checkpoint migration.
COURSE_VERSION = 1
COURSE_NAME = "stance_walk_operation"

LEGACY_ACTOR_DIM = 48
ACTOR_OBS_DIM = 51
CRITIC_OBS_DIM = 131
BASE_VELOCITY_SCALE = 2.
BASE_VELOCITY_NOISE = .02  # Uniform half-width, physical m/s before scaling.
