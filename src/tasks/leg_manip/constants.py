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
STAGE_BIPED = 4
