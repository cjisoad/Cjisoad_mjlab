"""Policy ABI and source task constants; independent of simulator joint indices."""

LEGS = ('FL', 'FR', 'RL', 'RR')
JOINT_NAMES = tuple(f'{leg}_{joint}_joint' for leg in LEGS for joint in ('hip', 'thigh', 'calf'))
DEFAULT_ANGLES = (.1, .8, -1.5, -.1, .8, -1.5, .1, 1., -1.5, -.1, 1., -1.5)
DESIRED_ANGLES = (0., .8, -1.5, 0., .8, -1.5, 0., 2.25, -1.75, 0., 2.25, -1.75)
# The front-foot line at DEFAULT_ANGLES, projected onto the base xoy plane,
# crosses y=0 at x=0.17782152; the goal is attached to base_link at z=0.
STAND_GOAL_POS = (0.17782152, 0., 0.)
REWARD_WEIGHTS = {
  'tracking_lin_vel': 2.5, 'tracking_ang_vel': 2.5,
  'lin_vel_z': .3, 'ang_vel_xy': .2, 'handstand_orientation': -1.,
  'torques': -.0002, 'dof_acc': -2.5e-7, 'base_height': 1.5,
  'handstand_feet_on_air': .4, 'collision': -2., 'action_rate': -.05,
  'default_pos_front': -.1, 'default_pos_rear': -.1,
  'default_hip_pos': -.1, 'feet_clearance': .4,
  'ang_xz': -.5, 'contact': .3, 'symmetric_joints': -.1,
  'orientation_symmetry': -.5, 'feet_height_symmetry': -.2,
  'handstand_feet_height_exp': 5.,
  'default_pos_reward_FL': .5, 'default_pos_reward_FR': .5,
  'dof_pos_limits': -2., 'alive': 1.,
}
