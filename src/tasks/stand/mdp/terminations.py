"""Contact and timeout conditions, without additional posture cutoffs."""


def base_contact(env):
  return (env.scene['base_contact'].data.force.norm(dim=-1) > 1.).any(-1)


def time_out(env):
  return env.episode_length_buf > env.max_episode_length
