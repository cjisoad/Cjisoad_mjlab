"""Shared foot-contact classification for rewards and critic observations."""


def foot_in_contact(env, sensor_name, threshold=1.0):
  # Net contact-force direction depends on the sensor's primary/secondary order.
  return env.scene[sensor_name].data.force.norm(dim=-1) > threshold
