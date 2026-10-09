"""Standing time credit shared by held-out trials and teacher handoff."""
import torch

STANDING_HOLD_SECONDS = 1.
STANDING_BAD_SAMPLE_LIMIT = 3


def standing_hold_policy():
  return dict(hold_s=STANDING_HOLD_SECONDS, bad_sample_limit=STANDING_BAD_SAMPLE_LIMIT,
    bad_sample_credit='pause', success_requires_good_sample=True,
    physical_failure='immediate', strict_endpoint='uninterrupted')


def advance_standing_hold(duration, bad_samples, good, *, eligible, active, dt):
  """Pause credit for two bad samples, resetting on the third.

  Ineligible active worlds clear immediately (e.g. reference incomplete or
  physical failure); inactive worlds retain their captured state. Call once per
  control step. A recovered good sample both earns credit and clears the streak.
  """
  next_bad = torch.where(good | ~eligible, 0, bad_samples+1).clamp(max=STANDING_BAD_SAMPLE_LIMIT)
  next_duration = torch.where(eligible,
    torch.where(good, duration+dt,
      torch.where(next_bad >= STANDING_BAD_SAMPLE_LIMIT, 0., duration)), 0.)
  return torch.where(active, next_duration, duration), torch.where(active, next_bad, bad_samples)
