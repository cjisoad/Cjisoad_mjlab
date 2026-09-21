"""Manager-based runner for the Go2 rear-leg standing task."""

from mjlab.rl.runner import MjlabOnPolicyRunner


class StandOnPolicyRunner(MjlabOnPolicyRunner):
  """Use mjlab checkpoint persistence with the source-compatible StandPPO."""
