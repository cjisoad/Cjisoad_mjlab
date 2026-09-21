"""Manager-based runner for the Go2 rear-leg standing task."""

from mjlab.rl.runner import MjlabOnPolicyRunner


class PedipulationOnPolicyRunner(MjlabOnPolicyRunner):
  """Use mjlab checkpoint persistence with the source-compatible PedipulationPPO."""
