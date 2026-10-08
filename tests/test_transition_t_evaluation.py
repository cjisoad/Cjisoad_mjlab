import unittest
import torch
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('transition_t_eval', Path(__file__).resolve().parents[1] / 'scripts/evaluate_transition_t.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EndpointHold = module.EndpointHold


class EndpointCriteria(unittest.TestCase):
  def test_requires_full_motion_and_continuous_hold_with_contacts(self):
    hold = EndpointHold(.02)
    good = dict(completed=False, orientation_error=.1, height_error=.01,
                feet_rms=.01, linear_speed=.1, angular_speed=.1,
                rear_supported=True, front_clear=True)
    for _ in range(60):
      self.assertFalse(hold.update(**good))
    good['completed'] = True
    for _ in range(30): hold.update(**good)
    good['rear_supported'] = False
    self.assertFalse(hold.update(**good))
    self.assertEqual(hold.duration, 0.)
    good['rear_supported'] = True
    for _ in range(49):
      self.assertFalse(hold.update(**good))
    self.assertTrue(hold.update(**good))
    good['front_clear'] = False
    self.assertFalse(hold.update(**good))

  def test_nonfinite_episode_metrics_remain_valid_json(self):
    import json
    record = {'groups': {'base': {'errors': [0., float('nan'), float('inf')],
                                 'termination_causes': ['physical_failure']}}}
    cleaned = module.json_safe(record)
    json.dumps(cleaned, allow_nan=False)
    self.assertEqual(cleaned['groups']['base']['errors'], [0., None, None])
    self.assertEqual(cleaned['groups']['base']['termination_causes'], ['physical_failure'])

  def test_nonfinite_errors_fail(self):
    hold = EndpointHold(.02)
    self.assertFalse(hold.update(True, float('nan'), 0., 0., 0., 0., True, True))


if __name__ == '__main__': unittest.main()
