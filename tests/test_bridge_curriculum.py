import unittest


class TrackingCourseTests(unittest.TestCase):
  def course(self, **kwargs):
    from src.tasks.pedipulation_bridge_t.curriculum import TrackingCourse
    return TrackingCourse(minimum_episodes=4, window_size=8, minimum_steps=10, **kwargs)

  def test_only_three_stages_and_initial_stage_validation(self):
    for stage in range(3):
      course=self.course(stage=stage)
      self.assertEqual(course.stage,stage)
    with self.assertRaises(ValueError): self.course(stage=3)

  def test_reference_endpoint_resets_cannot_promote(self):
    course=self.course()
    course.record([True]*8,[False]*8,[False]*8,step=100)
    self.assertEqual(course.stage,0)
    course.record([True]*4,[True]*4,[False]*4,step=100)
    self.assertEqual(course.stage,1)

  def test_real_source_success_controls_third_stage_ramp(self):
    course=self.course(stage=2)
    self.assertEqual(course.live_probability,.25)
    self.assertEqual(course.randomization_fraction,.25)
    course.record([True]*4,[True]*4,[False]*4,step=100)
    self.assertEqual(course.live_probability,.25)
    course.record([True]*4,[False]*4,[True]*4,step=100)
    self.assertEqual(course.live_probability,.5)
    self.assertEqual(course.randomization_fraction,.5)

  def test_freeze_and_roundtrip(self):
    course=self.course(enabled=False)
    course.record([True]*8,[True]*8,[False]*8,step=100)
    self.assertEqual(course.stage,0)
    clone=self.course()
    clone.load_state_dict(course.state_dict())
    self.assertEqual(clone.state_dict(),course.state_dict())


if __name__=='__main__': unittest.main()
