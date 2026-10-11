import unittest
from src.tasks.transition_down_v2.gates import GateConfig, GateState, qualifies

def report(n=100, rate=.95, unsafe=0, distinct=None):
    return dict(attempts=n, successes=int(n*rate), unsafe_attempts=unsafe,
                distinct_trajectories=n if distinct is None else distinct)

class AdaptiveGates(unittest.TestCase):
    def test_missing_evidence_and_unsafe_cannot_qualify(self):
        cfg=GateConfig()
        self.assertFalse(qualifies({},.9,64,distinct=True))
        self.assertFalse(qualifies(report(64,1,1),.85,64,distinct=True))
        self.assertFalse(qualifies(report(64,1,0,63),.85,64,distinct=True))
        self.assertTrue(qualifies(report(64,.9),.85,64,distinct=True))

    def test_no_fixed_round_count_and_two_good_windows(self):
        s=GateState();c=GateConfig()
        evidence={'nominal':report(),'push_forward':report(32),'push_backward':report(32)}
        self.assertFalse(s.observe(evidence,c))
        self.assertEqual(s.phase,'nominal')
        self.assertTrue(s.observe(evidence,c))
        self.assertEqual((s.phase,s.level),('entry',0))
        self.assertEqual(s.settings['entry_probability'],.2)

    def test_failure_breaks_streak_and_final_requires_last_level(self):
        s=GateState(phase='entry',level=0);c=GateConfig()
        good={'nominal':report(),'entry':report(64,.9)}
        bad={'nominal':report(),'entry':report(64,.9,1)}
        s.observe(good,c);s.observe(bad,c);s.observe(good,c)
        self.assertEqual(s.level,0)
        self.assertTrue(s.observe(good,c))
        self.assertEqual(s.level,1)
        for _ in range(4):s.observe(good,c)
        self.assertEqual((s.phase,s.level),('entry',3))
        s.observe(good,c);s.observe(good,c)
        self.assertEqual(s.phase,'acceptance')

    def test_resume_roundtrip_and_validation(self):
        s=GateState();s.observe({'nominal':report(),'push_forward':report(32),'push_backward':report(32)},GateConfig())
        self.assertEqual(GateState.from_dict(s.to_dict()),s)
        with self.assertRaises(ValueError):GateConfig(nominal_rate=1.1)
        with self.assertRaises(ValueError):GateState.from_dict({'phase':'done','level':0,'streak':0,'evaluations':0})

if __name__=='__main__':unittest.main()
