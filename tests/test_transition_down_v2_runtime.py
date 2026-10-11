import unittest,tempfile
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from dataclasses import asdict
import torch

class LandingRuntime(unittest.TestCase):
    def test_real_substeps_ppo_and_strict_resume(self):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from src.tasks.transition_down_v2.env_cfg import landing_env_cfg
        from src.tasks.transition_down_v2.safety import install_monitor
        from src.tasks.transition_down_v2.rl import LandingRunner,landing_ppo_cfg
        from src.tasks.transition_down_v2.gates import GateConfig
        torch.set_num_threads(1)
        cfg=landing_env_cfg();cfg.scene.num_envs=2
        env=ManagerBasedRlEnv(cfg,device='cpu')
        try:
            m=install_monitor(env,GateConfig());wrapped=RslRlVecEnvWrapper(env,clip_actions=10.)
            agent=landing_ppo_cfg();agent.num_steps_per_env=8;agent.algorithm.num_learning_epochs=1;agent.algorithm.num_mini_batches=2
            with tempfile.TemporaryDirectory() as directory:
                runner=LandingRunner(wrapped,asdict(agent),directory,device='cpu')
                obs,_=env.reset(seed=42)
                self.assertEqual(obs['actor'].shape,(2,75))
                with torch.inference_mode():env.step(torch.zeros(2,12))
                self.assertEqual(m.physics_samples_total,4)
                before=[p.detach().clone() for p in runner.alg.actor.parameters()]
                runner.learn(2)
                self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,runner.alg.actor.parameters())))
                runner.term.course.phase='entry';runner.term.course.level=2
                runner.completed_updates=2
                path=Path(directory)/'resume.pt';runner.save(str(path))
                optimizer=runner.alg.optimizer.state_dict()
                runner.term.course.level=0;runner.completed_updates=0
                runner.resume(path)
                self.assertEqual((runner.term.course.phase,runner.term.course.level),('entry',2))
                self.assertEqual(runner.completed_updates,2)
                runner.term.course.phase='acceptance'
                runner.save(str(path))
                runner.resume(path)
                self.assertEqual(runner.completed_updates,2)
                self.assertEqual(runner.term.course.phase,'acceptance')
                self.assertEqual(len(runner.alg.optimizer.state_dict()['state']),len(optimizer['state']))
                from src.tasks.transition_down_v2.training import train
                from src.tasks.transition_down_t.data import ROOT
                def failed_eval(*args,**kw):
                    return dict(attempts=kw['count'],successes=0,unsafe_attempts=0,
                        distinct_trajectories=kw['count'],records=[])
                args=SimpleNamespace(output=Path(directory)/'acceptance_resume',gate_config=None,
                    quad_teacher=ROOT/'mark/quad_teacher.pt',biped_teacher=ROOT/'mark/biped_teacher.pt',
                    seed=42,num_envs=2,device='cpu',smoke=False,resume=path,max_updates=3,eval_interval=2)
                with patch('src.tasks.transition_down_v2.rl.landing_ppo_cfg',return_value=agent), \
                     patch('src.tasks.transition_down_v2.training.evaluate_policy',side_effect=failed_eval), \
                     patch('src.tasks.transition_down_v2.chain.evaluate_chain',side_effect=failed_eval):
                    result=train(args)
                self.assertEqual(result['status'],'acceptance_failed')
                self.assertEqual(result['updates_completed'],2)
                self.assertTrue(Path(result['checkpoint']).is_file())
                runner.gate_config=GateConfig(nominal_rate=.89)
                with self.assertRaises(ValueError):runner.resume(path)
        finally:env.close()

if __name__=='__main__':unittest.main()
