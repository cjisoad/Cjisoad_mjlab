"""Optional CUDA integration test: PPO, physical ABI and safe teacher resume."""
import unittest
from pathlib import Path
from dataclasses import asdict
from tempfile import TemporaryDirectory
import json
import torch

@unittest.skipUnless(torch.cuda.is_available(), 'CUDA teacher integration requires a GPU')
class TeacherRuntimeTests(unittest.TestCase):
  def test_full_ppo_checkpoint_and_custom_experiment_contract(self):
    with TemporaryDirectory(prefix='mjlab_teachers_') as temporary:
      import src.tasks
      from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
      from mjlab.envs import ManagerBasedRlEnv
      from mjlab.rl import RslRlVecEnvWrapper
      from src.tasks.pedipulation.constants import JOINT_NAMES, DEFAULT_ANGLES
      from src.tasks.leg_manip.mdp.observations import base_velocity

      torch.set_num_threads(1)
      output = Path(temporary)
      output.mkdir(parents=True, exist_ok=True)
      records = []
      for task, critic_dim in (('pedipulation_t', 89), ('loco_pedipulation_t', 95)):
        print('BUILD', task, flush=True)
        cfg = load_env_cfg(task)
        cfg.scene.num_envs = 64
        cfg.commands['twist'].debug_vis = False
        agent = load_rl_cfg(task)
        agent.logger = 'tensorboard'
        agent.upload_model = False
        agent.experiment_name = 'custom_teacher_experiment_' + task
        env = ManagerBasedRlEnv(cfg, device='cuda:0')
        try:
          wrapped = RslRlVecEnvWrapper(env, clip_actions=10.)
          runner_cls = load_runner_cls(task)
          runner = runner_cls(wrapped, asdict(agent), device='cuda:0')
          assert runner.teacher_task == task, 'Experiment name must not change task identity'
          term = env.command_manager.get_term('twist')
          action = env.action_manager.get_term('joint_pos')
          assert tuple(action._entity.joint_names[i] for i in action.joint_ids.tolist()) == JOINT_NAMES
          torch.testing.assert_close(action.default_angles, torch.tensor(DEFAULT_ANGLES, device=env.device).expand(64, 12))
          arm = env.sim.model.dof_armature[:, env.scene['robot'].indexing.joint_v_adr]
          assert bool((arm > 0).all()), 'Rotational armature must remain nonzero'
          losses_per_update = []
          for update in range(2):
            old = [p.detach().clone() for p in runner.alg.actor.parameters()]
            with torch.inference_mode():
              for step in range(agent.num_steps_per_env):
                obs = wrapped.get_observations()
                assert obs['actor'].shape == (64, 51) and obs['critic'].shape == (64, critic_dim)
                prefix = slice(3, 51) if task == 'pedipulation_t' else slice(0, 48)
                torch.testing.assert_close(obs['critic'][:, prefix], obs['actor'][:, :48])
                assert float((obs['actor'][:,48:51]-base_velocity(env)).abs().max()) <= .04001
                actions = runner.alg.act(obs)
                obs, reward, done, info = wrapped.step(actions)
                assert torch.isfinite(actions).all() and torch.isfinite(reward).all()
                assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all()
                runner.alg.process_env_step(obs, reward, done, info)
              runner.alg.compute_returns(obs)
            losses = runner.alg.update()
            assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
            assert any(not torch.equal(before, after) for before, after in zip(old, runner.alg.actor.parameters()))
            losses_per_update.append(losses)
          if task == 'loco_pedipulation_t':
            term.stage = 2
            term.stage_started = 10
            term.curriculum_window[:] = torch.arange(10, device=env.device)
            term.curriculum_episodes = 7
          runner.current_learning_iteration = 2
          checkpoint = output/(task+'_model_2.pt')
          runner.save(str(checkpoint))
          with torch.no_grad(): expected = runner.alg.actor(wrapped.get_observations()).clone()
          if task == 'loco_pedipulation_t':
            term.stage = 0
            term.stage_started = 0
            term.curriculum_window.zero_()
            term.curriculum_episodes = 0
          restored = runner_cls(wrapped, asdict(agent), device='cuda:0')
          restored.load(str(checkpoint))
          with torch.no_grad(): actual = restored.alg.actor(wrapped.get_observations())
          torch.testing.assert_close(actual, expected, rtol=0, atol=0)
          assert restored.current_learning_iteration == 2 and restored.alg.optimizer.state
          action.cfg.scale = .3
          assert restored._contract()['action_scale'] == .3
          try:
            restored.load(str(checkpoint))
          except ValueError:
            pass
          else:
            raise AssertionError('Actual action scale mismatch must be rejected')
          action.cfg.scale = .25
          wrapped.clip_actions = 5.
          assert restored._contract()['clip_actions'] == 5.
          try:
            restored.load(str(checkpoint))
          except ValueError:
            pass
          else:
            raise AssertionError('Actual clip mismatch must be rejected')
          wrapped.clip_actions = 10.
          if task == 'loco_pedipulation_t':
            assert term.stage == 2 and term.stage_started == 10 and term.curriculum_episodes == 7
            torch.testing.assert_close(term.curriculum_window, torch.arange(10, device=env.device, dtype=torch.float32))
          incompatible = torch.load(checkpoint, weights_only=False)
          incompatible['infos']['teacher_contract']['anchor'] = 'old_source_frame'
          bad = output/(task+'_incompatible.pt')
          torch.save(incompatible, bad)
          try:
            restored.load(str(bad))
          except ValueError:
            pass
          else:
            raise AssertionError('Incompatible teacher checkpoint must be rejected')
          record = {'task':task, 'num_envs':64, 'actor_dim':51, 'critic_dim':critic_dim,
            'updates':2, 'epochs':agent.algorithm.num_learning_epochs, 'minibatches':agent.algorithm.num_mini_batches,
            'actor_updated':True, 'checkpoint_roundtrip':True, 'incompatible_checkpoint_rejected':True,
            'experiment_rename_safe':True, 'actual_action_contract_validated':True,
            'nonzero_rotor_armature':True, 'losses':losses_per_update}
          records.append(record)
          print('TEACHER_PPO_PASSED', task, flush=True)
        finally:
          env.close()
        torch.cuda.empty_cache()

      print('TEACHER_TASKS_VERIFIED', json.dumps(records), flush=True)

if __name__ == '__main__':
  unittest.main()
