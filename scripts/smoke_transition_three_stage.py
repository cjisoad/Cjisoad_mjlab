"""Small physical/optimizer reconstruction check; synthetic promotion is NOT ability evidence."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/transition_t_mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/transition_t_cache')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--entry-bank', type=Path, required=True)
  parser.add_argument('--checkpoint', type=Path, required=True, help='New stage-0 scratch checkpoint')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--device', default='cpu')
  args = parser.parse_args()
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from src.tasks.transition_t.three_stage import three_stage_env_cfg
  from src.tasks.transition_t.three_stage_rl import ThreeStageOnPolicyRunner
  from src.tasks.transition_t.rl import transition_ppo_runner_cfg
  from src.tasks.transition_t.entry_evaluation import evaluate_entries

  torch.set_num_threads(1)
  args.output.mkdir(parents=True, exist_ok=False)
  agent = transition_ppo_runner_cfg(); agent.save_interval = 100
  env = None
  rows = []
  previous = args.checkpoint
  expected_state = None
  def check_equal(a,b):
    if isinstance(a, torch.Tensor):
      torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a, dict):
      assert a.keys() == b.keys()
      for key in a: check_equal(a[key],b[key])
    elif isinstance(a, (list,tuple)):
      assert len(a) == len(b)
      for x,y in zip(a,b): check_equal(x,y)
    else: assert a == b
  try:
    for stage in range(3):
      cfg = three_stage_env_cfg(args.entry_bank, stage=stage, num_envs=8)
      cfg.seed = 42
      env = ManagerBasedRlEnv(cfg, device=args.device)
      wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
      runner = ThreeStageOnPolicyRunner(wrapped, asdict(agent), str(args.output/f'stage{stage}'), device=args.device)
      if stage == 0:
        runner.resume(previous)
      else:
        runner.restore_stage_transition(previous)
        check_equal(expected_state, runner.alg.save())
        assert runner.alg.learning_rate == expected_lr
        assert runner.current_learning_iteration == expected_updates
        assert env.common_step_counter == expected_clock
        torch.testing.assert_close(runner.term.bin_failed_count, expected_bins, rtol=0, atol=0)
        assert runner.term.course.level == stage
      robot = env.scene['robot']
      local_body = robot.find_bodies(('base_link',), preserve_order=True)[0][0]
      body = robot.indexing.body_ids[local_body]
      local_geoms = robot.find_geoms('.*_foot_collision')[0] if stage else []
      geoms = robot.indexing.geom_ids[local_geoms] if stage else []
      # Model fields are broadcast-only in nominal stage, expanded per world
      # by startup event requirements in randomized stages.
      model = env.sim.model
      com = model.body_ipos[:, body].detach().cpu()
      default_com = torch.as_tensor(env.sim.mj_model.body_ipos[body], dtype=com.dtype)
      difference = (com-default_com).abs().amax().item()
      if stage == 0:
        assert difference < 1e-7
        friction = [1.,1.]
      else:
        assert difference > 1e-5 and difference <= .01*(stage/2)+1e-7
        f = model.geom_friction[:, geoms, 0].detach().cpu()
        limits = cfg.events['foot_friction'].params['ranges']
        assert f.min() >= limits[0] and f.max() <= limits[1]
        assert f.std() > 1e-5
        friction = [float(f.min()),float(f.max())]
      sensor_cfg = next(s for s in cfg.scene.sensors if s.name == 'knees_ground_contact')
      assert all('foot' not in pattern for pattern in sensor_cfg.primary.pattern)
      matched = {slot.primary_name for slot in env.scene['knees_ground_contact']._slots}
      assert matched and all('foot' not in name for name in matched)
      assert any('RL_calf' in name for name in matched) and any('RR_calf' in name for name in matched)
      start = runner.current_learning_iteration
      runner.learn(1)
      for network in (runner.alg.actor,runner.alg.critic):
        assert all(torch.isfinite(p).all() for p in network.parameters())
      if runner.logger.writer is not None:
        runner.logger.writer.flush(); runner.logger.writer.close(); runner.logger.writer = None
      report = evaluate_entries(runner.alg.actor, args.entry_bank, level=stage,
        device=args.device, attempts=8, seed=424242+stage, curriculum='three-stage',
        initialization='nominal' if stage == 0 else 'entries')
      rows.append(dict(stage=stage, restored_updates=start, com_offset_max=difference,
        friction_range=friction, eval_attempts=report['attempts'], successes=report['successes'],
        distinct_trajectories=report['distinct_trajectories'], diagnostics=report['records'][0]['diagnostics']))
      runner.current_learning_iteration = start
      resumable = args.output/f'stage{stage}_resume.pt'
      runner.save(str(resumable))
      before_resume = deepcopy(runner.alg.save())
      env.close()
      cfg = three_stage_env_cfg(args.entry_bank, stage=stage, num_envs=8)
      cfg.seed = 42
      env = ManagerBasedRlEnv(cfg, device=args.device)
      wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
      runner = ThreeStageOnPolicyRunner(wrapped, asdict(agent),
        str(args.output/f'stage{stage}'), device=args.device)
      runner.resume(resumable)
      check_equal(before_resume, runner.alg.save())
      assert runner.current_learning_iteration == start+1
      assert runner.term.course.level == stage
      rows[-1]['strict_resume_verified'] = True
      runner.current_learning_iteration = start
      if stage < 2:
        runner.term.course.level = stage+1  # Controlled restore test, never a genuine promotion.
        previous = args.output/f'synthetic_stage{stage}_boundary.pt'
        runner.save(str(previous))
        expected_state = deepcopy(runner.alg.save())
        expected_lr, expected_updates = runner.alg.learning_rate, start+1
        expected_clock = env.common_step_counter
        expected_bins = runner.term.bin_failed_count.clone()
      else:
        previous = resumable
      env.close(); env = None
    report = dict(test='synthetic_stage_reconstruction_not_promotion_evidence', stages=rows,
      final_checkpoint=str(previous))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
  finally:
    if env is not None: env.close()


if __name__ == '__main__':
  main()
