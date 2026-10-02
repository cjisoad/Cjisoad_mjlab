"""Verify a fresh stance course, PPO, velocity observations, reload and ONNX.

Use this smoke for course checkpoints. The old migration smoke exercises a
historic curriculum and cannot validate the versioned stance course.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'disable')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from src.tasks.leg_manip.env_cfg import leg_manip_env_cfg
from src.tasks.leg_manip.mdp.observations import base_velocity
from src.tasks.leg_manip.rl import LegManipOnPolicyRunner, leg_manip_ppo_runner_cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu')
    p.add_argument('--num-envs',type=int,default=4)
    p.add_argument('--iterations',type=int,default=2)
    p.add_argument('--result-file',type=Path,required=True)
    args=p.parse_args()
    if args.num_envs < 1 or args.iterations < 1:
        p.error('--num-envs and --iterations must be positive')
    torch.set_num_threads(1)
    cfg=leg_manip_env_cfg()
    cfg.scene.num_envs=args.num_envs
    env=ManagerBasedRlEnv(cfg,device=args.device)
    try:
        agent=leg_manip_ppo_runner_cfg()
        agent.logger='tensorboard';agent.upload_model=False;agent.save_interval=1
        if args.device=='cpu':
            agent.num_steps_per_env=8
            agent.algorithm.num_learning_epochs=1
            agent.algorithm.num_mini_batches=1
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        with tempfile.TemporaryDirectory(prefix='legmanip_stance_smoke_',dir='/tmp') as directory:
            runner=LegManipOnPolicyRunner(wrapped,asdict(agent),directory,args.device)
            term=env.command_manager.get_term('twist')
            assert term.stage==0 and runner.current_learning_iteration==0
            assert not runner.alg.optimizer.state
            obs,_=env.reset()
            assert obs['actor'].shape==(args.num_envs,51)
            assert obs['critic'].shape==(args.num_envs,131)
            torch.testing.assert_close(obs['actor'][:,:48],obs['critic'][:,:48])
            torch.testing.assert_close(obs['critic'][:,48:51],base_velocity(env))
            error=float((obs['actor'][:,48:51]-obs['critic'][:,48:51]).abs().max())
            assert error<=.040001
            assert torch.count_nonzero(term.command[:,:3])==0
            starts=time.monotonic()
            runner.learn(args.iterations,init_at_random_ep_len=False)
            elapsed=time.monotonic()-starts
            assert runner.alg.optimizer.state
            assert all(torch.isfinite(p).all() for model in (runner.alg.actor,runner.alg.critic) for p in model.parameters())
            obs=wrapped.get_observations()
            with torch.inference_mode():expected=runner.get_inference_policy()(obs).clone()
            path=Path(directory)/f'model_{runner.current_learning_iteration}.pt'
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['infos']['leg_manip_observations']['actor_dim']==51
            state=runner._curriculum_state()
            assert any('version' in k for k in state), 'Curriculum state must carry an explicit version'
            second_env=ManagerBasedRlEnv(deepcopy(cfg),device=args.device)
            try:
                restored=LegManipOnPolicyRunner(RslRlVecEnvWrapper(second_env,clip_actions=agent.clip_actions),
                                               asdict(agent),device=args.device)
                restored.load(str(path),map_location=args.device)
                with torch.inference_mode():torch.testing.assert_close(restored.get_inference_policy()(obs),expected)
                restored_state=restored._curriculum_state()
                expected_state=deepcopy(state)
                # Physics starts a new episode on load. Unfinished recorded
                # attempts become interruptions, never synthetic successes.
                for key in ('landing_counts','switch_counts'):
                    counts=expected_state.get(key)
                    if counts is not None:
                        pending=(counts[...,0]-counts[...,1:].sum(-1)).clamp_min(0)
                        counts[...,3] += pending
                assert restored_state.keys()==expected_state.keys()
                for key,value in expected_state.items():
                    if isinstance(value,str):
                        assert restored_state[key]==value
                    else:
                        torch.testing.assert_close(restored_state[key],value)
            finally:
                second_env.close()
            runner.export_policy_to_onnx(directory)
            import onnx
            model=onnx.load(str(Path(directory)/'policy.onnx'))
            onnx.checker.check_model(model)
            assert model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value==51
            result={'device':args.device,'num_envs':args.num_envs,'actor_dim':51,'critic_dim':131,
                    'max_scaled_velocity_noise':error,'exact_critic_velocity_verified':True,
                    'fresh_initial_stage':0,'checkpoint_loaded':False,'ppo_updates':args.iterations,
                    'ppo_seconds':elapsed,'finite_parameters':True,'save_reload_predictions_verified':True,
                    'versioned_course_state_verified':True,'checkpoint_info_keys':list(saved['infos']),
                    'onnx_input_dim':51,'stage_after_smoke':term.stage}
            args.result_file.parent.mkdir(parents=True,exist_ok=True)
            args.result_file.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2),flush=True)
    finally:
        env.close()


if __name__=='__main__':main()
