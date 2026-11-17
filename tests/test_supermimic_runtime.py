"""Real MuJoCo joint updates, resume and teacher-free student export."""
from dataclasses import asdict
from pathlib import Path
import torch
from tensordict import TensorDict


def test_joint_training_resume_and_standalone_playback(tmp_path):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from src.tasks.legmanip_distil.supermimic.config import SuperMimicRunnerCfg
    from src.tasks.legmanip_distil.supermimic.env import supermimic_env_cfg
    from src.tasks.legmanip_distil.supermimic.training import SuperMimicRunner
    from src.tasks.legmanip_distil.rl.model import load_student
    from src.tasks.legmanip_distil.deployment import validate_contract
    cfg = SuperMimicRunnerCfg()
    cfg.num_steps_per_env = cfg.algorithm.gradient_length = 4
    cfg.algorithm.num_learning_epochs = 1
    cfg.save_interval = 1
    cfg.supermimic.hidden_dim = 32
    cfg.supermimic.joint_updates = 1
    cfg.supermimic.gradient_interval = 1
    cfg.supermimic.detailed_gradient_interval = 1
    cfg.supermimic.validation_interval = 1
    cfg.supermimic.validation_batch_size = 4
    cfg.supermimic.max_horizon = 4
    cfg.supermimic.model_updates = 1
    cfg.supermimic.model_batch_size = 4
    cfg.supermimic.imagination_start_update = 0
    def make(path):
        env_cfg = supermimic_env_cfg()
        env_cfg.scene.num_envs = 2
        env_cfg.episode_length_s = .08
        env_cfg.commands['intent'].scenario = 'cycle'
        env = ManagerBasedRlEnv(env_cfg, device='cpu')
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        runner = SuperMimicRunner(wrapped, asdict(cfg), str(path), 'cpu')
        return env, runner
    env, runner = make(tmp_path/'first')
    try:
        initial = [p.detach().clone() for p in runner.alg.world_model.parameters()]
        teacher_initial = [p.detach().clone() for p in runner.alg.teacher.parameters()]
        runner.learn(2)
        assert runner.alg.num_updates == 2
        assert any(not torch.equal(a,b) for a,b in zip(initial,runner.alg.world_model.parameters()))
        for a,b in zip(teacher_initial,runner.alg.teacher.parameters()):
            torch.testing.assert_close(a,b)
        checkpoint = tmp_path/'first/model_2.pt'
        saved = torch.load(checkpoint,weights_only=False)
        assert 'supermimic' in saved and len(saved['supermimic']['replay']) == 8
        assert saved['supermimic']['version'] == 2
        assert saved['supermimic']['gradient_monitor']
        assert 'real_features' in saved['supermimic']['replay'][0]
        assert 'validation_scores' in saved['supermimic']
        assert saved['supermimic']['replay'][3]['done'].all()
        assert saved['supermimic']['replay'][7]['done'].all()
        for module in (runner.alg.student,runner.alg.world_model,runner.alg.critic):
            assert all(torch.isfinite(p).all() for p in module.parameters())
    finally:
        env.close()
    env, resumed = make(tmp_path/'resumed')
    try:
        resumed.load(str(checkpoint))
        assert resumed.alg.num_updates == 2
        assert resumed.alg.replay.steps[-1]['done'].all()
        resumed.learn(1)
        assert resumed.alg.num_updates == 3
        assert (tmp_path/'resumed/model_3.pt').is_file()
    finally:
        env.close()
    play_cfg = supermimic_env_cfg(play=True)
    play_cfg.scene.num_envs = 2
    play = ManagerBasedRlEnv(play_cfg,device='cpu')
    try:
        student,metadata = load_student(tmp_path/'resumed/student.pt',num_envs=2)
        validate_contract(metadata,play)
        assert set(play.cfg.commands) == {'intent'}
        with torch.inference_mode():
            obs,_=play.reset()
            for _ in range(8):
                actions=student(TensorDict(obs,batch_size=[2]))
                obs,_,terminated,truncated,_=play.step(actions)
                student.reset(terminated|truncated)
                assert torch.isfinite(obs['student']).all()
    finally:
        play.close()
