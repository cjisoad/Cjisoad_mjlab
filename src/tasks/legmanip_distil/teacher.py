"""Frozen original actor policies, queried on the student's live physical state."""
from copy import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from mjlab.envs import mdp

from src.tasks.transition_t.keyboard import AlignedKeyboardTeacher
from src.tasks.pedipulation.mdp.target_limiter import JointTargetLimiter
from src.tasks.transition_t.env_cfg import TRANSITION_TARGET_VELOCITY_LIMITS, TRANSITION_TARGET_ACCELERATION_LIMITS
from src.tasks.tracking.mdp.observations import motion_anchor_pos_b, motion_anchor_ori_b
from src.tasks.loco_pedipulation.mdp.commands import QUAD, HOLD
from .commands import QUADRUPED, RISE, BIPED, DOWN
from .intent import ROOT


def motion_observations(env, reference):
    view = SimpleNamespace(scene=env.scene, action_manager=env.action_manager,
                           num_envs=env.num_envs,
                           command_manager=SimpleNamespace(get_term=lambda name: reference))
    return torch.cat((reference.command, motion_anchor_pos_b(view, 'motion'),
                      motion_anchor_ori_b(view, 'motion'), mdp.base_lin_vel(env),
                      mdp.base_ang_vel(env), mdp.joint_pos_rel(env, biased=True),
                      mdp.joint_vel_rel(env), mdp.last_action(env)), dim=-1)


def _load_transition_actor(kind, path, motion_path, bank_path, device):
    """Validate against the checkpoint's original physical/reference contract."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from src.tasks.transition_t.env_cfg import transition_env_cfg
    from src.tasks.transition_t.rl import TransitionOnPolicyRunner, transition_ppo_runner_cfg
    from src.tasks.transition_down_v2.env_cfg import landing_env_cfg
    from src.tasks.transition_down_v2.rl import LandingRunner, landing_ppo_cfg
    cfg = transition_env_cfg(play=True) if kind == RISE else landing_env_cfg(play=True, split='train')
    cfg.commands['motion'].motion_file = str(motion_path)
    if kind == DOWN:
        cfg.commands['motion'].entry_bank_file = str(bank_path)
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=10.)
        runner_cls, runner_cfg = ((TransitionOnPolicyRunner, transition_ppo_runner_cfg())
                                  if kind == RISE else (LandingRunner, landing_ppo_cfg()))
        runner = runner_cls(wrapped, asdict(runner_cfg), device=device)
        kwargs = {'allow_legacy_target_limiter': True} if kind == RISE else {}
        runner.load(str(path), load_cfg={'actor': True}, map_location=device, **kwargs)
        return runner.get_inference_policy(device=device)
    finally:
        env.close()


class FSMTeacher(torch.nn.Module):
    def __init__(self, env, quad_checkpoint_file=None, biped_checkpoint_file=None,
                 rise_checkpoint_file=None, down_checkpoint_file=None):
        super().__init__()
        if 'twist' not in env.cfg.commands:
            raise ValueError('The frozen teacher requires the training FSM environment')
        self.env = env
        self.term = env.command_manager.get_term('twist')
        self.action = env.action_manager.get_term('joint_pos')
        self.paths = {name: Path(value or ROOT / f'mark/{default}_teacher.pt').resolve()
                      for name, default, value in (
                          ('quadruped', 'quad', quad_checkpoint_file),
                          ('biped', 'biped', biped_checkpoint_file),
                          ('rise', 'up', rise_checkpoint_file),
                          ('down', 'down', down_checkpoint_file))}
        for name, path in self.paths.items():
            if not path.is_file():
                raise FileNotFoundError(f'{name} teacher checkpoint missing: {path}')
        # Scratch validation worlds seed their own simulators. They must not
        # perturb the sampling stream of this already initialized rollout.
        device = torch.device(env.device)
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.quad = AlignedKeyboardTeacher(self.paths['quadruped'], 'loco_pedipulation_t', env.device)
            self.biped = AlignedKeyboardTeacher(self.paths['biped'], 'pedipulation_t', env.device)
            self.actors = torch.nn.ModuleDict({'quadruped': self.quad.actor, 'biped': self.biped.actor})
            for kind, name in ((RISE, 'rise'), (DOWN, 'down')):
                self.actors[name] = _load_transition_actor(kind, self.paths[name],
                    self.term.references[kind].cfg.motion_file,
                    env.cfg.commands['intent'].entry_bank_file, env.device)
        self.requires_grad_(False)
        self.eval()
        self.limiter = JointTargetLimiter(self.action.executed_position,
            TRANSITION_TARGET_VELOCITY_LIMITS, TRANSITION_TARGET_ACCELERATION_LIMITS, env.physics_dt)
        self._preview_engaged = self.term.fr_engaged.clone()
        self.loaded = True
        self._metadata = {'checkpoints': {name: {'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in self.paths.items()},
            'references': {str(kind): {'path': reference.cfg.motion_file,
                'sha256': reference.reference.sha256} for kind, reference in self.term.references.items()},
            'entry_split': env.cfg.commands['intent'].entry_split,
            'entry_bank_sha256': hashlib.sha256(Path(env.cfg.commands['intent'].entry_bank_file).read_bytes()).hexdigest(),
            'label': 'final_substep_applied_common_action',
            'thresholds': [self.term.cfg.rise_trigger_height, self.term.cfg.down_trigger_height,
                           self.term.cfg.trigger_seconds]}

    @torch.inference_mode()
    def requests(self):
        result = torch.zeros_like(self.action.raw_action)
        for owner, teacher in ((QUADRUPED, self.quad), (BIPED, self.biped)):
            ids = (self.term.owner == owner).nonzero().flatten()
            if len(ids):
                result[ids] = teacher.actions(self.env, ids)
        for owner, name in ((RISE, 'rise'), (DOWN, 'down')):
            ids = (self.term.owner == owner).nonzero().flatten()
            if len(ids):
                obs = motion_observations(self.env, self.term.references[owner])[ids]
                result[ids] = self.actors[name](TensorDict({'actor': obs}, batch_size=[len(ids)]))
        if not torch.isfinite(result).all():
            raise FloatingPointError('Frozen teacher produced a nonfinite action')
        return result

    @torch.inference_mode()
    def preview(self, requests):
        if requests.shape != self.action.raw_action.shape or not torch.isfinite(requests).all():
            raise ValueError('Teacher requests must be finite [num_envs, 12] values')
        action, term = self.action, self.term
        endpoint = (term.owner == QUADRUPED) | (term.owner == BIPED)
        current = requests.clamp(-10., 10.)
        current[endpoint] = requests[endpoint].clamp(-100., 100.)
        limiter = copy(self.limiter)
        limiter.position = action.executed_position.clone()
        limiter.velocity = action.executed_velocity.clone()
        engaged = term.fr_engaged.clone()
        engaged[(term.owner == QUADRUPED) & (term.phase == QUAD)] = False
        engaged |= (term.owner != QUADRUPED) | ((term.phase == HOLD) & (term.blend >= 1.))
        bypass = (term.owner == QUADRUPED) & ~engaged
        device = torch.device(self.env.device)
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            delay = torch.empty_like(action.delay_steps).random_(0, self.env.cfg.decimation)
        targets = []
        for substep in range(self.env.cfg.decimation):
            delayed = (term.owner != RISE)[:, None] & (substep < delay)
            selected = torch.where(delayed, action.last_request, current)
            target = action.default_angles + action.cfg.scale * selected + action.motor_offsets
            previous = limiter.position[bypass].clone()
            limiter.step(target, validate=False)
            limiter.position[bypass] = target[bypass]
            limiter.velocity[bypass] = ((target[bypass] - previous) / limiter.dt).clamp(
                -limiter.velocity_limits, limiter.velocity_limits)
            targets.append(limiter.position.clone())
        targets = torch.stack(targets)
        return (targets[-1] - action.default_angles - action.motor_offsets) / action.cfg.scale, targets, current, engaged

    @torch.inference_mode()
    def forward(self, obs=None):
        labels, targets, requests, engaged = self.preview(self.requests())
        self.action.teacher_targets.copy_(targets)
        self.action.teacher_requests.copy_(requests)
        self.action.teacher_plan_ready = True
        self._preview_engaged.copy_(engaged)
        return labels

    def set_control_mask(self, mask):
        if mask.shape != (self.env.num_envs,) or mask.dtype != torch.bool:
            raise ValueError('Teacher control mask must be boolean [num_envs]')
        self.action.teacher_mask.copy_(mask)
        self.term.fr_engaged.copy_(self._preview_engaged)

    def reset(self, dones=None, hidden_state=None):
        pass  # Physical and reference resets are owned by the environment.

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        pass

    def metadata(self):
        return self._metadata
