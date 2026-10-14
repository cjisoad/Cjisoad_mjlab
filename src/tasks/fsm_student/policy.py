"""Current-frame GRU student with sequence training and batched inference."""
from dataclasses import asdict, dataclass
import math
import re

import torch
from torch import nn

from src.assets.robots.unitree_go2.go2_constants import INIT_STATE
from src.tasks.pedipulation.constants import JOINT_NAMES
from .observations import STUDENT_OBSERVATION_DIM, observation_contract


# FullChainAction uses the transition world's Go2 zero, rather than either
# endpoint teacher's original zero. Derive it from that same source config.
_COMMON_DEFAULT_ANGLES = tuple(next(angle for pattern, angle in INIT_STATE.joint_pos.items()
    if re.fullmatch(pattern, name)) for name in JOINT_NAMES)


@dataclass(frozen=True)
class GRUStudentConfig:
    encoder_dims: tuple[int, ...] = (128,)
    hidden_size: int = 128
    num_layers: int = 1
    head_dims: tuple[int, ...] = (128, 64)
    action_scale: float = .25
    default_angles: tuple[float, ...] = _COMMON_DEFAULT_ANGLES


def _mlp(input_dim, hidden_dims, output_dim=None):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    if output_dim is not None:
        layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class GRUStudent(nn.Module):
    """Trainable actor; hidden state is explicit and never stored by forward.

    Input is [B, 48] for one step or [B, T, 48] for sequence BC. Returns actions
    with the corresponding leading dimensions and hidden [layers, B, hidden].
    Actions are common position offsets: q_target = default + scale*a + offset.
    No output tanh or clipping is imposed on translated teacher action labels.
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = GRUStudentConfig() if cfg is None else cfg
        widths = (*self.cfg.encoder_dims, self.cfg.hidden_size, self.cfg.num_layers, *self.cfg.head_dims)
        if any(type(width) is not int or width <= 0 for width in widths):
            raise ValueError('Student layer sizes must be positive integers')
        if not math.isfinite(self.cfg.action_scale) or self.cfg.action_scale <= 0:
            raise ValueError('Student action scale must be positive and finite')
        if len(self.cfg.default_angles) != 12 or not all(math.isfinite(x) for x in self.cfg.default_angles):
            raise ValueError('Student requires 12 finite common default angles')
        self.encoder = _mlp(STUDENT_OBSERVATION_DIM, self.cfg.encoder_dims)
        encoded_dim = self.cfg.encoder_dims[-1] if self.cfg.encoder_dims else STUDENT_OBSERVATION_DIM
        self.gru = nn.GRU(encoded_dim, self.cfg.hidden_size, self.cfg.num_layers, batch_first=True)
        self.head = _mlp(self.cfg.hidden_size, self.cfg.head_dims, 12)

    def initial_hidden(self, num_envs):
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError('Student requires a positive number of environments')
        return next(self.parameters()).new_zeros(self.cfg.num_layers, num_envs, self.cfg.hidden_size)

    def forward(self, observations, hidden=None, *, episode_starts=None):
        if (observations.ndim not in (2, 3) or observations.shape[-1] != STUDENT_OBSERVATION_DIM
                or observations.shape[0] == 0 or (observations.ndim == 3 and observations.shape[1] == 0)):
            raise ValueError('Student observations must be [B, 48] or [B, T, 48] with nonempty B/T')
        single_step = observations.ndim == 2
        sequence = observations[:, None] if single_step else observations
        count, length = sequence.shape[:2]
        if hidden is None:
            hidden = self.initial_hidden(count)
        if hidden.shape != (self.cfg.num_layers, count, self.cfg.hidden_size):
            raise ValueError('Student hidden state must be [layers, batch, hidden_size]')
        if episode_starts is not None:
            expected_shape = (count,) if single_step else (count, length)
            if episode_starts.shape != expected_shape or episode_starts.dtype != torch.bool:
                raise ValueError('Episode starts must be a boolean mask matching observation B/T dimensions')
            starts = episode_starts[:, None] if single_step else episode_starts
        encoded = self.encoder(sequence)
        if episode_starts is None:
            features, hidden = self.gru(encoded, hidden)
        else:
            steps = []
            for index in range(length):
                hidden = hidden.masked_fill(starts[:, index][None, :, None], 0.)
                feature, hidden = self.gru(encoded[:, index:index+1], hidden)
                steps.append(feature)
            features = torch.cat(steps, dim=1)
        actions = self.head(features)
        return (actions[:, 0] if single_step else actions), hidden

    def action_contract(self):
        return {
            'dimension': 12,
            'joint_names': list(JOINT_NAMES),
            'representation': 'common_joint_position_offset',
            'scale': self.cfg.action_scale,
            'default_angles': list(self.cfg.default_angles),
            'bc_label': 'applied_teacher_target_for_this_control_step_in_common_action_coordinates',
        }

    def save(self, path):
        """Save weights/config/contracts; environment-specific memory is excluded."""
        torch.save({
            'version': 2,
            'config': asdict(self.cfg),
            'state_dict': self.state_dict(),
            'observation_contract': observation_contract(),
            'action_contract': self.action_contract(),
        }, path)

    @classmethod
    def load(cls, path, *, device='cpu'):
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved.get('version') != 2 or saved.get('observation_contract') != observation_contract():
            raise ValueError('Checkpoint does not match the GRU student observation contract')
        model = cls(GRUStudentConfig(**saved['config'])).to(device)
        if saved.get('action_contract') != model.action_contract():
            raise ValueError('Checkpoint does not match the GRU student action contract')
        model.load_state_dict(saved['state_dict'], strict=True)
        return model


class GRUStudentPolicy:
    """Stateful inference wrapper. Advance exactly once per control period.

    Reset memory on physical environment resets, not teacher state transitions
    or ordinary changes to the user's command. Use model.forward for training.
    """
    def __init__(self, model, *, num_envs):
        self.model = model
        self.model.eval()
        self.hidden = model.initial_hidden(num_envs)

    @torch.no_grad()
    def reset(self, env_ids=None):
        if env_ids is None:
            self.hidden.zero_()
        else:
            self.hidden[:, env_ids] = 0.

    @torch.no_grad()
    def act(self, observations, *, reset_mask=None):
        if observations.shape != (self.hidden.shape[1], STUDENT_OBSERVATION_DIM):
            raise ValueError('Inference requires one current observation per environment')
        actions, self.hidden = self.model(observations, self.hidden, episode_starts=reset_mask)
        return actions

    __call__ = act
