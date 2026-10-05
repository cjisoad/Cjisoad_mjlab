"""Source command behavior expressed in the common teacher frame/foot zero."""
from dataclasses import dataclass

import torch

from .anchor_frame import anchor_basis, to_anchor
from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace, LOW, HIGH
from src.tasks.pedipulation.mdp.commands import PedipulationCommand, PedipulationCommandCfg
from src.tasks.loco_pedipulation.mdp.commands import LocoPedipulationCommand, LocoPedipulationCommandCfg
from src.tasks.leg_manip.constants import BIPED_HIGH, TRIPOD_HIGH


def _create_teacher_gui(term, server, get_env_idx, *, biped):
  from viser import Icon
  nominal = term.nominal_biped_offset.tolist() if biped else [0., 0., 0.]
  with server.gui.add_folder('Biped teacher' if biped else 'Loco teacher'):
    manual = server.gui.add_checkbox('Manual', initial_value=False)
    bounds = (term.cfg.lin_vel_x, term.cfg.lin_vel_y, getattr(term.cfg, 'ang_vel_z', (-1., 1.)))
    velocity = [server.gui.add_slider(label, min=low, max=high, step=.01, initial_value=0.)
      for label, (low, high) in zip(('vx', 'vy', 'wz'), bounds)]
    offsets = [server.gui.add_slider(label, min=low, max=high, step=.005, initial_value=value)
      for label, low, high, value in (
        ('FR dx', -.12, .30, nominal[0]), ('FR dy', -.10, .10, nominal[1]),
        ('FR dz', 0., BIPED_HIGH if biped else TRIPOD_HIGH, nominal[2]))]
    stop = server.gui.add_button('Stop', icon=Icon.PLAYER_STOP)
    @stop.on_click
    def _(_event):
      for slider in velocity:
        slider.value = 0.
      for slider, value in zip(offsets, nominal):
        slider.value = value
  term._gui = (manual, velocity, offsets, get_env_idx)
  term._gui_previous = None
  term._gui_env = None


def _invalidate_reset_gui(term, env_ids):
  current = getattr(term, '_gui_env', None)
  if current is not None and bool((torch.as_tensor(env_ids, device=term.device) == current).any()):
    term._gui_previous = None


class CommonAnchorFrame:
  """Stateless compatibility view for the original standing reward API."""
  def __init__(self, term):
    self.term = term

  @property
  def basis_w(self):
    return self.term.basis_w

  @property
  def heading(self):
    forward = self.basis_w[:, :, 0]
    return torch.atan2(forward[:, 1], forward[:, 0])

  def to_frame(self, values):
    return to_anchor(self.basis_w, values)

  def reset(self, env_ids):
    pass  # The common teacher anchor has no history.

  def update(self, quat, dt, step):
    pass  # Orientation is read directly from the matched simulator state.


@dataclass(kw_only=True)
class PedipulationTeacherCommandCfg(PedipulationCommandCfg):
  def build(self, env):
    return PedipulationTeacherCommand(self, env)


class PedipulationTeacherCommand(PedipulationCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene['robot']
    self.workspace = build_foot_workspace()
    self.zero = torch.as_tensor(self.workspace.zero, dtype=torch.float32, device=self.device)
    self.nominal_biped_offset = torch.as_tensor(self.workspace.nominal_biped_offset,
      dtype=torch.float32, device=self.device)
    self.target_bank = torch.as_tensor(self.workspace.offsets[self.workspace.segment == HIGH],
      dtype=torch.float32, device=self.device)
    self._foot_zero_anchor = self.zero
    self._foot_offsets = self.target_bank
    self._anchor_frame = CommonAnchorFrame(self)
    self.manual = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  @property
  def basis_w(self):
    return anchor_basis(self.robot.data.root_link_quat_w, self.robot.data.gravity_vec_w)

  @property
  def has_foot_target(self):
    # The common nominal biped target encodes stance. Preserve the source
    # reward's no-operation semantics for that target and the zero sentinel.
    offset = self.command[:, 3:]
    return (offset != 0).any(-1) & ((offset-self.nominal_biped_offset).abs().amax(-1) > 1e-6)

  def _resample_command(self, env_ids):
    ids = env_ids[~self.manual[env_ids]]
    if not len(ids):
      return
    super()._resample_command(ids)
    inactive = (self._command[ids, 3:] == 0).all(-1)
    self._command[ids[inactive], 3:] = self.nominal_biped_offset

  def _update_command(self):
    # Keep the source heading controller for automatic training, but allow
    # direct vx/vy/wz commands for matched-state queries and keyboard play.
    previous = self._command[:, :3].clone()
    super()._update_command()
    self._command[self.manual, :3] = previous[self.manual]

  def set_command(self, env_ids, *, velocity=None, target_offset=None):
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
    values = {}
    for key, value in (('velocity', velocity), ('target_offset', target_offset)):
      if value is not None:
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device).expand(len(ids), 3)
        if not torch.isfinite(value).all():
          raise ValueError(f'{key} must be finite')
        values[key] = value
    self.manual[ids] = True
    if 'velocity' in values:
      self._command[ids, :3] = values['velocity']
    if 'target_offset' in values:
      self._command[ids, 3:] = values['target_offset']

  def release_manual(self, env_ids):
    self.manual[env_ids] = False

  def reset(self, env_ids):
    _invalidate_reset_gui(self, env_ids)
    self.manual[env_ids] = False
    return super().reset(env_ids)

  def create_gui(self, name, server, get_env_idx):
    _create_teacher_gui(self, server, get_env_idx, biped=True)

  def _read_gui(self):
    from src.tasks.leg_manip.mdp.commands import LegManipCommand
    if hasattr(self, '_gui'):
      LegManipCommand._read_gui(self)

  def compute(self, dt):
    self._read_gui()
    super().compute(dt)


@dataclass(kw_only=True)
class LocoPedipulationTeacherCommandCfg(LocoPedipulationCommandCfg):
  def build(self, env):
    return LocoPedipulationTeacherCommand(self, env)


class LocoPedipulationTeacherCommand(LocoPedipulationCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.workspace = build_foot_workspace()
    self.zero = torch.as_tensor(self.workspace.zero, dtype=torch.float32, device=self.device)
    self.target_bank = torch.as_tensor(self.workspace.offsets[self.workspace.segment == LOW],
      dtype=torch.float32, device=self.device)
    self.offsets = self.target_bank
    easy = (self.offsets[:, :2].abs() <= .045).all(-1) & (self.offsets[:, 2] <= .085)
    self.easy_indices = easy.nonzero().flatten()
    if not len(self.easy_indices):
      raise ValueError('Common workspace lacks easy tripod targets')
    self.reference[:] = self.zero
    self.start[:] = self.zero
    self.goal[:] = self.zero

  @property
  def basis_w(self):
    return anchor_basis(self.robot.data.root_link_quat_w, self.robot.data.gravity_vec_w)

  @property
  def foot_target_pos_w(self):
    target = self.zero + self.command[:, 3:]
    return self.robot.data.root_link_pos_w + (self.basis_w @ target[..., None]).squeeze(-1)

  def create_gui(self, name, server, get_env_idx):
    _create_teacher_gui(self, server, get_env_idx, biped=False)

  def reset(self, env_ids):
    _invalidate_reset_gui(self, env_ids)
    return super().reset(env_ids)
