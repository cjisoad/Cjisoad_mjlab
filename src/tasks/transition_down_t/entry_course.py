"""Original physical entry curriculum with biped-stable heading alignment."""
from dataclasses import dataclass, fields
from src.tasks.transition_t.entry_course import EntryCommand, EntryCommandCfg
from .commands import align_reference
from .data import ALIGNMENT, DEFAULT_ENTRY_BANK, TEACHER_SHA256, verify_approved_assets
from .env_cfg import down_env_cfg


@dataclass(kw_only=True)
class DownEntryCommandCfg(EntryCommandCfg):
  def build(self,env):return DownEntryCommand(self,env)


class DownEntryCommand(EntryCommand):
  def __init__(self,cfg,env):
    verify_approved_assets(motion_file=cfg.motion_file,bank_file=cfg.entry_bank_file)
    super().__init__(cfg,env)
    metadata=self.bank.metadata
    if metadata.get('transition_direction')!='biped_to_tripod' or metadata.get('reference_alignment')!=ALIGNMENT:
      raise ValueError('Entry direction / heading alignment differs')
    for name,expected in TEACHER_SHA256.items():
      if metadata.get('endpoint_teachers',{}).get(name,{}).get('sha256')!=expected:
        raise ValueError(f'Entry {name} teacher differs')

  def restore_entries(self,ids,selected):
    super().restore_entries(ids,selected)
    # Schema v2 records applied targets, the teacher's action history contract.
    # Initialize delayed request caches from those histories for source-teacher continuation.
    action=self._env.action_manager.get_term('joint_pos')
    action._requested_action[ids]=self.bank.data['raw_action'][selected]
    action._previous_requested_action[ids]=self.bank.data['previous_action'][selected]
    root=self.bank.data['root_state'][selected].clone()
    root[:,:3]+=self._env.scene.env_origins[ids]
    align_reference(self,ids,root[:,:3],root[:,3:7])
    actual_pos,actual_quat=self.robot_anchor_pos_w.clone(),self.robot_anchor_quat_w.clone()
    actual_pos[ids],actual_quat[ids]=root[:,:3],root[:,3:7]
    self._refresh_relative(actual_pos,actual_quat)


def down_entry_env_cfg(bank_file=DEFAULT_ENTRY_BANK,*,motion_file=None,num_envs=4096,play=False,split='train'):
  cfg=down_env_cfg(play=play);cfg.scene.num_envs=num_envs
  old=cfg.commands['motion'];values={field.name:getattr(old,field.name) for field in fields(old)}
  if motion_file is not None:values['motion_file']=str(motion_file)
  cfg.commands['motion']=DownEntryCommandCfg(**values,entry_bank_file=str(bank_file),entry_split=split)
  cfg.commands['motion'].first_frame_probability=0.
  if play:cfg.commands['motion'].sampling_mode='adaptive'
  return cfg
