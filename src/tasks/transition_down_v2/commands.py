"""V2 reference/teacher resets with unchanged execution caches and heading."""
from dataclasses import dataclass
import torch
from src.tasks.transition_t.entry_course import EntryCommand,EntryCommandCfg
from src.tasks.transition_down_t.commands import DownMotion,align_reference
from .data import MOTION_FILE,BANK_FILE,verify_assets,LandingBank
from .gates import GateState

@dataclass(kw_only=True)
class LandingCommandCfg(EntryCommandCfg):
    motion_file: str = str(MOTION_FILE)
    entry_bank_file: str = str(BANK_FILE)
    def build(self,env):return LandingCommand(self,env)

class LandingCommand(EntryCommand):
    def __init__(self,cfg,env):
        verify_assets(cfg.motion_file,cfg.entry_bank_file)
        super().__init__(cfg,env)
        # Reuse validated tensors while extending the split selector.
        bank=object.__new__(LandingBank);bank.__dict__.update(self.bank.__dict__);self.bank=bank
        self.course=GateState()
    def restore_entries(self,ids,selected):
        super().restore_entries(ids,selected)
        action=self._env.action_manager.get_term('joint_pos')
        action._requested_action[ids]=self.bank.data['raw_action'][selected]
        action._previous_requested_action[ids]=self.bank.data['previous_action'][selected]
        root=self.bank.data['root_state'][selected].clone();root[:,:3]+=self._env.scene.env_origins[ids]
        align_reference(self,ids,root[:,:3],root[:,3:7])
        actual_pos,actual_quat=self.robot_anchor_pos_w.clone(),self.robot_anchor_quat_w.clone()
        actual_pos[ids],actual_quat[ids]=root[:,:3],root[:,3:7]
        self._refresh_relative(actual_pos,actual_quat)
    start_from_current_state=DownMotion.start_from_current_state
