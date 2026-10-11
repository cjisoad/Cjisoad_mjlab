"""Original reverse PPO environment with reviewed V2 data and knee shaping."""
from dataclasses import fields
from mjlab.managers import RewardTermCfg,TerminationTermCfg
from src.tasks.transition_down_t.env_cfg import down_env_cfg
from src.tasks.transition_t.env_cfg import _ground_sensor
from src.tasks.pedipulation.robot import NONFOOT_BODIES
from .commands import LandingCommandCfg
from .safety import knee_penalty,landing_failure

def landing_env_cfg(play=False,*,split='train'):
    cfg=down_env_cfg(play=play)
    old=cfg.commands['motion']
    values={f.name:getattr(old,f.name) for f in fields(old) if f.name!='motion_file'}
    cfg.commands['motion']=LandingCommandCfg(**values,entry_split=split)
    cfg.commands['motion'].first_frame_probability=0.
    cfg.scene.sensors=(*cfg.scene.sensors,_ground_sensor('landing_nonfoot',NONFOOT_BODIES))
    cfg.rewards['knee_clearance']=RewardTermCfg(func=knee_penalty,weight=-.5)
    cfg.terminations['landing_safety']=TerminationTermCfg(func=landing_failure)
    cfg.episode_length_s=5.5
    return cfg
