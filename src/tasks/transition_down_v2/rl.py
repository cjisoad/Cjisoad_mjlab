"""One PPO clock and optimizer across performance-driven curriculum phases."""
import torch
import copy
from src.tasks.transition_down_t.rl import DownOnPolicyRunner,down_ppo_runner_cfg,augmentation_recipe
from src.tasks.transition_t.rl import TransitionOnPolicyRunner
from src.tasks.transition_t.entry_rl import _recipe_value
from .data import verify_assets
from .env_cfg import landing_env_cfg
from .gates import GateState

def landing_ppo_cfg():
    cfg=down_ppo_runner_cfg();cfg.class_name='LandingRunner';cfg.experiment_name='transition_down_v2'
    return cfg

class LandingRunner(DownOnPolicyRunner):
    def __init__(self,*args,**kwargs):
        self.optimization_recipe={k:copy.deepcopy(args[1][k]) for k in ("algorithm","actor","critic","num_steps_per_env")}
        from .safety import install_monitor
        from .gates import GateConfig
        install_monitor(args[0].unwrapped,GateConfig())
        super().__init__(*args,**kwargs)
        self.gate_config=self.env.unwrapped.landing_monitor.cfg
        self.completed_updates=0;self._learning_active=False
        self.term.bank._groups.clear()
    def _contract(self):
        c=super()._contract()
        c.pop('approved_review_sha256',None)
        c.update(task='transition_down_v2',approved_assets=verify_assets(),
            training_augmentation=augmentation_recipe(landing_env_cfg()),
            training_objective=dict(episode_length_s=self.env.unwrapped.cfg.episode_length_s,
                scale_rewards_by_dt=self.env.unwrapped.cfg.scale_rewards_by_dt,
                rewards=_recipe_value(self.env.unwrapped.cfg.rewards)),
            gate_config=self.gate_config.to_dict(),
            evaluation_split='trajectory_id_mod10_validation5_final0',
            safety_sampling='post_integration_forward_every_physics_step_v1')
        c['endpoint_limits']['strict_world_feet_rms_m']=.08
        return c
    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        if augmentation_recipe(self.env.unwrapped.cfg)!=augmentation_recipe(landing_env_cfg()):
            raise ValueError('V2 training augmentation differs')
        self.current_learning_iteration=self.completed_updates
        self._learning_active=True
        try:
            result=TransitionOnPolicyRunner.learn(self,num_learning_iterations,init_at_random_ep_len=False)
            self.completed_updates=self.current_learning_iteration+1
            return result
        finally:self._learning_active=False
    def save(self,path,infos=None):
        completed=self.current_learning_iteration+1 if self._learning_active else self.completed_updates
        state=dict(version=2,gate_config=self.gate_config.to_dict(),course=self.term.course.to_dict(),
            updates_completed=completed,optimizer_continuous_across_phases=True,optimization_recipe=self.optimization_recipe)
        super().save(path,{**(infos or {}),'landing_training':state})
    def resume(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        state=(saved.get('infos') or {}).get('landing_training')
        if not state or state['gate_config']!=self.gate_config.to_dict() or state.get('optimization_recipe')!=self.optimization_recipe:raise ValueError('V2 resume gate recipe differs')
        if type(state['updates_completed']) is not int or state['updates_completed']<=0:raise ValueError('V2 updates')
        course=GateState.from_dict(state['course'])
        with torch.inference_mode():
            self.load(path,map_location=self.device)
            self.term.course=course;self.completed_updates=state['updates_completed']
            self.current_learning_iteration=self.completed_updates
            rates=[g['lr'] for g in self.alg.optimizer.param_groups]
            if not rates or any(not 0<r<1 or r!=rates[0] for r in rates):raise ValueError('V2 optimizer learning rate')
            self.alg.learning_rate=rates[0]
            self.env.reset()
