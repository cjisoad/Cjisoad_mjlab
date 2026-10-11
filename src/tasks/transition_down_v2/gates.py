"""Pure evaluation-driven curriculum; update counts are resource budgets only."""
from dataclasses import asdict, dataclass
import math
from src.tasks.transition_t.entry_bank import LEVELS

@dataclass(frozen=True)
class GateConfig:
    version: int = 2
    nominal_rate: float = .90
    probe_rate: float = .85
    entry_rate: float = .85
    final_rate: float = .90
    final_nominal_rate: float = .95
    consecutive_windows: int = 2
    nominal_attempts: int = 100
    probe_attempts: int = 32
    entry_attempts: int = 64
    knee_margin_m: float = .040
    knee_target_m: float = .050
    training_knee_stop_m: float = .025
    landing_deadline_s: float = 2.4
    landing_contact_s: float = .060
    hold_s: float = 1.
    timeout_s: float = 5.
    def __post_init__(self):
        if self.version!=2:raise ValueError('gate version')
        for k in ('nominal_rate','probe_rate','entry_rate','final_rate','final_nominal_rate'):
            v=getattr(self,k)
            if not math.isfinite(v) or not 0<=v<=1:raise ValueError(k)
        for k in ('consecutive_windows','nominal_attempts','probe_attempts','entry_attempts'):
            if type(getattr(self,k)) is not int or getattr(self,k)<1:raise ValueError(k)
        for k in ('knee_margin_m','knee_target_m','training_knee_stop_m','landing_deadline_s','landing_contact_s','hold_s','timeout_s'):
            if not math.isfinite(getattr(self,k)) or getattr(self,k)<=0:raise ValueError(k)
        if not self.training_knee_stop_m < self.knee_margin_m <= self.knee_target_m:raise ValueError('knee margins')
    def to_dict(self):return asdict(self)

def qualifies(report, rate, minimum, *, distinct=False):
    try:
        n=report['attempts'];good=report['successes'];unsafe=report['unsafe_attempts']
        if any(type(v) is not int for v in (n,good,unsafe)):return False
        if n<minimum or not 0<=good<=n or unsafe!=0:return False
        if distinct and report.get('distinct_trajectories',0)<minimum:return False
        return good/n>=rate
    except (KeyError,TypeError,ZeroDivisionError):return False

@dataclass
class GateState:
    phase: str = 'nominal'
    level: int = 0
    streak: int = 0
    evaluations: int = 0
    @property
    def settings(self):
        if self.phase=='nominal':
            return dict(LEVELS[0],entry_probability=0.,nominal_probability=.5)
        return dict(LEVELS[self.level])
    def observe(self, reports, cfg):
        self.evaluations+=1
        good=qualifies(reports.get('nominal',{}),cfg.nominal_rate,cfg.nominal_attempts)
        if self.phase=='nominal':
            good &= all(qualifies(reports.get(name,{}),cfg.probe_rate,cfg.probe_attempts) for name in ('push_forward','push_backward'))
        elif self.phase=='entry':
            good &= qualifies(reports.get('entry',{}),cfg.entry_rate,cfg.entry_attempts,distinct=True)
        else:return False
        self.streak=self.streak+1 if good else 0
        if self.streak<cfg.consecutive_windows:return False
        self.streak=0
        if self.phase=='nominal':self.phase='entry'
        elif self.level<len(LEVELS)-1:self.level+=1
        else:self.phase='acceptance'
        return True
    def to_dict(self):return asdict(self)
    @classmethod
    def from_dict(cls,value):
        result=cls(**value)
        if result.phase not in ('nominal','entry','acceptance'):raise ValueError('phase')
        if any(type(getattr(result,k)) is not int or getattr(result,k)<0 for k in ('level','streak','evaluations')) or result.level>=len(LEVELS):raise ValueError('course state')
        return result
