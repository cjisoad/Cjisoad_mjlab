"""Three tracking courses with promotion based on complete starts."""
from collections import deque


class TrackingCourse:
  def __init__(self, stage=0, enabled=True, minimum_episodes=128,
               window_size=512, minimum_steps=1500, success_threshold=.85):
    if stage not in range(3) or minimum_episodes<1 or window_size<minimum_episodes or minimum_steps<0 or not 0<success_threshold<=1:
      raise ValueError('Invalid three-stage tracking course')
    self.stage=stage; self.enabled=enabled; self.level=1
    self.minimum_episodes=minimum_episodes; self.window_size=window_size
    self.minimum_steps=minimum_steps; self.success_threshold=success_threshold
    self.started=0; self.window=deque(maxlen=window_size)

  @property
  def live_probability(self):
    return self.level/4. if self.stage==2 else 0.

  @property
  def randomization_fraction(self):
    return self.live_probability

  def record(self, success, full_start, live_start, *, step):
    for result,full,live in zip(success,full_start,live_start,strict=True):
      if (live if self.stage==2 else full and not live): self.window.append(bool(result))
    if (self.enabled and len(self.window)>=self.minimum_episodes and
        step-self.started>=self.minimum_steps and sum(self.window)/len(self.window)>=self.success_threshold):
      if self.stage<2: self.stage+=1
      elif self.level<4: self.level+=1
      else: return
      self.started=int(step); self.window.clear()

  def state_dict(self):
    return dict(stage=self.stage,level=self.level,started=self.started,
      window=list(self.window),enabled=self.enabled)

  def load_state_dict(self, state):
    if state['stage'] not in range(3) or state['level'] not in range(1,5):
      raise ValueError('Invalid saved tracking course')
    self.stage=state['stage']; self.level=state['level']; self.started=state['started']
    self.enabled=state['enabled']; self.window=deque(state['window'],maxlen=self.window_size)
