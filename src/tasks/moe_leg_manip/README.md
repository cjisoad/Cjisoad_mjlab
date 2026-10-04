# MoE leg manipulation

Independent snapshot of leg_manip at c2ddaed. The leg_manip task is restored to
the approved MLP baseline fd133c1. Both tasks initially use identical rewards,
curriculum, observations, exploration, and PPO hyperparameters. Their MDP and
RL code are separate so future changes can be made independently.

moe_leg_manip uses four actor experts, use_explicit_expert=False, dense routing
(top_k=-1), and the original MLP critic. Routing and per-expert gradient conflict
diagnostics are retained. The model tensor keys are unchanged, so checkpoints
from the former leg_manip_moe task load directly in moe_leg_manip. Training logs
for new runs use logs/rsl_rl/moe_leg_manip.

```bash
python scripts/train.py leg_manip --agent.resume False
python scripts/train.py moe_leg_manip --agent.resume False
python scripts/play.py moe_leg_manip --checkpoint-file /path/to/model.pt
python scripts/keyboard_leg_manip.py --checkpoint-file /path/to/mlp_model.pt
python scripts/keyboard_moe_leg_manip.py --checkpoint-file /path/to/moe_model.pt
```

Existing run artifacts retain their original folder names. This split does not
restart active trainers or viewers.
