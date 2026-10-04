# Vendored MoE networks

`ac_moe_common.py` and `ac_moe_gated.py` are unmodified copies from the user's
local `/home/e980/projects/moe-rsl-rl/moe_rsl_rl/modules` at commit
`ebd4bcffc5a98b152611dbef401ae1a5ccd6d21f`. The upstream BSD-3-Clause
license is included in `LICENSE`. Only the neural networks are used: installing
the full package would require RSL-RL 5.4.2, while this training stack uses 5.0.1.

`moe_leg_manip` uses four independent `[512,256,128]` actor experts, an `[128]`
gate, `use_explicit_expert=False`, and `top_k=-1` (dense). All experts contribute
to a weighted action mean; their roles are learned rather than assigned to
stances. The critic, 51D actor observations, per-joint StanceGaussianDistribution,
symmetry loss, PPO settings, exploration, curriculum and rewards are unchanged.
No auxiliary gate/balance loss or Gaussian mixture is enabled.

Train from scratch with `python scripts/train.py moe_leg_manip --agent.resume False`.
The keyboard script detects expert parameter keys automatically, and the
original `leg_manip` task remains an MLP for legacy checkpoints.

Every diagnostic interval logs group-average expert weights and normalized
router entropy under `GradConflict/moe_routing`. Per-router and per-expert
gradient cosines are alongside the existing overall actor cosines under
`GradConflict/<pair>/<objective>/<router|expert_i>`. A missing/zero-gradient
group has `valid=0`; it does not imply zero conflict. Dense routing can still
collapse towards one expert, so routing balance and tracking outcomes must
be assessed together.
