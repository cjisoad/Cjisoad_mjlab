# Loco Pedipulation Pedipulation-Compatible Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Loco Pedipulation to Pedipulation's 48-value actor observation and `[vx, vy, wz, dx, dy, dz]` command ABI while retaining its tripod transition machine and rewards.

**Architecture:** The command term owns a six-value public command. Its last three values derive the internal FR-enabled state and are mapped to the existing FK target bank for phase execution. Actor observations are implemented in the task-local observations module and copied exactly into the critic observation before critic-only state is appended.

**Tech Stack:** Python, PyTorch, mjlab manager environment, RSL-RL PPO, unittest.

---

### Task 1: Establish the six-value command contract

**Files:**
- Modify: `tests/test_loco_pedipulation.py`
- Modify: `src/tasks/loco_pedipulation/mdp/commands.py`

- [ ] **Step 1: Write failing command tests**

Add tests that construct the existing fake environment and assert:

```python
self.assertEqual(term.command.shape, (4, 6))
term.set_command([0], velocity=(.2, -.1, .3), target_offset=(.03, 0., .07))
torch.testing.assert_close(term.command[0], torch.tensor((.2, -.1, .3, .03, 0., .07)))
self.assertTrue(term.enabled[0])
term.set_command([0], target_offset=(0., 0., 0.))
self.assertFalse(term.enabled[0])
torch.testing.assert_close(term.command[0, 3:], torch.zeros(3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_loco_pedipulation.py -v`
Expected: FAIL because the command still has shape `(4, 3)`.

- [ ] **Step 3: Implement public command synchronization**

Make `_command` shape `[num_envs, 6]`; use `_command[:, :3]` for filtered velocity. In `_resample_command`, write both the sampled velocity and requested FK offset to `_command`. In `set_command`, update the supplied velocity and target independently; target `(0, 0, 0)` sets `enabled=False`, otherwise map to nearest FK offset and store the mapped offset in `_command[:, 3:]`. Maintain `enabled` from `(_command[:, 3:] != 0).any(-1)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_loco_pedipulation.py -v`
Expected: PASS.

### Task 2: Replace actor observation with the 48-value ABI

**Files:**
- Modify: `tests/test_loco_pedipulation.py`
- Modify: `src/tasks/loco_pedipulation/mdp/observations.py`
- Modify: `src/tasks/loco_pedipulation/env_cfg.py`

- [ ] **Step 1: Write failing observation test**

Add a test that sets command target and action/robot tensors in the fake environment, calls `policy_state(env, add_noise=False)`, and asserts shape `(4, 48)`, command positions `6:12`, and exact concatenation order:

```python
expected = torch.cat((
    robot.root_link_ang_vel_b * .25,
    robot.projected_gravity_b,
    term.command * torch.tensor((2., 2., .25, 1., 1., 1.)),
    action.joint_pos - action.default_angles,
    action.joint_vel * .05,
    action.raw_action,
), dim=-1)
torch.testing.assert_close(policy_state(env, add_noise=False), expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_loco_pedipulation.py -v`
Expected: FAIL because current actor observation includes a two-value gait phase and 18-value internal foot-control state.

- [ ] **Step 3: Implement policy/shared state**

Implement `policy_state(env, add_noise=True)` and `shared_policy_state(env)` matching Pedipulation's scaling, clipping, one shared noisy actor sample, and noise amplitudes. Remove `phase` and `foot_control` from the actor config; actor group contains a single `policy` observation. Add the shared policy state to the critic group, preserving critic-only base velocity and foot information.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests/test_loco_pedipulation.py -v`
Expected: PASS with a 48-dimensional policy observation.

### Task 3: Adapt reward and command consumers to six-value commands

**Files:**
- Modify: `src/tasks/loco_pedipulation/mdp/rewards.py`
- Modify: `src/tasks/loco_pedipulation/mdp/commands.py`
- Test: `tests/test_loco_pedipulation.py`

- [ ] **Step 1: Update velocity consumers**

Use `term.command[:, :3]` wherever velocity magnitude, desired velocity, gait activation, and tracking are computed. Preserve all existing reward weights and phase logic. Use `term.command[:, 3:]` as the only source for target activation; do not add target phase/reference values to actor observations.

- [ ] **Step 2: Add regression assertions**

Assert that a six-value command with a target still reaches `HOLD`, that a target removal enters `LOWER`, and that velocity tracking remains finite and uses only its first three values.

- [ ] **Step 3: Run task tests**

Run: `python -m unittest tests/test_loco_pedipulation.py -v`
Expected: PASS.

### Task 4: Update playback, documentation, and fresh-training checks

**Files:**
- Modify: `scripts/keyboard_loco_pedipulation.py`
- Modify: `doc/KEYBOARD_LOCO_PEDIPULATION.md`
- Modify: `doc/LOCO_PEDIPULATION.md`
- Modify: `tests/test_keyboard_loco_pedipulation.py`

- [ ] **Step 1: Adapt keyboard bridge**

The bridge calls `set_command(..., velocity=..., target_offset=...)` only. It no longer passes `fr_enabled`; zero keyboard offsets send a zero target, nonzero offsets activate it through the command term.

- [ ] **Step 2: Update model compatibility behavior**

Replace the old default checkpoint with no default checkpoint. Require `--checkpoint-file`; document that `model_14900.pt` has a 65-dimensional actor input and is intentionally incompatible. Make smoke accept an explicit newly trained checkpoint.

- [ ] **Step 3: Update docs and test bridge**

Document the six-value interface, zero-target activation rule, 48-value actor ABI, and retraining command. Update the bridge test to assert no separate enable argument.

- [ ] **Step 4: Verify all tests and a tiny training startup**

Run:

```bash
python -m unittest discover -s tests
python scripts/train.py loco_pedipulation --env.scene.num-envs 2 --agent.max-iterations 1 --agent.save-interval 1 --device cuda:0
```

Expected: tests pass; training constructs a 48-input actor and completes one iteration. Do not attempt to load `model_14900.pt`.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/loco_pedipulation scripts/keyboard_loco_pedipulation.py doc/LOCO_PEDIPULATION.md doc/KEYBOARD_LOCO_PEDIPULATION.md tests
git commit -m "feat: align loco pedipulation policy interface"
```
