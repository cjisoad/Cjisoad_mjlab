# Loco Pedipulation Keyboard Playback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated native MuJoCo keyboard playback command for the trained Loco Pedipulation checkpoint while reusing the existing Pedipulation keyboard controller and X11 focus handling.

**Architecture:** Keep task and training code unchanged. Add a small bridge around `LocoPedipulationCommand.set_command()` and a viewer script that mirrors `scripts/keyboard_play.py`; reuse the existing `KeyboardController` and `X11Keyboard` modules. The bridge sends velocity every poll and maps zero/nonzero FR offsets to `fr_enabled` plus the task's FK target projection.

**Tech Stack:** Python, PyTorch, mjlab `ManagerBasedRlEnv`, `NativeMujocoViewer`, system X11 polling, unittest.

---

### Task 1: Add a failing bridge test

**Files:**
- Create: `tests/test_keyboard_loco_pedipulation.py`
- Reference: `src/tasks/loco_pedipulation/mdp/commands.py:186-205`

- [ ] **Step 1: Write the failing test**

```python
"""Keyboard bridge tests for Loco Pedipulation playback."""
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class KeyboardLocoBridgeTests(unittest.TestCase):
  def test_bridge_maps_velocity_and_target_and_clears_target(self):
    from keyboard_loco_pedipulation import apply_keyboard_command

    calls = []

    class Term:
      def set_command(self, env_ids, *, velocity=None, fr_enabled=None, target_offset=None):
        calls.append((tuple(env_ids), velocity, fr_enabled, target_offset))

    controller = SimpleNamespace(command=(.2, 0., -.4, .03, -.02, .07))
    apply_keyboard_command(Term(), controller, torch.device("cpu"), 1)
    self.assertEqual(calls[-1], ((0,), (.2, 0., -.4), True, (.03, -.02, .07)))

    controller.command = (0., 0., 0., 0., 0., 0.)
    apply_keyboard_command(Term(), controller, torch.device("cpu"), 1)
    self.assertEqual(calls[-1], ((0,), (0., 0., 0.), False, (0., 0., 0.)))


if __name__ == "__main__":
  unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests/test_keyboard_loco_pedipulation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'keyboard_loco_pedipulation'`.

### Task 2: Implement the command bridge and playback script

**Files:**
- Create: `scripts/keyboard_loco_pedipulation.py`
- Reference: `scripts/keyboard_play.py`, `scripts/keyboard_control.py`, `scripts/keyboard_x11.py`, `src/tasks/loco_pedipulation/env_cfg.py`, `src/tasks/loco_pedipulation/mdp/commands.py`

- [ ] **Step 1: Implement the bridge and argument parser**

Add `apply_keyboard_command(term, controller, device, num_envs)` that converts the six-value controller tuple to velocity and target tensors/tuples and calls `term.set_command(range(num_envs), velocity=..., fr_enabled=..., target_offset=...)`. Treat a zero target as disabled; use the task's existing target projection.

Add a `main()` parser with defaults:

```python
DEFAULT_CHECKPOINT = ROOT / "logs/rsl_rl/loco_pedipulation/2026-09-24_03-13-35/model_14900.pt"
```

Support `--checkpoint-file`, `--device`, `--seed`, `--status-file`, `--smoke`, and `--steps`; require `--steps >= 501` for smoke.

- [ ] **Step 2: Implement native viewer input handling**

Reuse `KeyboardController` and `X11Keyboard`. Build `loco_pedipulation_env_cfg(play=True)`, replace the `twist` command with manual updates through the bridge, load `LocoPedipulationOnPolicyRunner` from `loco_pedipulation.rl`, and run the policy in a `NativeMujocoViewer` subclass. Reserve W/S/A/D/Q/E/R and arrow keys, preserve Backspace and Space, reset the controller with the environment, and poll focused X11 input at 100 Hz.

- [ ] **Step 3: Add overlay and status output**

Show state, step count, focus, `vx`, `vy`, `wz`, target offsets, FR enabled state, and the command term phase. If `--status-file` is given, atomically write the same values plus `observed_command` and `last_error` every 0.2 seconds.

- [ ] **Step 4: Implement bounded smoke**

Run 550 default steps headlessly. Apply a nonzero keyboard command during steps 150-170, call the bridge each step, assert command and observations remain finite, assert the term remains manually commanded across its resampling boundary, then reset controller and environment and assert velocity, target, and `manual` are cleared/released.

- [ ] **Step 5: Run the focused test to verify it passes**

Run: `python -m unittest tests/test_keyboard_loco_pedipulation.py -v`
Expected: PASS.

### Task 3: Document and validate the dedicated command

**Files:**
- Create: `doc/KEYBOARD_LOCO_PEDIPULATION.md`
- Modify: `tests/test_keyboard_loco_pedipulation.py` if integration assertions need to be isolated

- [ ] **Step 1: Document launch and keys**

Document the default checkpoint and command:

```bash
python scripts/keyboard_loco_pedipulation.py
```

Document the existing key mapping, target projection to the FK bank, the zero-target locomotion sentinel, and optional explicit checkpoint/device/status/smoke arguments.

- [ ] **Step 2: Run all keyboard and task unit tests**

Run: `python -m unittest discover -s tests -p 'test_keyboard*.py'`
Expected: PASS with no failures.

Run: `python -m unittest tests/test_loco_pedipulation.py -v`
Expected: PASS.

- [ ] **Step 3: Run the checkpoint smoke test when the runtime is available**

Run: `python scripts/keyboard_loco_pedipulation.py --smoke --device cuda:0`
Expected: JSON output containing `"smoke": "passed"`, a finite command shape, and reset confirmation. If CUDA is unavailable, rerun with `--device cpu` and report the limitation.

- [ ] **Step 4: Commit the implementation**

```bash
git add scripts/keyboard_loco_pedipulation.py tests/test_keyboard_loco_pedipulation.py doc/KEYBOARD_LOCO_PEDIPULATION.md
git commit -m "feat: add loco pedipulation keyboard playback"
```
