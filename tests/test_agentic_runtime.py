"""Bounded, action-only runtime controls never trust model output blindly."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.agentic_robot.runtime import validate_command, duration_steps, failure_checks


def test_validate_command_requires_finite_velocity_position_and_bounded_time():
    result = validate_command({'cmd_v': [0.1, 0, 0], 'cmd_p': [0, .02, .08], 't': .21})
    assert result == ([.1, 0., 0.], [0., .02, .08], .21)
    for invalid in (
        {'cmd_v': [np.nan, 0, 0], 'cmd_p': [0, 0, 0], 't': .1},
        {'cmd_v': [0, 0], 'cmd_p': [0, 0, 0], 't': .1},
        {'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': 1.01},
        {'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': 0},
    ):
        with pytest.raises(ValueError):
            validate_command(invalid)


@pytest.mark.parametrize(('seconds', 'dt', 'steps'), [(.21, .02, 11), (1., .005, 200), (.1, .1, 1)])
def test_duration_rounds_to_nearest_control_boundary_without_exceeding_one_second(seconds, dt, steps):
    assert duration_steps(seconds, dt) == steps


def test_duration_rejects_policy_step_smaller_than_one_policy_interval():
    with pytest.raises(ValueError, match='at least one control step'):
        duration_steps(.001, .02)


@pytest.mark.parametrize(('owner', 'expected'), [
    (0, {'physical_failure'}),
    (2, {'physical_failure', 'anchor_pos', 'anchor_ori'}),
    (3, {'physical_failure'}),
    (4, {'physical_failure', 'anchor_pos', 'anchor_ori'}),
])
def test_tracking_failures_only_apply_while_transition_policy_owns_robot(owner, expected):
    from src.tasks.transition_t import terminations

    checks = failure_checks(owner, terminations)
    assert set(checks) == expected
