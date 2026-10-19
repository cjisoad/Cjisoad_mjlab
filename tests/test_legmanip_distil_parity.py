"""Compare native actor routing and motion adapters with the frozen teacher source.

The keyboard action/motion classes are local to ``load_full_chain``. Compile
their unchanged method ASTs into test classes so the comparisons exercise
that source without loading four policies or constructing extra simulators.
"""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from src.tasks.legmanip_distil.commands import DOWN_MOTION, RISE_MOTION, ReferenceState
from src.tasks.legmanip_distil.teacher import FSMTeacher, motion_observations
from src.tasks.transition_down_v2.commands import LandingCommand
from src.tasks.transition_down_v2.env_cfg import landing_env_cfg
from src.tasks.transition_t.commands import (
    JOINT_NAMES, TRACKED_BODY_NAMES, TransitionCommand, TransitionCommandCfg,
)
from src.tasks.transition_t.env_cfg import transition_env_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz


_source_path = Path(__file__).resolve().parents[1] / 'scripts/keyboard_transition_fsm.py'
_source_spec = importlib.util.spec_from_file_location('_legmanip_parity_keyboard_fsm', _source_path)
original = importlib.util.module_from_spec(_source_spec)
sys.modules[_source_spec.name] = original
_source_spec.loader.exec_module(original)


def _original_class(class_name, base, methods):
    path = Path(original.__file__)
    tree = ast.parse(path.read_text())
    source = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in selected} == set(methods)
    definition = ast.ClassDef(name=class_name,
        bases=[ast.Name(id='_OriginalBase', ctx=ast.Load())], keywords=[],
        body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    namespace = {**vars(original), '_OriginalBase': base}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[class_name]


def test_fsm_routes_native_requests_without_clipping_or_substep_processing(monkeypatch):
    from types import MethodType
    from src.tasks.legmanip_distil import teacher as teacher_module

    owners = torch.tensor([0, 2, 3, 4, 0, 2, 3, 4])
    native = torch.arange(96, dtype=torch.float32).reshape(8, 12) - 40.
    # Values above the former transition clip must remain supervision targets.
    env = SimpleNamespace(num_envs=8)
    endpoint = SimpleNamespace(actions=lambda env, ids: native[ids])
    teacher = SimpleNamespace(env=env, action=SimpleNamespace(raw_action=torch.zeros_like(native)),
        term=SimpleNamespace(owner=owners, references={2: object(), 4: object()}),
        quad=endpoint, biped=endpoint,
        actors={'rise': lambda obs: obs['actor'], 'down': lambda obs: obs['actor']})
    teacher.requests = MethodType(FSMTeacher.requests, teacher)
    monkeypatch.setattr(teacher_module, 'motion_observations', lambda env, reference: native)
    actual = FSMTeacher.forward(teacher)
    torch.testing.assert_close(actual, native, atol=0., rtol=0.)


class _Scene(dict):
    pass


def _reference_environment():
    """Only tensors and entity lookups used by the real reference implementation."""
    generator = torch.Generator().manual_seed(31)
    count, bodies = 3, len(TRACKED_BODY_NAMES)
    random = lambda *shape: torch.randn(*shape, generator=generator)
    scene = _Scene()
    scene.env_origins = torch.tensor([[0., 0., 0.], [5., 2., 0.], [-3., 8., 0.]])
    # Include reared poses: ordinary yaw extraction and teacher heading differ.
    quaternion = quat_from_euler_xyz(torch.tensor([.1, -.2, .15]),
        torch.tensor([-1.45, -.4, -1.25]), torch.tensor([.8, -.5, 1.3]))
    body_position = .1 * random(count, bodies, 3) + scene.env_origins[:, None]
    body_position[:, 0, 2] = torch.tensor([.52, .3, .48])
    joint_position = random(count, 12)
    data = SimpleNamespace(
        body_link_pos_w=body_position,
        body_link_quat_w=quaternion[:, None].expand(-1, bodies, -1).clone(),
        body_link_lin_vel_w=random(count, bodies, 3),
        body_link_ang_vel_w=random(count, bodies, 3),
        root_link_lin_vel_b=random(count, 3), root_link_ang_vel_b=random(count, 3),
        joint_pos=joint_position, joint_pos_biased=joint_position + .07,
        joint_vel=random(count, 12), default_joint_pos=.2 * random(count, 12),
        default_joint_vel=.1 * random(count, 12))
    scene['robot'] = SimpleNamespace(data=data, joint_names=JOINT_NAMES,
        body_names=TRACKED_BODY_NAMES,
        find_bodies=lambda names, preserve_order: (
            [TRACKED_BODY_NAMES.index(name) for name in names], list(names)))
    return SimpleNamespace(scene=scene, num_envs=count, device='cpu', step_dt=.02,
        common_step_counter=0, action_manager=SimpleNamespace(action=random(count, 12)))


@pytest.mark.parametrize('motion_path', [RISE_MOTION, DOWN_MOTION], ids=['rise', 'down'])
def test_reference_alignment_clock_and_completion_match_original(motion_path):
    env = _reference_environment()
    source_type = _original_class('FullChainMotion', LandingCommand, ('_update_command',))
    source = object.__new__(source_type)
    # Bypass entry-bank initialization only. All compared reference methods,
    # including LandingCommand's heading alignment, are the original methods.
    cfg = TransitionCommandCfg(motion_file=motion_path, sampling_mode='start', debug_vis=False)
    TransitionCommand.__init__(source, cfg, env)
    source.active = torch.zeros(env.num_envs, dtype=torch.bool)
    actual = ReferenceState(motion_path, env)
    ids = torch.tensor([0, 2])
    source.start_from_current_state(ids)
    actual.start_from_current_state(ids)
    source.active[ids] = actual.active[ids] = True
    source.time_steps[0] = actual.time_steps[0] = source.motion.time_step_total - 2
    source.time_steps[1] = actual.time_steps[1] = 7
    fields = ('time_steps', 'completed', 'reference_rotation', 'reference_translation',
              'command', 'anchor_pos_w', 'anchor_quat_w', 'body_pos_relative_w',
              'body_quat_relative_w', 'body_lin_vel_w', 'body_ang_vel_w')
    for step in range(4):
        env.common_step_counter = step
        source._update_command()
        actual.advance()
        for field in fields:
            torch.testing.assert_close(getattr(actual, field), getattr(source, field), atol=0., rtol=0.)
        assert actual.time_steps[1].item() == 7  # Inactive world never advances.
        assert actual.completed[0].item() == (step >= 2)  # Act on the final frame once.


@pytest.mark.parametrize('cfg_factory,motion_path',
    [(transition_env_cfg, RISE_MOTION), (landing_env_cfg, DOWN_MOTION)], ids=['rise', 'down'])
def test_actor75_matches_original_configured_observation_terms(cfg_factory, motion_path):
    env = _reference_environment()
    reference = ReferenceState(motion_path, env)
    reference.start_from_current_state(torch.arange(env.num_envs))
    reference.time_steps[:] = torch.tensor([3, 17, 29])
    reference.completed[1] = True
    env.command_manager = SimpleNamespace(get_term=lambda name: reference,
        get_command=lambda name: reference.command)
    original_cfg = cfg_factory(play=True)
    group = original_cfg.observations['actor']
    assert not group.enable_corruption
    expected = torch.cat([term.func(env, **term.params) for term in group.terms.values()], dim=-1)
    actual = motion_observations(env, reference)
    assert actual.shape == (env.num_envs, 75)
    torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
