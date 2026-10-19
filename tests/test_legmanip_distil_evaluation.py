"""Physical success criteria follow the script and tolerate contact flicker."""
import pytest

from src.tasks.legmanip_distil import evaluation


def test_cycle_does_not_require_a_tripod_operation_stage_it_never_runs():
    record = evaluation.new_record('cycle', None)
    record.update(completed=True, biped_hold=.5, final_quad_hold=.5)
    assert evaluation.episode_result(record, .02)['success']


def test_quad_still_requires_tripod_operation_and_final_return():
    record = evaluation.new_record('quad', None)
    record.update(completed=True, final_quad_hold=.5)
    assert 'tripod_hold_too_short' in evaluation.episode_result(record, .02)['failure_reasons']
    record['tripod_hold'] = .5
    assert evaluation.episode_result(record, .02)['success']
    record['final_quad_hold'] = 0.
    assert not evaluation.episode_result(record, .02)['success']


def sample(record, contacts=(False, False, True, True), *, gravity=(-1., 0., 0.),
           height=.49, stage='biped_operation', foot_error=.01):
    evaluation.update_physical_holds(record, stage=stage, gravity=gravity, height=height,
                                     contacts=contacts, foot_error=foot_error, dt=.02)


def test_short_alternating_hind_contact_gaps_allow_a_continuous_biped_hold():
    record = evaluation.new_record('biped', None)
    for i in range(40):
        sample(record, (False, False, i % 2 == 0, i % 2 == 1))
    assert record['biped_hold'] >= .5


@pytest.mark.parametrize('contacts', [(False,) * 4, (False, False, True, False)])
def test_absent_or_one_sided_hind_support_cannot_pass(contacts):
    record = evaluation.new_record('biped', None)
    for _ in range(40):
        sample(record, contacts)
    assert record['biped_hold'] == 0.


def test_long_contact_gap_resets_hold_and_fresh_contact_recovers():
    record = evaluation.new_record('biped', None)
    for _ in range(15):
        sample(record)
    for _ in range(8):
        sample(record, (False,) * 4)
    assert record['biped_run'] == 0.
    assert record['biped_hold'] < .5
    sample(record)
    assert record['biped_run'] == pytest.approx(.02)


@pytest.mark.parametrize('gravity,height', [((0., 0., -1.), .49), ((-1., .4, 0.), .49),
                                           ((-1., 0., 0.), .3)])
def test_contact_grace_never_overrides_bad_posture(gravity, height):
    record = evaluation.new_record('biped', None)
    for _ in range(40):
        sample(record, gravity=gravity, height=height)
    assert record['biped_hold'] == 0.


def test_support_memory_is_not_carried_across_non_biped_stages():
    record = evaluation.new_record('cycle', None)
    sample(record)
    sample(record, stage='rise')
    sample(record, (False,) * 4)
    assert record['biped_run'] == 0.
