"""Frozen-teacher command coverage and lightweight command-line contract."""
from pathlib import Path


def test_probe_grid_includes_every_stop_reverse_arc_and_inplace_turn():
    from scripts.probe_supermimic_teachers import command_grid
    cases=command_grid()
    assert len(cases)==24
    assert {row['mode'] for row in cases}=={'quad','biped'}
    for mode in ('quad','biped'):
        velocities={tuple(row['velocity']) for row in cases if row['mode']==mode}
        assert velocities=={(vx,0.,wz) for vx in (-.2,0.,.15,.6) for wz in (0.,-1.,1.)}
        assert (0.,0.,0.) in velocities
        assert (0.,0.,-1.) in velocities and (0.,0.,1.) in velocities
        assert (-.2,0.,0.) in velocities


def test_probe_cli_supports_short_smoke_device_and_output(tmp_path):
    from scripts.probe_supermimic_teachers import build_parser
    args=build_parser().parse_args(['--steps','50','--device','cpu','--output',str(tmp_path/'probe.json')])
    assert args.steps==50 and args.device=='cpu'
    assert args.output==Path(tmp_path/'probe.json')
    assert args.num_envs>=1
