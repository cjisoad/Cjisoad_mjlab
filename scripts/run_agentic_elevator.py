"""Run the teacher/FSM in the three-floor elevator for agent tool calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/agentic_robot_frames'))
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--smoke', action='store_true', help='execute one bounded action and print a compact validation report')
    parser.add_argument('--no-images', action='store_true', help='disable RGBD for structured-state smoke only')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--quad-checkpoint-file', type=Path, default=ROOT/'mark/quad_teacher.pt')
    parser.add_argument('--biped-checkpoint-file', type=Path, default=ROOT/'mark/biped_teacher.pt')
    parser.add_argument('--rise-checkpoint-file', type=Path,
        default=ROOT/'logs/rsl_rl/transition_t/2026-10-09_01-13-48_transition_t_direct_4090_20261009/model_30000.pt')
    parser.add_argument('--rise-motion-file', type=Path,
        default=ROOT/'outputs/teacher_endpoint_rise_clearance20mm_20261008/tripod_to_biped_reference.npz')
    parser.add_argument('--down-checkpoint-file', type=Path, default=ROOT/'mark/down_teacher.pt')
    parser.add_argument('--motion-file', type=Path, default=ROOT/'src/assets/motions/go2/landing_v2/biped_to_tripod_v2.npz')
    parser.add_argument('--entry-bank', type=Path, default=ROOT/'src/assets/motions/go2/landing_v2/biped_entry_v2.npz')
    parser.add_argument('--initial-state', choices=('quadruped', 'biped'), default='quadruped')
    parser.add_argument('--rise-trigger-height', type=float, default=.35)
    parser.add_argument('--down-trigger-height', type=float, default=.25)
    parser.add_argument('--trigger-seconds', type=float, default=.10)
    args = parser.parse_args(argv)
    if args.num_envs != 1:
        parser.error('agentic elevator supports exactly one physical world')
    if args.host not in ('127.0.0.1', 'localhost', '::1'):
        parser.error('agent control service must bind to loopback')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from src.agentic_robot.runtime import AgenticRuntime
    from src.agentic_robot.observation.service import ObservationService
    from src.agentic_robot.server import serve
    runtime = None
    try:
        # Load policy runtime first to obtain the policy's actual command zero.
        runtime = AgenticRuntime(args)
        zero = runtime.term.zero.detach().cpu().numpy()
        observation = ObservationService(runtime.env.sim.mj_model, args.output_dir,
            fr_zero_anchor=zero[0] if zero.ndim == 2 else zero, width=args.width,
            height=args.height, capture_images=not args.no_images)
        runtime.observation = observation
        observation.reset(runtime.episode_id, runtime.adapter.data, fsm=runtime._fsm_state(command_accepted=None))
        if args.smoke:
            packet = runtime.observe()
            command = {'cmd_v': [0, 0, 0], 'cmd_p': [0, 0, 0], 't': .1}
            result = runtime.action(command)
            report = {'episode_id': packet['episode_id'], 'observation_time': packet['sim_time'],
                'robot_position_w': packet['robot']['position_w'], 'control_count': len(packet['controls']),
                'initial_frame_count': len(packet['frames']), 'initial_image_status': packet['image_status'],
                'action': {key: result[key] for key in ('cmd_v', 'cmd_p', 'requested_duration',
                    'executed_duration', 'steps', 'command_accepted', 'owner_before', 'owner_after', 'stop_reason')},
                'result_position_w': result['observation']['robot']['position_w'],
                'result_frame_count': len(result['observation']['frames']),
                'result_image_status': result['observation']['image_status']}
            args.output_dir.mkdir(parents=True, exist_ok=True)
            report_path = args.output_dir/'smoke_report.json'
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(json.dumps({'report': str(report_path), **report}, allow_nan=False), flush=True)
            runtime.close()
            return
        serve(runtime, host=args.host, port=args.port)
    except Exception:
        if runtime:
            runtime.close()
        raise


if __name__ == '__main__':
    main()
