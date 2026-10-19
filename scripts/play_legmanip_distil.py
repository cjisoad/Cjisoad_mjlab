"""Play an exported legmanip_distil student without any teacher or motion file."""
import argparse
import json
import os
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-file', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--scenario', choices=('quad', 'biped', 'cycle', 'mixed'))
    parser.add_argument('--command', type=float, nargs=6, default=[0.] * 6,
                        metavar=('VX', 'VY', 'WZ', 'FR_DX', 'FR_DY', 'FR_DZ'))
    args = parser.parse_args(argv)
    if not args.headless and args.num_envs != 1:
        parser.error('Interactive keyboard playback uses one environment')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import torch
    from src.tasks.legmanip_distil.deployment import (
        create_student_world, run_headless, update_keyboard_intent)
    torch.set_num_threads(1)
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    env, wrapped, student, metadata = create_student_world(
        args.checkpoint_file, device=device, num_envs=args.num_envs, scenario=args.scenario)
    keyboard = None
    try:
        intent = env.command_manager.get_term('intent')
        if args.scenario is None:
            intent.set_command(torch.arange(env.num_envs, device=device),
                               [args.command] * env.num_envs)
        if args.headless:
            print(json.dumps(run_headless(wrapped, student, args.steps), indent=2))
            return
        from mjlab.viewer import NativeMujocoViewer
        from keyboard_teacher import TeacherKeyboardController, workspace_for_mode
        from keyboard_x11 import X11Keyboard
        keyboard = X11Keyboard()
        controller = TeacherKeyboardController(workspace_for_mode('union'))
        controller.velocity = list(args.command[:3])
        controller.target = controller.requested_target = tuple(args.command[3:])
        controller.target_active = any(args.command[3:])

        class StudentViewer(NativeMujocoViewer):
            def __init__(self):
                super().__init__(wrapped, student)
                self.last_input = time.monotonic()

            def _safe_key_callback(self, key):
                if key == 259:
                    self.request_reset()
                elif key not in {ord(k) for k in 'WASDQER'} | {262, 263, 264, 265}:
                    super()._safe_key_callback(key)

            def reset_environment(self):
                controller.reset()
                super().reset_environment()

            def tick(self):
                now = time.monotonic()
                if args.scenario is None and now - self.last_input >= .01:
                    update_keyboard_intent(intent, controller, keyboard.read(),
                        now - self.last_input, focused=keyboard.focused, paused=self._is_paused)
                    self.last_input = now
                return super().tick()

        print('Student only: W/S velocity, A/D yaw, arrows FR dx/dy, Q/E FR height, '
              'R six zeros, Backspace reset. Completed updates:', metadata['completed_updates'])
        StudentViewer().run()
    finally:
        if keyboard is not None:
            keyboard.close()
        env.close()


if __name__ == '__main__':
    main()
