"""Launch the standalone three-floor MuJoCo elevator environment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_XML = Path(__file__).with_name("three_floor_elevator.xml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="MJCF path (defaults to the bundled three-floor scene)",
    )
    parser.add_argument("--web", action="store_true", help="serve an interactive browser viewer")
    parser.add_argument("--host", default="127.0.0.1", help="web bind address (default: local only)")
    parser.add_argument("--port", type=int, default=8765, help="web port (default: 8765)")
    args = parser.parse_args()

    if args.web and sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from simulate.mujoco.model.three_floor_elevator.controller import ThreeFloorElevatorController

    model = mujoco.MjModel.from_xml_path(str(args.xml.expanduser().resolve()))
    data = mujoco.MjData(model)
    controller = ThreeFloorElevatorController(model, data)
    controller.reset()
    if args.web:
        from simulate.mujoco.model.three_floor_elevator.web_viewer import run
        run(model, data, controller, host=args.host, port=args.port)
    else:
        from simulate.mujoco.model.three_floor_elevator.viewer import run
        run(model, data, controller)


if __name__ == "__main__":
    main()
