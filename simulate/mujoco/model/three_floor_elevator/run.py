"""Launch the standalone three-floor MuJoCo elevator environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simulate.mujoco.model.three_floor_elevator.controller import ThreeFloorElevatorController
from simulate.mujoco.model.three_floor_elevator.viewer import run


DEFAULT_XML = Path(__file__).with_name("three_floor_elevator.xml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help="MJCF path (defaults to the bundled three-floor scene)",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml.expanduser().resolve()))
    data = mujoco.MjData(model)
    controller = ThreeFloorElevatorController(model, data)
    controller.reset()
    run(model, data, controller)


if __name__ == "__main__":
    main()
