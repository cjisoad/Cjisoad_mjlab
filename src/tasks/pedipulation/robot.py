"""Task-local Go2 physics, with the target repository's visual mesh assets.

The snapshot preserves the source URDF's separate fixed links and inertials.
PhysX's cylinder-to-capsule option is represented by MuJoCo capsules. Missing
inertials use the source asset density (0.001 kg/m^3); the source importer may
handle these tiny collision-only links differently. Contact response and
adjacent-body filtering remain engine-specific approximations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .constants import DEFAULT_ANGLES, JOINT_NAMES, LEGS


SNAPSHOT_PATH = Path(__file__).with_name("source_physics.json")
BASE_CONTACT = "base_link"
FOOT_BODIES = tuple(f"{leg}_foot" for leg in LEGS)
NONFOOT_BODIES = (
  BASE_CONTACT, "Head_upper", "Head_lower",
  *(f"{leg}_{part}" for leg in LEGS
    for part in ("hip", "thigh", "calf", "calflower", "calflower1")),
  "imu", "radar",
)


def load_source_physics() -> dict:
  return json.loads(SNAPSHOT_PATH.read_text())


def _numbers(values) -> str:
  return " ".join(format(float(value), ".17g") for value in values)


def _quat(rpy) -> tuple[float, float, float, float]:
  roll, pitch, yaw = (angle / 2 for angle in rpy)
  cr, sr = math.cos(roll), math.sin(roll)
  cp, sp = math.cos(pitch), math.sin(pitch)
  cy, sy = math.cos(yaw), math.sin(yaw)
  return (cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
          cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy)


def _body_name(source_name: str) -> str:
  return BASE_CONTACT if source_name == "base" else source_name


def get_stand_spec():
  """Build an independent MjSpec without loading anything from legged_gym."""
  import mujoco
  from src.assets.robots.unitree_go2.go2_constants import GO2_XML, get_assets

  physics = load_source_physics()
  target = ET.parse(GO2_XML).getroot()
  root = ET.Element("mujoco", model="go2_source_stand")
  ET.SubElement(root, "compiler", angle="radian", meshdir="assets",
                autolimits="true", fusestatic="false")
  root.append(copy.deepcopy(target.find("default")))
  root.append(copy.deepcopy(target.find("asset")))
  world = ET.SubElement(root, "worldbody")
  bodies = {}
  parents = {joint["child"]: joint for joint in physics["joints"]}
  visual_bodies = {body.attrib["name"]: body for body in target.findall(".//body")}

  for link in physics["links"]:
    source_name = link["name"]
    name = _body_name(source_name)
    joint = parents.get(source_name)
    parent = world if joint is None else bodies[joint["parent"]]
    body = ET.SubElement(parent, "body", name=name)
    bodies[source_name] = body
    if joint is None:
      body.set("pos", "0 0 0.42")
      ET.SubElement(body, "freejoint", name="floating_base_joint")
    else:
      body.set("pos", _numbers(joint["origin"]["xyz"]))
      body.set("quat", _numbers(_quat(joint["origin"]["rpy"])))
      if joint["type"] != "fixed":
        ET.SubElement(body, "joint", name=joint["name"], type="hinge",
                      axis=_numbers(joint["axis"]),
                      range=_numbers([joint["limit"]["lower"], joint["limit"]["upper"]]),
                      damping="0", armature="0", frictionloss="0")

    inertial = link["inertial"]
    if inertial is not None and inertial["mass"] > 0:
      # URDF inertia frames are unrotated in the captured Go2 asset.
      if any(inertial["origin"]["rpy"]):
        raise ValueError("Snapshot inertia rotation requires tensor transformation")
      ET.SubElement(body, "inertial", mass=str(inertial["mass"]),
                    pos=_numbers(inertial["origin"]["xyz"]),
                    fullinertia=_numbers(inertial["inertia"]))

    visual_source = visual_bodies.get(name)
    if source_name.endswith("_foot"):
      visual_source = visual_bodies[source_name.replace("_foot", "_calf")]
    if visual_source is not None:
      visuals = visual_source.findall("geom[@class='visual']")
      for index, original in enumerate(visuals):
        is_foot = original.attrib.get("mesh") == "foot"
        if is_foot != source_name.endswith("_foot"):
          continue
        geom = copy.deepcopy(original)
        geom.set("name", f"{name}_visual_{index}")
        if is_foot:
          geom.set("pos", "0 0 0")
        body.append(geom)

    for index, collision in enumerate(link["collisions"]):
      kind = collision["type"]
      shape = collision["shape"]
      suffix = "" if len(link["collisions"]) == 1 else f"_{index}"
      attributes = dict(
        name=f"{name}{suffix}_collision", type=kind,
        pos=_numbers(collision["origin"]["xyz"]),
        quat=_numbers(_quat(collision["origin"]["rpy"])),
        group="3", contype="1", conaffinity="1", condim="3", priority="1",
        friction="1 0 0", density=str(physics["asset_options"]["density"]),
      )
      if kind == "box":
        attributes["size"] = _numbers([size / 2 for size in shape["size"]])
      elif kind == "sphere":
        attributes["size"] = str(shape["radius"])
      elif kind == "cylinder":
        attributes["type"] = "capsule"
        attributes["size"] = _numbers([shape["radius"], shape["length"] / 2])
      else:
        raise ValueError(f"Unsupported source collision shape: {kind}")
      ET.SubElement(body, "geom", **attributes)

    if source_name.endswith("_foot"):
      ET.SubElement(body, "site", name=source_name[:2], type="sphere",
                    size="0.022", pos="0 0 0", rgba="1 0 0 1", group="5")
    if source_name == "imu":
      ET.SubElement(body, "site", name="imu", pos="0 0 0", group="5")

  contact = ET.SubElement(root, "contact")
  for joint in physics["joints"]:
    ET.SubElement(contact, "exclude", body1=_body_name(joint["parent"]),
                  body2=_body_name(joint["child"]))
  root.append(copy.deepcopy(target.find("sensor")))
  spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_stand_robot_cfg():
  """Create fresh source-specific PD and physical configuration for this task."""
  from mjlab.actuator import IdealPdActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

  actuators = tuple(
    IdealPdActuatorCfg(
      target_names_expr=(name,), stiffness=40., damping=1.,
      effort_limit=35.55 if "calf" in name else 23.7,
      armature=0., frictionloss=0.,
    )
    for name in JOINT_NAMES
  )
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0., 0., .42), joint_pos=dict(zip(JOINT_NAMES, DEFAULT_ANGLES)),
      joint_vel={".*": 0.},
    ),
    spec_fn=get_stand_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=actuators, soft_joint_pos_limit_factor=.9,
    ),
  )


def snapshot_from_urdf(path: Path) -> dict:
  """Extract physics using only structured XML; used for explicit regeneration."""
  raw = path.read_bytes()
  urdf = ET.fromstring(raw)

  def vector(text):
    return [float(value) for value in text.split()]

  def origin(element):
    node = element.find("origin")
    return {name: vector(node.attrib.get(name, "0 0 0")) if node is not None
            else [0., 0., 0.] for name in ("xyz", "rpy")}

  links = []
  for link in urdf.findall("link"):
    inertial = link.find("inertial")
    inertial_data = None
    if inertial is not None:
      tensor = inertial.find("inertia").attrib
      inertial_data = dict(
        mass=float(inertial.find("mass").attrib["value"]), origin=origin(inertial),
        inertia=[float(tensor[key]) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")],
      )
    collisions = []
    for collision in link.findall("collision"):
      geometry = list(collision.find("geometry"))[0]
      shape = {key: vector(value) if key == "size" else float(value)
               for key, value in geometry.attrib.items()}
      collisions.append(dict(type=geometry.tag, shape=shape, origin=origin(collision)))
    links.append(dict(name=link.attrib["name"], inertial=inertial_data, collisions=collisions))

  joints = []
  for joint in urdf.findall("joint"):
    result = dict(name=joint.attrib["name"], type=joint.attrib["type"],
                  parent=joint.find("parent").attrib["link"],
                  child=joint.find("child").attrib["link"], origin=origin(joint))
    axis = joint.find("axis")
    result["axis"] = vector(axis.attrib["xyz"]) if axis is not None else [1., 0., 0.]
    limit = joint.find("limit")
    if limit is not None:
      result["limit"] = {key: float(value) for key, value in limit.attrib.items()}
    joints.append(result)
  return dict(
    schema_version=1, source="resources/robots/go2/urdf/go2.urdf",
    sha256=hashlib.sha256(raw).hexdigest(),
    regenerate=("python -m src.tasks.pedipulation.robot --snapshot-from "
                "/path/to/My_unitree_go2_gym/resources/robots/go2/urdf/go2.urdf "
                "--output src/tasks/pedipulation/source_physics.json"),
    asset_options=dict(density=.001, replace_cylinder_with_capsule=True,
                       collapse_fixed_joints=False, self_collisions=0),
    approximation_notes=[
      "Collision-only links use source density; PhysX inferred inertials may differ.",
      "Capsule cylinder sections preserve URDF length; engine contact response differs.",
      "Direct parent-child excludes approximate PhysX adjacent-link filtering.",
    ],
    links=links, joints=joints,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Regenerate the source physics snapshot")
  parser.add_argument("--snapshot-from", required=True, type=Path)
  parser.add_argument("--output", type=Path, default=SNAPSHOT_PATH)
  arguments = parser.parse_args()
  arguments.output.write_text(json.dumps(snapshot_from_urdf(arguments.snapshot_from), indent=2) + "\n")
