# Continuous FR Workspace Implementation Plan

**Goal:** Compute and cache the FR geometric workspace and restore continuous
quadruped keyboard commands over its larger usable region.

**Architecture:** Offline chain-specific IK plus MuJoCo geometry validation
creates a Cartesian signed-distance cache. A NumPy runtime reader performs
continuous membership and segment clipping in the current body frame.

**Tech Stack:** Python, NumPy, SciPy ndimage, MuJoCo, Matplotlib.

## Tasks

- [x] Add `src/tasks/teacher_common/reachability.py`: validated chain geometry,
  analytic IK, cache fingerprint/load, trilinear query, movement clipping.
- [x] Add `scripts/build_teacher_reachability.py`: build once, report bounds and
  cache metadata, export boundary slice and 3D visualization.
- [x] Update only relevant sections of `scripts/keyboard_teacher.py`: quadruped
  boundary adapter, integrate accepted targets, update live frame, CLI cache path.
  Preserve concurrent transition CLI work.
- [x] Generate the actual cache under `outputs/fr_reachability/` and inspect its
  model FK residuals, component connectivity, usable ranges and generation log.
- [x] Update user documentation and provide the exact local playback command.

## Verification

The user requested computing the region and will inspect playback themselves.
Follow the session instruction against unsolicited test additions/runs: do not
add or run a unit test suite. Validate the generated geometry as part of the
requested calculation, inspect the surface plot and use syntax compilation and
CLI help to check integration. Do not claim GUI playback was inspected unless
it was actually launched. Do not commit or push in this task.

## Result

Generated `outputs/fr_reachability/go2_fr_v1.npz` at 5 mm resolution in 27.4 s,
with 819739 checked IK witnesses and maximum FK residual 4.14e-16 m. Runtime
reader loaded the cache to export the boundary plot and sagittal intervals.
Syntax compilation, keyboard CLI help and edited-script whitespace check passed.
No unit suite or native GUI playback was run; the user will inspect the GUI.
