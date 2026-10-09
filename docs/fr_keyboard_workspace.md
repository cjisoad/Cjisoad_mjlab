# Continuous right-front-foot keyboard workspace

Quadruped keyboard playback now loads a cached geometric boundary. Valid
interior targets remain continuous; they do not snap to the training target bank.
Training uses a path-filtered interior bank derived from this same geometry;
the keyboard can explore more of the boundary.

## Start

```bash
cd /home/eden/unitree_rl_mjlab
conda activate mjlab
python scripts/keyboard_teacher.py loco_pedipulation_t \
  --checkpoint-file outputs/loco_return_20261009/checkpoints/model_14999.pt \
  --device cuda:0 --num-envs 1 --workspace quadruped
```

The default cache is `outputs/fr_reachability/go2_fr_v1.npz`. Override it with
`--reachability-cache /path/to/cache.npz`. If missing or the robot snapshot has
changed, the loader asks for regeneration rather than silently using a box.

## Controls and expected behavior

| Key | Action |
|---|---|
| Right / Left | Increase / decrease anchor-forward FR offset dx |
| Up / Down | Increase / decrease anchor-left FR offset dy |
| Q / E | Raise / lower FR offset dz |
| W / S | Move robot forward / backward |
| A / D | Turn robot |
| R | Clear operation target and request normal return |
| Backspace | Reset environment |
| Space | Pause |

The offset overlay uses three decimal places (millimeters). `Boundary: FREE`
means the last input was accepted; `LIMIT` means geometry, the minimum lift,
or the ground constraint limited it. Interior single-axis moves leave other
coordinates unchanged. At a boundary, a small local tangential correction may
follow the surface; it cannot jump across a hole. Outward input is discarded,
so reversing direction responds immediately.

The minimum operation lift is 5cm. The upper extent is obtained from geometry,
without a manually imposed height ceiling. The nominal bounding extents printed at startup
are not a reachable rectangular box. Live base orientation is applied to queries;
in multi-environment playback the selected viewer world supplies that orientation.
Use one environment for boundary inspection.

## Generate or inspect the cache

```bash
MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/build_teacher_reachability.py
```

Default Cartesian resolution is 5mm. The calculation checks analytic IK against
MuJoCo FK for every witness, retains 0.025rad joint-limit margin and 5mm nominal
collision clearance, removes disconnected voxels, then constructs a trilinearly
interpolated signed-distance field with a 5mm query margin. The cache includes
interior target points and their joint witnesses. The default generator also
builds the offline training bank.
There is no runtime IK in keyboard boundary queries.

Artifacts beside the cache:

- `.json`: model fingerprint, grid settings, FK residual, bounds and counts.
- `.png`: 3D sampled boundary and sagittal slices.
- `.slices.json`: separate forward/backward intervals at selected heights and
  lateral offsets, including holes.

This is a sampled geometry approximation. Other legs are at the nominal
reference pose for collision checks; it does not certify arbitrary live support
leg collisions, joint-solution continuity, torque feasibility or policy balance.
## Training integration

`loco_pedipulation_t` loads `outputs/fr_reachability/go2_fr_v1_train.npz` once,
then samples GPU tensors on reset. There is no IK during initialization or reset.
The original loco task keeps its original workspace builder. The biped teacher,
policy observation sizes, reward weights, gait and unified return are unchanged.

The training bank uses a 10mm inward distance margin and minimum lift 50mm.
Straight nominal reaches from the shared zero and their reverse returns are
sampled every 2mm at most, requiring geometric membership and a continuous IK
branch (maximum per-sample joint change 0.15rad). These are offline checks at
nominal body/support-leg poses; actual landing height and dynamics can differ.
The stored joint witnesses follow the selected path branch.

For the generated Go2 cache, 579,517 candidates yield **386,764** targets, with
2,888 easy curriculum targets. Bounding extents relative to the shared zero:

| Axis | Training extent (m) |
|---|---|
| dx | -0.355 to +0.390 |
| dy | -0.240 to +0.335 |
| dz | +0.050 to +0.350 |

These extrema do not form a reachable box. No 0.35m cap is imposed: higher
geometrically reachable endpoints are rejected by nominal straight-path checks.
Keyboard paths can curve around holes, so its geometric height extent is larger.
The existing curriculum retains easy targets until stage 3, then uses the full
bank. Checkpoints include `infos.fr_workspace` with geometry/cache fingerprints,
counts, bounds, margins and path settings.

To regenerate only the training bank:

```bash
python scripts/build_teacher_reachability.py --training-only
```

Both NPZ files are generated artifacts under ignored `outputs/`. Deployment
must copy the matching caches with the committed source, or run the generator.
