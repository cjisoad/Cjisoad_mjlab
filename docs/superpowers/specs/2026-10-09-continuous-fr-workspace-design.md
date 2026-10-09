# Continuous FR keyboard workspace

The user approved computing a continuous, geometry-based right-front-foot
workspace, allowing keyboard exploration beyond the training envelope. This
delivery computes the cache and connects quadruped keyboard playback first.

## Geometry and cache

Use the task-local MuJoCo Go2 model, its FR hinge axes, link lengths, hard joint
limits, collision geometry and shared quadruped foot zero. Generate a Cartesian
grid with analytic IK witnesses for this explicitly validated three-joint chain.
Check witnesses using MuJoCo FK and collision distances. Preserve the component
connected to the reference stance; do not fill concavities with a convex hull.
Store an interpolated signed-distance grid, model/config fingerprint, valid
targets and joint witnesses in a compressed NPZ. This is a sampled nominal
geometric approximation, not a proof of dynamic tracking or arbitrary support
leg collision clearance.

## Keyboard

Interior targets pass through unchanged. Integrate inputs from the last accepted
target, clip each small movement at its first boundary crossing, and discard
rejected displacement. Transform anchor offsets into the cached body frame using
current base orientation. Keep a minimum operation lift and a current flat-ground
constraint. Remove nearest-neighbor target snapping in quadruped mode. The biped
and union transition workflows retain their existing behavior for this delivery.

The boundary cache is loaded once; no IK executes in keyboard updates. Provide a
generation command, plot, metadata report and the existing model playback command.
Training can consume the cached targets in a later change; its sampler is not
changed by this delivery.
