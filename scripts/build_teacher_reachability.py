"""Generate the cached continuous FR keyboard workspace from MuJoCo geometry."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fr-reachability-mpl')


def plot_cache(path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import numpy as np
  from src.tasks.teacher_common.reachability import ReachabilityGrid
  grid = ReachabilityGrid(path)
  targets = grid.targets[grid.targets[:, 2] >= .05]
  fig = plt.figure(figsize=(14, 6))
  ax = fig.add_subplot(121, projection='3d')
  # Thin boundary samples for a readable surface preview without another dependency.
  from scipy.ndimage import binary_erosion
  inside = grid.distance >= grid.margin
  edge = inside & ~binary_erosion(inside)
  ijk = np.argwhere(edge)
  points = grid.lower+ijk*grid.spacing
  points = points[points[:, 2] >= .05]
  points = points[::max(1, len(points)//16000)]
  ax.scatter(*points.T, c=points[:, 2], s=1, alpha=.35, cmap='viridis')
  ax.set(xlabel='FR dx (m)', ylabel='FR dy (m)', zlabel='FR dz (m)',
         title='Sampled FR boundary (nominal body pose)')
  ax.set_box_aspect(np.ptp(targets, axis=0))
  side = fig.add_subplot(122)
  x = np.arange(grid.distance.shape[0])*grid.spacing+grid.lower[0]
  z = np.arange(grid.distance.shape[2])*grid.spacing+grid.lower[2]
  xx, zz = np.meshgrid(x, z, indexing='ij')
  for y in (-.10, 0., .10):
    query = np.stack((xx, np.full_like(xx, y), zz), axis=-1)
    values = grid.query(query.reshape(-1, 3)).reshape(xx.shape)
    contour = side.contour(xx, zz, values, levels=[grid.margin],
                           colors=[{-.10:'tab:blue', 0.:'tab:orange', .10:'tab:green'}[y]])
    side.plot([], [], color=contour.get_edgecolor()[0] if hasattr(contour, 'get_edgecolor') else {-.10:'tab:blue', 0.:'tab:orange', .10:'tab:green'}[y], label=f'dy={y:+.2f}m')
  side.axhline(.05, color='gray', linestyle=':')
  side.add_patch(plt.Rectangle((-.30, .05), .60, .30, fill=False, linestyle='--',
                               edgecolor='black', label='Previous training box, dy=0'))
  side.set(xlabel='FR dx (m)', ylabel='FR dz (m)', ylim=(0., grid.upper[2]),
           title='Continuous boundary slices; inside must remain connected')
  side.grid(alpha=.25); side.legend(); side.set_aspect('equal')
  fig.tight_layout()
  output = Path(path).with_suffix('.png')
  fig.savefig(output, dpi=160)
  plt.close(fig)
  return output


def write_slices(path):
  """Report separate admissible X intervals rather than an enclosing box."""
  import numpy as np
  from src.tasks.teacher_common.reachability import ReachabilityGrid
  grid = ReachabilityGrid(path)
  x = np.arange(grid.lower[0], grid.upper[0], .0005)
  rows = []
  for y in (-.10, 0., .10):
    for z in (.05, .10, .15, .20, .25, .30, .35, .40, .50, .60):
      p = np.column_stack((x, np.full_like(x, y), np.full_like(x, z)))
      mask = grid.query(p) >= grid.margin
      changes = np.diff(np.r_[False, mask, False].astype(int))
      intervals = [[round(float(x[a]), 4), round(float(x[b-1]), 4)]
                   for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]
      rows.append(dict(dy_m=y, dz_m=z, dx_intervals_m=intervals))
  output = Path(path).with_suffix('.slices.json')
  output.write_text(json.dumps(dict(note='Nominal body pose; sampled boundary. Multiple intervals indicate holes; not all enclosing-box targets are reachable.', slices=rows), indent=2)+'\n')
  return output


def main():
  from src.tasks.teacher_common.reachability import DEFAULT_CACHE, build_cache
  from src.tasks.teacher_common.training_workspace import DEFAULT_TRAINING_CACHE, build_training_cache
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=DEFAULT_CACHE)
  parser.add_argument('--spacing', type=float, default=.005)
  parser.add_argument('--joint-margin', type=float, default=.025)
  parser.add_argument('--collision-margin', type=float, default=.005)
  parser.add_argument('--query-margin', type=float, default=.005)
  parser.add_argument('--plot-only', action='store_true')
  parser.add_argument('--training-only', action='store_true',
                      help='Reuse the geometric cache and only build its training target bank')
  parser.add_argument('--skip-training', action='store_true')
  parser.add_argument('--training-output', type=Path, default=DEFAULT_TRAINING_CACHE)
  parser.add_argument('--training-margin', type=float, default=.01)
  args = parser.parse_args()
  if args.training_only and (args.plot_only or args.skip_training):
    parser.error('--training-only cannot be combined with --plot-only or --skip-training')
  if not args.plot_only and not args.training_only:
    build_cache(args.output, spacing=args.spacing, joint_margin=args.joint_margin,
                collision_margin=args.collision_margin, query_margin=args.query_margin)
  if not args.training_only:
    print(f'Plot: {plot_cache(args.output)}', flush=True)
    print(f'Slices: {write_slices(args.output)}', flush=True)
  if not args.plot_only and not args.skip_training:
    build_training_cache(args.output, args.training_output, margin=args.training_margin)


if __name__ == '__main__':
  main()
