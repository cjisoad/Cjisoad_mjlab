"""Convert an original 48/131 leg_manip checkpoint without overwriting it."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.tasks.leg_manip.rl.checkpoint import migrate_velocity_checkpoint


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('source', type=Path)
  parser.add_argument('destination', type=Path)
  args = parser.parse_args()
  checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
  migrated = migrate_velocity_checkpoint(checkpoint)
  args.destination.parent.mkdir(parents=True, exist_ok=True)
  # Exclusive creation protects both the source and any earlier migration.
  with args.destination.open('xb') as output:
    torch.save(migrated, output)
  print(json.dumps({'source': str(args.source.resolve()),
    'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
    'destination': str(args.destination.resolve()), 'iteration': migrated['iter'],
    'stage': migrated['infos']['loco_pedipulation']['stage'],
    'observations': migrated['infos']['leg_manip_observations']}, indent=2))


if __name__ == '__main__':
  main()
