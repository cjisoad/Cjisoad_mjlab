"""Restart version1 weights with the stage0 stance course and smaller exploration."""
import argparse
from pathlib import Path
import torch
from src.tasks.leg_manip.rl.checkpoint import migrate_stance_checkpoint

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('source', type=Path)
  parser.add_argument('destination', type=Path)
  args = parser.parse_args()
  if args.destination.exists():
    parser.error('Destination exists; choose a new path')
  checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
  migrated = migrate_stance_checkpoint(checkpoint)
  args.destination.parent.mkdir(parents=True, exist_ok=True)
  torch.save(migrated, args.destination)
  print(f'Saved {args.destination}: stage0, std=.2, fresh optimizer and course')

if __name__ == '__main__': main()
