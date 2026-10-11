"""Retarget a preserved physical teacher state bank to another rise reference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--entry-bank', type=Path, required=True)
  parser.add_argument('--source-motion-file', type=Path, required=True)
  parser.add_argument('--motion-file', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  sys.path.insert(0, str(ROOT))
  from src.tasks.transition_t.entry_retarget import retarget_entry_bank
  report = retarget_entry_bank(args.entry_bank, args.motion_file, args.output,
    source_motion_file=args.source_motion_file)
  args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
  main()
