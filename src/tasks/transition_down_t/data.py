"""Approved local assets and portable checkpoint fingerprints."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT/'outputs/biped_to_tripod_preparation_20261010'
DEFAULT_MOTION_PATH = DATA_DIR/'biped_to_tripod_reference.npz'
DEFAULT_ENTRY_BANK = DATA_DIR/'biped_entry_bank.npz'
MOTION_SHA256 = 'fa49119da56b0eb2ac2dc49d5b65dfe310aad8033a5e15565ba5079872bb11aa'
BANK_SHA256 = 'ff962e464cd928e44dea76f60363867f5361a45564ad7adbd6a43632d7193058'
ALIGNMENT = 'teacher_body_y_cross_up_v1_xy_heading'
TEACHER_SHA256 = {
  'quadruped': '1176fb6c71ba5c05c7d2c8ac36f6492f9b3dd37ba8dcafc172e2c3da0bc5d0fd',
  'biped': 'f919492bbb59fe29158e6aa6cc0cc62acf12bcf16cbdec70a86705c9ba084b6b',
}
REVIEW_SHA256 = {
  'manifest.json': '265b970972df379397a2c9dc014af29ccaf29aee60845e25f392499f8359ce09',
  'randomization_config.json': 'f4fecfdc152a0d0a3f92b3b8651f3358fd336bf51229d0dfb31288ffbe8c20fc',
  'baseline_configuration.json': '03abcbabfa64d88d354eb797d3709ff8fb56f9444d48301085ba0d69a55b9c5a',
}


def file_sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024*1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def verify_approved_assets(*, motion_file=DEFAULT_MOTION_PATH, bank_file=None):
  required = {Path(motion_file): MOTION_SHA256}
  required.update({DATA_DIR / name: value for name, value in REVIEW_SHA256.items()})
  if bank_file is not None:
    required[Path(bank_file)] = BANK_SHA256
  for path, expected in required.items():
    if file_sha256(path) != expected:
      raise ValueError(f'Approved reverse-transition asset SHA256 differs: {path}')


def verify_teachers(quad_file, biped_file):
  for name, path in [('quadruped', quad_file), ('biped', biped_file)]:
    if file_sha256(path) != TEACHER_SHA256[name]:
      raise ValueError(f'Approved {name} teacher SHA256 differs: {path}')
