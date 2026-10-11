"""Reviewed V2 assets and sealed trajectory splits, with V1 preserved."""
import json
from pathlib import Path
import torch
from src.tasks.transition_down_t.data import file_sha256
from src.tasks.transition_t.entry_bank import EntryStateBank

ROOT=Path(__file__).resolve().parents[3]
ASSET_DIR=ROOT/'src/assets/motions/go2/landing_v2'
MOTION_FILE=ASSET_DIR/'biped_to_tripod_v2.npz'
BANK_FILE=ASSET_DIR/'biped_entry_v2.npz'
MOTION_SHA256='a806e849906846e66bf2421afad780132478a93f71c4b4f4a15300ae7b902646'

def verify_assets(motion_file=MOTION_FILE,bank_file=BANK_FILE):
    manifest=json.loads((ASSET_DIR/'manifest.json').read_text())
    if manifest['motion_sha256']!=MOTION_SHA256 or file_sha256(motion_file)!=MOTION_SHA256:raise ValueError('V2 reference SHA differs')
    if bank_file is not None and file_sha256(bank_file)!=manifest['bank_sha256']:raise ValueError('V2 bank SHA differs')
    return manifest

class LandingBank(EntryStateBank):
    def eligible(self,*,split,max_speed,max_fr_error):
        if split in ('train','holdout'):return super().eligible(split=split,max_speed=max_speed,max_fr_error=max_fr_error)
        if split not in ('validation','final'):raise ValueError('Unknown V2 split')
        eligible=super().eligible(split='holdout',max_speed=max_speed,max_fr_error=max_fr_error)
        remainder=5 if split=='validation' else 0
        return eligible[self.data['trajectory_id'][eligible]%10==remainder]
    def coverage(self):
        from src.tasks.transition_t.entry_bank import LEVELS
        result=super().coverage()
        for split in ('validation','final'):
            result[split]=[]
            for level,c in enumerate(LEVELS):
                ids=self.eligible(split=split,max_speed=c['max_speed'],max_fr_error=c['max_fr_error'])
                result[split].append(dict(level=level,states=len(ids),trajectories=len(self.data['trajectory_id'][ids].unique())))
        return result
