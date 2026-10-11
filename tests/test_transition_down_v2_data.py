import unittest
import numpy as np
from src.tasks.transition_down_v2.data import MOTION_FILE, BANK_FILE, verify_assets, LandingBank

class VersionedData(unittest.TestCase):
    def test_frozen_reference_and_disjoint_splits(self):
        manifest=verify_assets()
        self.assertEqual(manifest['motion_sha256'],'a806e849906846e66bf2421afad780132478a93f71c4b4f4a15300ae7b902646')
        b=LandingBank(BANK_FILE,motion_sha256=manifest['motion_sha256'])
        splits={}
        for name,count in [('train',512),('validation',64),('final',64)]:
            ids=b.eligible(split=name,max_speed=1.2,max_fr_error=.45)
            splits[name]=set(b.data['trajectory_id'][ids].tolist())
            self.assertEqual(len(splits[name]),count)
        self.assertFalse(splits['train']&splits['validation'])
        self.assertFalse(splits['validation']&splits['final'])
        with np.load(MOTION_FILE) as z:self.assertEqual(len(z['qpos']),186)
        self.assertLess(manifest['FR_error_recompute_max_delta_m'],1e-5)

if __name__=='__main__':unittest.main()
