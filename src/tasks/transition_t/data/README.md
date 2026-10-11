# V3 rise training reference

Reviewed pose path: `outputs/teacher_guided_rise_v3_20261011/tripod_to_biped_v3.npz`,
SHA256 `0e02179308aae91aa682f698281107394df57ae0ea0c9cf31ba97ff8c3568e47`.

Training derivative: `tripod_to_biped_v3_train.npz`, 173 frames, 50 Hz, 3.44 s,
SHA256 `1abcb44557a1e1b64dfa0e4bde5305908574b0f55f71c6efbb96b4cf44fcbf3d`.
The preview's final 50 stationary hold frames were removed; final stored
velocities are zero. All retained pose samples are unchanged. Evaluation
supplies one continuous second of upright position hold, so a complete
first-frame success takes at least about 4.44 s.

Kinematic checks passed, including rear shell clearance >=44.948 mm and
support foot drift <=0.0905 mm. Unassisted forward-dynamics tests with PD,
inverse-dynamics PD and WBC did not complete this rise. This reference is a
training target whose dynamic learnability must be assessed in PPO, not a
validated deployable controller. Curriculum promotion requires actual physics
rollouts; no ghost/pose replay or reference fragment counts as full rise evidence.

The real entry bank is intentionally not committed. Recompute its FR error and
reference metadata with `scripts/retarget_transition_entries.py`; physical
states, velocities, action/limiter histories and trajectory splits are retained.
