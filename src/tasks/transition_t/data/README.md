# V3 rise training reference

Reviewed pose path: `outputs/teacher_guided_rise_v3_20261011/tripod_to_biped_v3.npz`,
SHA256 `0e02179308aae91aa682f698281107394df57ae0ea0c9cf31ba97ff8c3568e47`.

Default training derivative: `tripod_to_biped_v3_speed125_train.npz`, 161
frames at 50 Hz and 3.20 s. SHA256
`db22ddd01b6b8c2e5ea46e1b571f7109c2a789cb53e35e88aa2d6bb0b8a01acf`.
The original 173-frame, 3.44 s derivative remains available as
`tripod_to_biped_v3_train.npz`. Evaluation supplies one continuous second of
upright position hold, so a complete first-frame success takes at least about
4.20 s with the default reference.

The matching physical entry bank is `entries_v3_speed125_train.npz`, SHA256
`ba99c841a5d04a708a4d1d66594fc0ad07a13fbcb87d6f5b4ce4dbee0e48cc74`. Its
metadata references the default motion hash; stored physical states,
velocities, action histories, limiter histories and trajectory split are
unchanged from the source bank.

Kinematic checks passed, including rear shell clearance >=44.948 mm and
support foot drift <=0.0905 mm. Unassisted forward-dynamics tests with PD,
inverse-dynamics PD and WBC did not complete this rise. This reference is a
training target whose dynamic learnability must be assessed in PPO, not a
validated deployable controller. Curriculum promotion requires actual physics
rollouts; no ghost/pose replay or reference fragment counts as full rise evidence.

The real entry bank is intentionally not committed. Recompute its FR error and
reference metadata with `scripts/retarget_transition_entries.py`; physical
states, velocities, action/limiter histories and trajectory splits are retained.
