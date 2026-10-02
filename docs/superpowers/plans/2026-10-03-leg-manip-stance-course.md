# leg_manip Stance Course Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or inline execution for coupled integration. Follow test-driven-development. Existing user changes are already overlaid in the isolated workspace and must be preserved; do not commit them implicitly.

**Goal:** Learn both stances and bidirectional switching first, then locomotion, then manipulation with fixed nominal FR posture commands in early stages.

**Architecture:** Keep the 6D command and 51D/131D observations. Add task-local stance references and course metrics; use three versioned stages and mode-specific readiness gates. Distinguish stance references from manipulation for rewards and check each capability over fixed windows.

**Tech Stack:** Python, PyTorch, MuJoCo/mjlab, rsl-rl, unittest.

## Task 1: Stance references, stages and sampling

Files: constants.py, mdp/foot_workspace.py or a new mdp/stance.py, mdp/commands.py; tests/test_leg_manip_course.py and existing command tests.

- [x] Write tests for stage 0 zero velocity, only zero/nominal FR targets, both mode requests, nonrandom nominal FK witness; stage 1 no random operation targets; stage 2 samples both operation segments.
- [x] Run `MUJOCO_GL=disable python -m unittest tests.test_leg_manip_course -q` and verify expected feature failures.
- [x] Derive nominal FR using `REARED_ROTATION @ FK(DESIRED_ANGLES) - workspace.zero`, not a hand-entered coordinate. Add named stage constants 0/1/2 and a course version.
- [x] Implement automatic sampling, retained easy tasks, bounded velocity levels, long enough stance dwell. Preserve manual commands and existing low/high hysteresis behavior.
- [x] Verify changed tests and manual phase-machine regressions.

## Task 2: Genuine stance feedback and initialization

Files: mdp/rewards.py, env_cfg.py, a new task-local reset event if used, command reference durations; reward and integration tests.

- [x] Test that nominal FR does not suppress default FR/front symmetry rewards; zero-speed rear contact rewards two contacts; swing clearance is disabled at zero speed; front support/tilted low-base states fail readiness.
- [x] Test mode-aware initialization and transitions, including subset reset and no immediate termination of a rear-leg stand.
- [x] Watch these tests fail before implementation.
- [x] Reuse existing stance rewards, distinguish nominal stance from operation, and provide continuous posture/support feedback outside the height gate. Reduce FR tracking priority in stages 0/1. Adapt contact/gait terms to static vs moving commands.
- [x] Implement reference/reset support with task-local changes and preserve all velocity observation contracts.
- [x] Run rewards, commands, velocity and real-env tests.

## Task 3: Fixed-window course validation, logging and checkpoint safety

Files: new task-local course metrics, mdp/curriculums.py, rl/__init__.py and task-local logging; tests/test_leg_manip_course.py, tests/test_leg_manip_rl.py.

- [x] Test that good quad + bad biped cannot promote; insufficient bidirectional switches cannot promote; an unsuccessful window cannot be retried immediately; two or three complete successful windows are needed.
- [x] Test partial reset does not erase other environments' switch tracking and terminal/interrupted transitions are failures rather than successes.
- [x] Test versioned checkpoint roundtrip and rejection of legacy course states.
- [x] Implement independent per-mode stance, transition and velocity statistics. Fixed windows must reset time after every decision. Log failure reasons, counts and pass streak; retain legacy landing counters while correcting tripod/biped tracking labels locally.
- [x] Persist all new stage/window/velocity-level state; never restore an old course silently. Leave other task runner behavior unchanged.
- [x] Verify targeted tests and `scripts/smoke_leg_manip.py --help` / real PPO smoke.

## Task 4: Documentation, review and experiment

Files: doc/LEG_MANIP.md, scripts/smoke_leg_manip.py if needed, outputs/leg_manip_stance_course_20261003.

- [x] Explain rewards vs hard constraints, stage numbers and exact nominal reference definition; update obsolete stage references in docs/tests.
- [x] Run the complete relevant task/source-task regression suite once new tests pass, then real CPU/PPO save-load and optional 4090 smoke.
- [x] Review design compliance then code quality; fix actual issues and rerun affected checks.
- [x] Compare changed worktree files with the saved baseline hashes and sync only new incremental files into the shared repository without overwriting concurrent user edits.
- [x] Preserve the old training run. If the authorized 4090 is available, package this version and launch a fresh independent run with resume=False, seed 42, 4096 environments, 15000 updates, every100 checkpoint retained/uploaded at /pedipulation/legmanip1. If the server is unavailable, stop GPU work and report that the user needs to start it.

## Implementation progress (isolated stance_course worktree)

- [x] Reproduced missing stage-0 biped sampling and nominal-operation distinction with unittest regressions; implemented three named stages, exact FK nominal biped reference, mixed-mode sampling, and retained manual phase/hysteresis behavior.
- [x] Reproduced nominal FR posture suppression, static rear-support and swing-clearance issues; implemented task-local reward wrappers and whole-body readiness. Automatic nominal transitions use 1.8 seconds; dwell is 6–9 seconds; biped exit allowance is 3.5 seconds. Optional biped reset deliberately omitted; every reset uses the normal quad physical reset.
- [x] Reproduced old curriculum/window/checkpoint incompatibility and missing held-endpoint switch accounting; implemented streaming independent stance/walking groups, rise/descent attempt outcomes, fixed-window restart on every decision, two passing windows, four velocity levels, task-local logger, versioned runner save/load and legacy rejection.
- [x] Fresh integrated suite: 69 tests passed; full leg_manip discovery: 93 tests passed including CPU environment; shared source-task/keyboard regressions: 55 tests passed. Added further timeout/window-stream/descent boundary coverage and re-verification is in progress.
- [x] Independent full-repository tests (220 total, 217 passed, 3 X11 skips), CPU4env2-update PPO/save-load/course-state/ONNX smoke, and specification recheck passed.
- [x] Final documentation complete; remote experiment prepared and waiting for server SSH to recover.

Course checkpoint metadata is `infos['leg_manip_course']`, version 1, name `stance_walk_operation`; it persists stage, fixed-window start, group/switch windows, episode/failure counts, passing streak, velocity level, and cumulative landing/switch counts. Old `loco_pedipulation` stage metadata does not satisfy this version contract.

### Review fixes and final targeted verification

- [x] Logger now separates six quad/tripod/biped × stationary/moving groups, including mode-relative gravity angle; reset tracking extras also separate the three support modes.
- [x] Switch attempts require a physically valid source stance. Healthy moving sources use the previous command velocity; invalid sources count as `uncertified` failures. Every reset starts with a normal quad 6–9 second dwell before mixed-mode sampling.
- [x] Added per-group survival gates (at least 16 window-local completed episodes and 80% survival in each required group), including active-command failures before a stable HOLD. Quad survival cannot mask failed biped walking.
- [x] Last-decision sample/readiness/tracking/survival/switch diagnostics survive window clearing and checkpoint resume. Lifetime switch counters now have five events: attempts, succeeded, timeouts, interrupted, uncertified.
- [x] Final targeted command/reward/course/velocity/runner unittest run passed 81 tests in 5.826 seconds. The added tests reproduce false descent certification, timeout final-frame certification, group survival dilution, log segmentation, malformed checkpoint counts, and missing diagnostics before their fixes.
- [x] Updated `doc/LEG_MANIP.md` for the three-stage course, nominal references, source/endpoint readiness, resets, reward gates, fixed windows, logger fields, explicit legacy rejection and fresh course PPO smoke commands.
- [x] Main-agent independent final full-repository tests, real CPU PPO/save-load smoke, reviewer recheck, and delivery. The root agent owns the new smoke script and deployment package; the remote server experiment remains its responsibility.

### Independent final verification

- [x] `DISPLAY='' MUJOCO_GL=disable ... PYTHONPATH=tests:. python -m unittest discover -s tests -q`: 220 tests, OK with 3 unavailable-X11 keyboard skips, exit 0.
- [x] `scripts/smoke_leg_manip_course.py`: fresh stage0, actor51/critic131, exact critic velocity/small actor noise, 2 actual CPU PPO updates, finite weights, policy/course reload and ONNX51 checked, exit 0.
- [x] Specification reviewer independently reproduced the former all-biped-moving-fail / overall80%-survival promotion case and verified it now blocks. Twelve targeted recheck tests passed; source-task baseline hashes unchanged.
- [x] Final code-quality review approved; 32 course tests independently passed, with no important/critical remaining issues. Synchronized delivery recorded in synced_changes.json.
- [ ] Remote training: two SSH checks returned `kex_exchange_identification: Connection closed by remote host`. Per user instruction, GPU work is stopped until the server is reopened; no new run has started.
