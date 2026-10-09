# FR surface training integration

**Goal:** Share the cached FR geometry with loco teacher training, push the
requested changes to origin/wsl-validation and prepare a fresh remote run.

**Architecture:** Offline filtering of the geometry cache produces an interior,
nominal straight-path target bank. Command initialization loads the bank once
and the existing curriculum/reset sampler indexes GPU tensors. The source loco
task retains its own builder. Checkpoints record workspace provenance separately
from their unchanged actor/action ABI.

## Tasks

- [x] Add a training-bank builder/loader with geometry and input-file fingerprints,
  minimum lift 5cm, full geometric upper extent, 1cm inward distance margin and
  sampled 2mm nominal paths to the goal and to the landing reference.
- [x] Add a workspace construction hook to the loco source command; override it
  in the teacher so initialization no longer builds an unused source IK bank.
- [x] Load the bank through the teacher workspace helper; record metadata in
  checkpoint infos without changing the policy ABI.
- [x] Generate and inspect the training-bank report, compile the changed modules
  and review the complete staged diff. Do not add or run unit tests unless asked.
- [ ] Commit only this feature and the preceding continuous keyboard work;
  preserve unrelated dirty transition work. Push origin/wsl-validation normally.
- [ ] Once the user supplies SSH, inspect the server, sync an exact commit and
  matching cache into persistent storage, then start fresh loco teacher training
  with online W&B and checkpoint uploads. Report process/run identifiers.

The user has already authorized implementation, push and server training. SSH endpoint supplied: root@xj-member.bitahub.com:42311.
