# Approved Go2 SuperMimic v2 implementation
User approval 2026-11-16. Existing feature branch, preserve existing uncommitted work. No long training. New models from random initialization; teachers frozen; deployment48D GRU256 actions12 at50Hz unchanged.
1. Six-task real-frame-balanced curriculum sampler .20/.20/.20/.15/.15/.10. Long command-changing walking, foot hold/retarget, combined and transitions. Retain teacher trajectories floor.10.
2. Sensitive velocity/foot exp+Huber rewards; operation gates, mode-specific support/gait and continuous posture.
3. Unified weighted student gradient step, task-balanced loss,100-update task and500-update objective/layer conflict diagnostics. No gradient surgery.
4. Task/contact-aware replay, heldout real multistep model validation, reliability-gated4->8 imagination horizon and real-reward critic calibration.
5. Focused/regression tests plus real short CPU/GPU startup, diagnostics, save/reload/export. Document launch and limitations. No baseline, ablation or strict performance acceptance.
