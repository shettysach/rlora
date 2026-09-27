# Device agent: finish the BONES BC pipeline check

Work in `~/Desktop/rlora` on balerion. Read `AGENTS.md`, `AIM.md`, and
`BC_BONES.md` first. This document authorizes the remaining read-only
evaluation and small, focused diagnostic work. Preserve existing artifacts;
write new results under distinct names. The BONES task is a bounded check of
the local Ψ₀ BC pipeline, not a project to perfect this one motion.

## Current evidence

- Direct replay of episode 47 body tokens through SONIC/MJLab works at both
  50 Hz and the policy's 30 Hz clock. The 30 Hz replay traveled 5.39 m and
  reached 0.89 rad maximum heading change without falling. The recorded
  motion returns near its initial heading; it is not a 180° turn.
- The 1,000-step episode-47 adapter improved paired body flow loss from 15.59
  to 2.93 and sampled body-token error from 0.133 to 0.089 on its training
  episode. In two matched 9.1-second rollouts it traveled 1.41 and 1.06 m,
  versus 0.64 and 0.63 m for the base. Its maximum heading changes were
  0.45 and 0.42 rad, versus 0.24 and 0.12 rad. None fell. The improvement is
  visible but remains far short of expert replay.
- A separate run trained on episodes 47–51 with 52–53 held out and was stopped
  at step 1,149. Its step-1,000 checkpoint exists on balerion. Its held-out
  and physical results have **not** been verified. Locate that checkpoint
  from the actual run directory; do not assume a path from this document.
- The current MJLab observation differs visibly from the BONES camera view.
  Its contribution to the rollout gap is unknown. Shared simulator changes
  were reverted.
- About 32.4% of episode-47 Dex3 targets exceed the starting checkpoint's
  hand bounds, versus 0.31% of body-token values. Dex3 is not controlled in
  these rollouts. Keep the 78 real demonstrated targets in BC and leave the
  hand mapping alone for this pipeline check.

## Next work, in order

1. Inspect `git status`, the split config, saved step-1,000 adapter, available
   paired evaluator, and existing result files on balerion. Confirm that the
   split is **47–51 train / 52–53 held out**, with no frame-level leakage.
   Record the exact code revision, checkpoint path, config, and evaluation
   settings used. Use the existing Ψ₀ policy and checkpoint loader.

2. Run paired offline evaluation of the **base and split adapter** on all
   held-out observations, with the same images, states, prompt, flow
   timesteps, noise, and sampling seeds for each pair. Report body `0:64`
   flow loss and sampled body-token error separately from Dex3 `64:78`.
   Inspect per-episode results for both episodes 52 and 53, not just a pooled
   average. A one-sample `--baseline-only` output is insufficient. Also
   compare train-split results so a good training score cannot be mistaken
   for held-out learning. Save a machine-readable summary.

3. If held-out body results improve credibly, run matched 9.1-second
   closed-loop base/adapter rollouts with the same initial conditions, seed,
   Ψ₀ sampling settings, 30 Hz action clock, SONIC controller, and MJLab
   scene. Use at least two seeds or starting conditions. Save video and JSON
   for every pair. Compare path length, heading trajectory and maximum turn,
   falls, and visual motion. The previous episode-47 comparisons and expert
   replay are context, not a target threshold that the split adapter must
   reach. Do not use a changed scene for only one side of a comparison.

4. Make one decision from the evidence:

   - If the split adapter improves held-out body behavior **and** produces
     recognizable, more reliable walk-and-turn motion than the base, mark
     the BONES BC pipeline validated. Stop BONES training and report what
     remains imperfect. The next research step is pushing demonstrations.
   - If held-out body behavior does not improve, report that the split has
     not generalized. Check one concrete suspect supported by the data,
     such as chunk alignment, normalization, or the split episode contents.
   - If held-out scores improve but motion remains weak, inspect the
     closed-loop inputs and sampling path first. Save representative BONES
     and MJLab images **after the existing Ψ₀ image transform** and compare
     framing/content. A controlled camera/scene comparison may follow, but
     preserve Ψ₀ checkpoint preprocessing, action semantics, and inference
     settings. Keep scene edits local to the experiment and compare base and
     adapter under identical conditions.

   Investigate only one supported cause before deciding whether another
   bounded run is justified. Do not spend GPU time merely to chase the
   expert's 5.39 m trajectory or the task label's implied turnaround.

## Report back

Provide the checkpoint and result paths, held-out body metrics for episodes
52 and 53, matched base/adapter rollout metrics and videos, the decision
above, and one specific next action. State separately what is measured and
what remains a hypothesis. Do not call upright walking alone a success.

Use `uv run --extra cu128 --frozen` for commands on balerion. Existing
`src/train_bc.py --baseline-only` checks one sample and reports a validation
loss, while the paired evaluator on the device should be used for the full
split comparison. Read each command's `--help` before choosing options; do
not invent flag names from older documentation.
