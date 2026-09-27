# Device agent: finish BONES walk-and-turn BC validation

Work from the `rlora` repository on `balerion` (`~/Desktop/rlora`). Treat this
document as authorization to run the remaining experiments, inspect artifacts,
fix local code, and iterate without asking the user for each command. Read
`AGENTS.md`, `AIM.md`, and `BC_BONES.md` first. Keep the implementation small
and auditable. Preserve existing result directories; use new names for new
runs. Report a real blocker only after investigating it. Do not claim success
from a single low loss or from upright walking alone.

## Objective and boundaries

Show that the released SONIC-compatible Ψ₀ checkpoint, after local flow-based
BC on BONES demonstrations, generates **body tokens** that execute the walking
and turning motion better through SONIC + MJLab than the original checkpoint.
Use the same local Ψ₀ policy for training and inference. Keep the base frozen
and train only its LoRA adapter. Keep all 78 demonstrated action dimensions
(64 body + 14 Dex3) in BC; the last two 80-D checkpoint channels are masked.
Dex3 is not controlled by this MJLab rollout, so hand clipping is a diagnostic,
not a reason to change hand targets, mask them, or delay the body experiment.
No RL, box pushing, distributed trainer, or generic framework.

The dataset and checkpoint are configured in `configs/bc_bones_walk.json`.
The dataset is `wsagi/SONIC-VLA-BonesSeed-V2` at revision
`4d84c8009ad601d24c1fe493c45b2715536b7bef`. Episode 47 is the initial
one-episode overfit target. Episodes 47–51 train and 52–53 validate only after
the one-episode result works. The original Ψ₀ repository at commit
`4f3720d45e102b36d7c3e9465ab8062274170518` is a reference, never a
runtime dependency.

## Established results

The released checkpoint loaded on balerion and returned finite `[1, 30, 80]`
actions. The existing `results/bones-overfit/ckpt_500` adapter reloads and
returns finite actions, but its random training losses fluctuate; that does
not establish overfitting. Its 10-second closed-loop rollout traveled 1.51 m,
displaced 0.92 m, reached 0.414 rad maximum heading change, and did not fall.

The source episode is **not a 180° turnaround**. Its recorded root orientation
reaches 0.885 rad (51°) maximum heading change and returns close to the
starting heading. Direct replay of its 64-D tokens through SONIC/MJLab worked:

| Replay | Path length | Displacement | Maximum turn | Fell |
| --- | ---: | ---: | ---: | --- |
| Source 50 Hz, 9.1 s | 5.404 m | 5.211 m | 0.908 rad | No |
| BC clock 30 Hz, 9.1 s | 5.395 m | 5.203 m | 0.889 rad | No |

Thus the SONIC interface, MJLab simulation, source tokens, and 50→30 Hz
resampling can execute the recorded motion. There is a substantial gap between
the BC policy and demonstrated execution. The 1.51 m BC rollout used 10 s;
compare future rollouts at the same 9.1 s duration and initial conditions.

The episode-47 preprocessing reported 0.31% of body-token values and 32.4% of
hand values outside the **checkpoint** action bounds. Several left Dex3
joints appear to have a different sign convention. Keep this visible, but
focus first on the body path. Do not silently widen bounds or alter the hand
mapping. The first BC sample had a base flow loss of 1.13 and an adapter flow
loss of 0.60; those are different random flow draws and cannot be compared.

## Work to do, in order

1. Inspect `git status`, existing results, and the current `train_bc.py`,
   `evaluate_bc.py`, `planner/psi0.py`, and `data/sonic_bones.py` before
   changing anything. Confirm the source episode, checkpoint, 30 Hz clock,
   image transform, 45-D state order, 80-D target order, and SONIC token
   quantization. Use the pinned upstream code to resolve any mismatch.

2. Measure **paired offline flow loss across all 273 resampled observations**
   of episode 47 for the original checkpoint and `ckpt_500`. Use identical
   images, states, prompts, timesteps, and noise for each pair. Set both
   policies to evaluation mode. Report per-sample and aggregate loss for
   body channels `0:64` and Dex3 channels `64:78` separately, respecting the
   existing mask and upstream time-sum loss reduction. Include more than one
   fixed flow draw per observation if needed to reduce noise. A one-sample
   `--baseline-only` run is insufficient. If a small reusable evaluator is
   needed, put it under `src/` and save JSON results in a new `results/`
   directory. Do not turn this into a generic evaluation framework.

3. Inspect the produced body tokens as well as loss. On fixed observations,
   compare base and adapter sampled chunks with the expert chunk using the
   same inference seed. Check shape, finite values, action normalization,
   token ranges, quantization, and whether the sampled tokens vary over the
   motion. Do not interpret one stochastic sample as a conclusive metric.

4. If the adapter has not learned the body targets, diagnose the specific
   cause before adding training steps: action/state layout, image/prompt
   pairing, 50→30 Hz chunk alignment, flow noise and target, masked loss,
   state dropout, LoRA coverage/gradients, parameter updates, and adapter
   loading. Compare local computations with the pinned Ψ₀ source where
   useful. Make the smallest justified fix and run focused checks for it.
   If the path is correct and training is simply insufficient, try a bounded
   one-episode run with adjusted step budget or learning rate in a copied
   config. Log why each change is made and retain the existing run.

5. When offline body learning is credible, run matched 9.1-second physical
   rollouts of the base and the candidate adapter, with the same seed,
   reset, action rate, controller, and simulation settings. Record metrics
   **and video**. Compare path length and heading trajectory against the
   30 Hz expert replay, not against an assumed 180° turn. If physical motion
   still fails despite good offline loss, investigate sampling, normalization,
   replan/chunk behavior, and closed-loop observation mismatch. Fix and
   repeat as needed. Use more than one seed or starting condition when a
   candidate appears successful so the result is not a one-off.

6. Only after episode 47 demonstrates clear physical improvement, copy the
   config for the episode-level 5/2 split (47–51 train, 52–53 held out),
   train, evaluate paired offline losses on both splits, and compare matched
   closed-loop rollouts with the base. Keep split episodes disjoint. The
   final claim should be about recognizable and more reliable execution,
   supported by videos and metrics; note the limits of a small dataset.

Useful commands (change output paths to avoid collisions):

```sh
uv run --extra cu128 --frozen python src/evaluate_bc.py \
  --seconds 9.1 --record-video results/base-9p1.mp4 \
  --output results/base-9p1.json

uv run --extra cu128 --frozen python src/evaluate_bc.py \
  --seconds 9.1 --bc-checkpoint results/bones-overfit/ckpt_500 \
  --record-video results/bc-500-9p1.mp4 \
  --output results/bc-500-9p1.json
```

Use `uv run --extra cu128 --frozen` for device commands. The existing default
config has one train episode and no validation episode. BC checkpoint folders
contain `adapter.safetensors` and `bc_config.json`; the base Ψ₀ artifacts are
still required to load an adapter. Training output directories must be new.

## What to deliver

Finish the experiment as far as the machine and artifacts permit. Leave the
repository clean and readable, with any small fixes and useful metrics saved.
Summarize: paired base/adapter body and hand losses, physical base/adapter
comparison to the expert replay, videos/results paths, changes made, focused
checks, and the remaining limitation or blocker. State clearly if the 5/2
phase has not earned a run because one-episode BC still fails. Do the next
diagnostic or repair yourself when the evidence identifies it; do not hand the
user another isolated command after each result.
