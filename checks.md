# GPU checkpoint compatibility checks

## Task for the agent

Verify the local Ψ₀ inference port against the released checkpoint's reference
inference on the GPU. Read `AGENTS.md`, `AIM.md`, `MILESTONE_1.md`, and
`src/planner/_psi0/UPSTREAM.md` first. Keep this work focused on numerical
compatibility. Preserve existing changes in the checkout.

Create a reproducible comparison script and write the measured results to
`checks-results.md`. Fix any demonstrated compatibility bug with a small change
and rerun the affected comparison. Do not change inference behavior merely to
make a tolerance pass.

## What has already been checked

CPU component comparisons with synthetic weights passed:

- The language normalization bypass matches the unmodified Qwen causal wrapper's
  recorded final features exactly in FP32 and BF16.
- The local joint attention processor matches the pinned upstream processor in
  24 cases: FP32/BF16, batches 1/2/4, dimensions 64/768, and updating/final blocks.
- Removing final-block output padding changes some CPU results, including BF16
  differences up to `0.0009765625`.

These results do **not** establish full released-checkpoint or CUDA parity.
The CPU comparison script was temporary and is not shipped with this checkout.
The permanent conditioning test now compares an independent, unmodified Qwen
wrapper, including mixed image grids, right padding, and metadata cache changes.

## Environment and artifacts

Run from the repository root. Use the locked environment:

```sh
uv sync --extra cu128 --frozen
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
.venv/bin/python -m pytest -q -rs tests/test_psi0_conditioning.py
```

Confirm the CUDA test actually runs. Record repository commit and dirty status,
GPU model, driver, PyTorch, Transformers, Diffusers, and attention backend.

Use these artifacts, downloading missing files with the commands in `README.md`:

- Run directory:
  `artifacts/psi-model/psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223`
- Its `run_config.json` and `checkpoints/ckpt_40000/model.safetensors`.
- Qwen tokenizer, processor, and configuration: `artifacts/qwen3-vl-2b`.
  The Ψ₀ checkpoint supplies the fine-tuned VLM weights; do not substitute the
  base Qwen weights for either side of the comparison.

Record artifact hashes and resolved download revisions. Checkpoint checks do
not require SONIC, HSSD, or a running simulator. Deterministic generated RGB
images and valid 43D states are sufficient; add a recorded observation if one
is readily available.

Obtain upstream Ψ₀ in a separate directory at commit
`2830f9367fc12ca49d9b4df4348726daae9db15d`:

```sh
git clone https://github.com/physical-superintelligence-lab/Psi0.git /tmp/rlora-psi0-reference
git -C /tmp/rlora-psi0-reference checkout --detach 2830f9367fc12ca49d9b4df4348726daae9db15d
```

If that directory already exists, inspect it before reusing it. Use the actual
upstream inference implementation as the reference. Inspect its model,
SONIC transform, and serving code to reproduce the released run configuration.
Document any adapter needed to call it under the locked dependencies. Do not
replace its computations with local implementations. A component comparison
may extract an unchanged upstream class and its required definitions.

## Comparisons to run

### 1. CUDA attention processor

Reproduce the CPU processor matrix above on CUDA with identical weights and
inputs, using 30 action tokens and 65 context tokens, including masked padding.
Compare action output for both kinds of block; compare context output only for
blocks that update it. Upstream receives the context mask, whereas the local
processor receives a joint mask with valid action tokens prepended.

Include a diagnostic comparison with local final-block padding removed in the
comparison script only. Report whether it changes CUDA results. Keep the
production padding while establishing parity.

### 2. Released VLM conditioning

Load identical checkpoint VLM weights for both paths. The reference must keep
the final language norm intact and obtain the features used by upstream from
the causal wrapper with `output_hidden_states=True`. The local path bypasses
that norm and uses the backbone's `last_hidden_state`.

First feed identical processor outputs to isolate model execution. Then compare
the complete local preprocessing and conditioning path against upstream:

- Batch sizes 1, 2, and 4, with different images and prompt lengths per row.
- Right padding and unequal image grids where preprocessing supports them.
- Repeated prompts/grids with changed pixels: metadata may be reused, but
  features must reflect the new pixels.
- Changed tokens, padding masks, and grids: metadata must refresh.

Compare token IDs, masks, grids, multimodal positions, pixel values after the
vision dtype conversion, and resulting VLM features. Check row order. Reset
reference KV/rope state as required so previous calls do not contaminate results.

### 3. Action head and full sampling

Use the released six-block action head, 78D actions, 43D normalized states, and
the horizon from `run_config.json`. Preserve the checkpoint precision of fixed
timestep frequencies; matrix weights and inference autocast are BF16.

Compare observation projection, each block's action output, each updated
context, and the final predicted velocity for identical context, state, noisy
actions, timestep, and masks. The local final block omits unused context queries;
its context output need not match the reference because it is never consumed.

Then compare complete **plain flow sampling**, using the same scheduler settings
and 10 inference steps. Disable RTC guidance and previous-action conditioning
on the reference. Save one initial noise tensor and feed clones to both paths;
matching seeds alone is insufficient if their random draw order differs.
Reset scheduler state for every prediction. Capture velocity and action tensors
at every denoising step, final normalized actions, and denormalized 78D actions.
Use the upstream normalization and image transforms for the reference.

Start with batch 1, then 2 and 4. Load models sequentially and save intermediate
tensors to CPU if both copies do not fit in VRAM. An OOM at a larger batch is a
capacity result, not a numerical failure.

### 4. Attention backends

Run SDPA first. If FlashAttention 2 is installed and supported, repeat the
checkpoint comparisons with FA2 for the Qwen VLM. Force the same backend on
both reference and local paths, including text and vision configuration, and
verify the selected implementation. The local loader selects FA2 automatically
when available; use the comparison harness to control that selection.

Compare local versus reference **within each backend**. Report SDPA versus FA2
differences separately. If FA2 is unavailable, mark it untested; do not silently
claim coverage from an SDPA fallback or change the lockfile just to install it.

## How to judge and report results

- First measure repeatability of the reference on identical inputs.
- Attempt exact equality for comparisons whose operations and layouts match.
- For every comparison report shape, dtype, backend, finite values, exact
  equality, maximum absolute error, and RMS error (compute errors in FP32).
- If equality fails, find the first divergent intermediate. Investigate
  preprocessing, masking, norm capture, tensor layout, frequency precision,
  autocast, and scheduler state before choosing a tolerance.
- Report any numerical tolerance explicitly with its evidence. Do not accept
  arbitrary BF16 tolerances or use relative error near zero as the only metric.
- Include commands, script path, inputs/seeds, artifact hashes, per-case results,
  changes made, and remaining untested cases in `checks-results.md`.

Distinguish component parity, complete checkpoint parity, and backend coverage.
A successful inference call or plausible robot motion alone does not establish
numerical compatibility. Keep a focused regression test if a bug is found;
avoid adding a broad test suite or runtime compatibility checks for this task.
