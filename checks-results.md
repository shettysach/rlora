# GPU checkpoint compatibility results

Measured 2026-10-04 on repository commit `39bc3b450ddc57a1afcf39adba9f97dbd8645589`.
The initial checkout had a staged deletion and an untracked replacement of
`MUJOCO_LOG.TXT`; both were preserved. The changes reported below are uncommitted.

After two small compatibility fixes, all tested released-checkpoint SDPA paths
match the pinned upstream reference exactly at batches 1, 2, and 4. No numerical
tolerance was accepted: maximum absolute and FP32 RMS error are both zero.
This establishes numerical compatibility for the generated inputs and locked
environment, not universal hardware, backend, or closed-loop simulation parity.

## Environment and commands

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB; driver 595.91.07.
- PyTorch 2.11.0+cu128, CUDA runtime 12.8; CUDA available.
- Transformers 4.57.1; Diffusers 0.37.0.
- Qwen text and vision implementations were both verified as `sdpa`.
- FlashAttention 2 availability was false. FA2 and SDPA-versus-FA2 comparisons
  remain untested; the lockfile was not changed.

```sh
uv sync --extra cu128 --frozen
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
.venv/bin/python -m pytest -q -rs tests/test_psi0_conditioning.py
# 2 passed; CUDA parameter ran, no skips.
git clone https://github.com/physical-superintelligence-lab/Psi0.git /tmp/rlora-psi0-reference
git -C /tmp/rlora-psi0-reference checkout --detach 2830f9367fc12ca49d9b4df4348726daae9db15d
OMP_NUM_THREADS=4 .venv/bin/python tools/check_psi0.py
.venv/bin/python -m pytest -q -rs tests/test_psi0_conditioning.py tests/test_psi0_compatibility.py
# 4 passed, no skips.
.venv/bin/ruff check tools/check_psi0.py tests/test_psi0_compatibility.py src/planner/psi0.py src/planner/_psi0/model.py
# All checks passed.
```

GPU commands required execution outside the filesystem sandbox: NVIDIA-SMI
could not communicate with the driver inside it. The initial uv cache access
and git DNS failure were also resolved by execution outside the sandbox.
The requested Hugging Face download commands were attempted, but artifacts
already existed as symlinks into the cache. The Qwen command reported SameFileError;
no replacement was needed. Actual files were read and hashed successfully.

## Reproduction and reference adapter

The permanent harness is [tools/check_psi0.py](tools/check_psi0.py).
It executes upstream model ASTs with class and method bodies unchanged, excluding
only `psi` configuration/util imports and the logging initialization. It supplies
an unused `LaunchConfig` annotation and a parameter counter for logging. The model
source is verified byte-for-byte against the pinned Git commit before execution.
For transforms, it extracts unchanged `ResizeImage`, `CenterCrop`, and
`SonicActionStateTransform` definitions; the otherwise behavior-free field base is
provided by Pydantic `BaseModel`, and path resolution is identity for the absent
training stats path. Released min/max arrays come from `run_config.json`.

The reference uses upstream `Psi0Model.predict_action`, including per-row
preprocessing, right-padding, observation projection, six action blocks, and
plain flow sampling. RTC and previous-action conditioning are absent from this
method. The reference causal Qwen wrapper retains its checkpoint final norm and
captures `hidden_states[-1]`; the local backbone bypasses the norm. Both use the
same fine-tuned checkpoint VLM weights (a separate copy), never base Qwen weights.
Action weights are identical, BF16, with fixed time frequencies retained in their
checkpoint FP32 precision. Both models run under CUDA BF16 autocast. Reference
rope state is cleared for every call, and schedulers reset their timesteps on
every prediction. Models alternate between CPU and CUDA to limit VRAM use.

NumPy seed is 292285. RGB pixels are deterministic uniform uint8 noise;
raw images have sizes `(360+16*i, 480+32*i)` per row. States are valid 43D FP32
samples between the published bounds. Prompts are `walk forward` followed by
` and carry the box carefully` repeated by row index. Mixed-grid cases append
` stop` and use raw unequal image sizes; deployed images resize/crop to 240×320.
Noise seeds are `292285 + batch`; one initial FP32 noise tensor `[B,30,78]` is
saved and cloned into both actual sampling methods through a scoped randn adapter.
Ten flow steps use 1,000 training timesteps and horizon 30.

Captured inputs, features, every block output, every velocity/action update,
and initial noises are saved to CPU under ignored `artifacts/psi0-checks/`.
Each `.pt` case contains reference and local tensors. Incremental `metrics.json`
and `metrics-before.json` retain measured results.

## Findings and fixes

1. The local resize used torchvision's bilinear default; upstream explicitly uses
   nearest-neighbor. On the first generated image, the discrepancy was max 158
   uint8 levels, RMS 55.6521454. `Psi0Planner` now selects NEAREST explicitly.
   All seven image rows match exactly after the change. A focused pixel-selection
   regression covers nontrivial downsampling.
2. The local scheduler created sigmas on CUDA, whereas upstream retains CPU scalar
   sigmas. With identical VLM features, observation projection, all six block
   outputs, and velocity at step zero, the first scheduler action update diverged:
   max 0.015625, RMS 0.001032219734 (BF16). CUDA FP32 sigma scalars change promotion
   and multiplication rounding versus CPU wrapped scalars in the locked PyTorch.
   Local sampling now leaves scheduler tensors on CPU and uploads expanded
   timesteps to CUDA, matching upstream. Before the fix, final normalized max
   error ranged from 0.0078125 to 0.01171875 over nine cases. Afterward all steps
   and final outputs match exactly. A focused CUDA sampling regression retains
   independent upstream-style CPU-sigma stepping.

Production final-block padding was preserved. Removing it only in the harness
changes eight CUDA cases: four FP32 cases (maximum 1.937150955e-7) and four BF16
cases (maximum 0.0009765625). These are diagnostic differences, not accepted
compatibility errors or a reason to remove padding.

## Coverage and measured results

[checks-after.csv](checks-after.csv) contains all 3,592 per-comparison results:
name, shape, local dtype, backend, finiteness of both operands, exact value
equality, max absolute error, and RMS error computed in FP32. Every value is
finite. All 3,584 production/reference-repeatability comparisons are exact.
The eight non-exact rows are deliberately unpadded attention diagnostics.
[checks-before.csv](checks-before.csv) records the original 3,115 comparisons.

| Comparison | Cases | Result |
|---|---:|---|
| CUDA attention: FP32/BF16 × B=1/2/4 × D=64/768 × updating/final | 24 | Exact action output; exact context for updating blocks |
| Reference attention repeatability | 24 | Exact |
| Released conditioning with identical processor outputs | B=1/2/4 | Exact features and sampling |
| Complete deployed preprocessing and conditioning | B=1/2/4 | Exact images, normalized states, metadata, positions, converted pixels, features |
| Same prompts/grids, inverted RGB pixels | B=1/2/4 | Exact reference match; features change on every batch |
| Changed prompts/padding and unequal image grids | B=1/2/4 | Exact metadata, positions, features, row order |
| Observation projection, time/action embeddings, six block action outputs, five updated contexts | All 9 checkpoint input cases, 10 steps | Exact |
| Predicted velocity and scheduler action updates | All 9 cases, every step | Exact `[B,30,78]` |
| Final normalized and denormalized actions | All 9 cases | Exact FP32 `[B,30,78]` |
| Reference full-prediction repeatability | All 9 cases, every captured intermediate | Exact |

Masks have identical values; upstream observation masks are FP32 and local masks
are boolean by design. Upstream VLM pixels are supplied in FP32 and local pixels
in BF16; comparisons use their identical BF16 vision input conversion. Features,
action embeddings, block outputs and velocities are BF16; initial noise is FP32,
scheduler outputs are BF16, and final public normalized/denormalized actions are
FP32. Metadata IDs/grids/positions are int64. The local projection is computed
once; its output matches the first reference projection, which upstream repeats
unchanged each step in eval mode. Final unused context output is excluded.

All batch sizes fit; no capacity/OOM result occurred. No readily available
recorded RGB observation was added. FA2 is untested. Synthetic permanent tests
also cover independent changes to token IDs, padding masks, and grids and
metadata refresh; the released-checkpoint harness changes these together through
real preprocessing. No simulator, SONIC, HSSD, or motion-quality claim is made.

## Artifacts

Resolved cached revisions:

- Psi model: `cb1480f50d9764ab2f6261bc2e651a67ad54da17`.
- Qwen config/processor: `89644892e4d85e24eaac8bacfd4f463576704203`.
- Upstream source: `2830f9367fc12ca49d9b4df4348726daae9db15d`.

SHA-256 values (also saved in `artifacts/psi0-checks/hashes.json`):

| Artifact | SHA-256 |
|---|---|
| `run_config.json` | `075149ab6417a5a3bfab20d7b7d481a2c43eef7a4d908c7632bf52eac245e02e` |
| `model.safetensors` | `b60c11f84887fc8b6307851c23dc9169cef4c7e292fcaaed706d6254952bc711` |
| `tokenizer.json` | `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7` |
| `vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `preprocessor_config.json` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `generation_config.json` | `1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4` |
| `chat_template.json` | `6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4` |
| `tokenizer_config.json` | `c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399` |
| `video_preprocessor_config.json` | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| `config.json` | `bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b` |
| `merges.txt` | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
