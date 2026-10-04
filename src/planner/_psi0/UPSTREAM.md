# Psi-0 source provenance

`model.py` is ported from `src/psi/models/psi0.py` in
[physical-superintelligence-lab/Psi0](https://github.com/physical-superintelligence-lab/Psi0)
at commit
`2830f9367fc12ca49d9b4df4348726daae9db15d`.

The upstream source is licensed under Apache 2.0. A copy is in `LICENSE`.
The local changes retain the released carry-box checkpoint's inference path:
the 43D state joins the VLM context, five blocks update that context, and the
sixth uses it for action output. The full 78D action is decoded locally.
Action-head matrix weights use bfloat16, while fixed timestep frequencies retain
their checkpoint precision. VLM conditioning reads the backbone's final features
without computing language logits, collecting layer outputs, or retaining a KV cache.
The unused final language norm is bypassed to preserve the causal wrapper's
last decoder features in the pinned Transformers version.
Sampling uses plain flow inference without RTC guidance or previous-action state.
Fixed context projection and the joint attention mask are computed once per
prediction and reused across denoising steps. Checkpoint parameter names are unchanged.
Image and text preprocessing use one processor call for the environment batch,
with the original right padding and image order preserved.
Multimodal position IDs are computed from CPU token/grid metadata before upload.
SDPA keeps the image grid on CPU for shape/split operations; FA2 uploads it for
CUDA sequence lengths. Pixel patches are cast to the vision model's BF16 dtype
before upload, preserving Qwen's input conversion with half the upload payload.
The last token/mask/grid batch caches device metadata while its CPU values stay
unchanged. Attention masks are converted to boolean before upload. Changing
tokens, padding, or grids refreshes the cache; image pixels are always updated.
The final block skips unused context queries. Padding the attention output keeps
the action projection's original memory layout to preserve BF16 rounding.

## CPU numerical checks

A small Qwen3-VL model with synthetic weights matches the unmodified causal
wrapper's recorded final features exactly after bypassing the language norm,
in FP32 and BF16. The conditioning regression test now uses a separate,
unmodified wrapper as its reference, including mixed image grids and padding.

A local comparison against the original attention processor from the pinned
source also matches exactly in 24 cases: FP32/BF16, batches of 1/2/4,
dimensions 64/768, and updating/final blocks. Each pair uses identical synthetic
weights and inputs, with 30 action tokens and 65 context tokens. Omitting the
padding changes four BF16 batch cases by up to `0.0009765625`, and two FP32
cases by up to `2.98e-8`. The cached reference source was checked against the
commit's Git blob hash before execution.

These are component comparisons, not released-checkpoint inference validation.
Full checkpoint weights are unavailable locally; CUDA attention execution and
its numerical behavior remain untested here.
