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
