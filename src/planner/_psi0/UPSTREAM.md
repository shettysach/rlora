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
