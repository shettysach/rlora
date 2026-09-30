# Psi-0 source provenance

`model.py` is ported from `src/psi/models/psi0.py` in
[physical-superintelligence-lab/Psi0](https://github.com/physical-superintelligence-lab/Psi0)
at commit
`4f3720d45e102b36d7c3e9465ab8062274170518`.

The upstream source is licensed under Apache 2.0. A copy is in `LICENSE`.
The local changes keep only the released SONIC checkpoint's inference path,
load weights lazily, store the action head in bfloat16, and skip VLM context
queries whose outputs every block discards. Checkpoint parameter names and
action outputs remain compatible with the upstream implementation.
