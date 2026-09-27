# Psi-0 source provenance

`model.py` is ported from `src/psi/models/psi0.py` in
[physical-superintelligence-lab/Psi0](https://github.com/physical-superintelligence-lab/Psi0)
at commit
`4f3720d45e102b36d7c3e9465ab8062274170518`.

The upstream source is licensed under Apache 2.0. A copy is in `LICENSE`.
The local changes keep the released SONIC checkpoint's action policy,
load weights lazily, store the action head in bfloat16, and skip VLM context
queries whose outputs every block discards. Checkpoint parameter names and
action outputs remain compatible with the upstream implementation.

BC extends this same model with the uniform-time flow objective from
`src/psi/trainers/sonic.py::forward_and_loss` (also used by `posttrain.py`):
`x_sigma = (1 - sigma) * action + sigma * noise`, target `noise - action`,
masked squared error summed over time and averaged over batch and all 80
checkpoint action dimensions. State-token dropout follows the saved model
configuration. Image, state-noise, and temporal-jitter augmentations are
disabled for this first experiment.

`src/data/sonic_bones.py` adapts the joint and Dex3 ordering from
`scripts/data/raw_sonic_to_psi_lerobot.py` at the same commit. It keeps hand
targets from `action.wbc`. Unlike that converter's one-for-one frame copy
with hard-coded 30 Hz timestamps, the local reader uses nearest frames to
resample the 50 Hz BONES recordings to the configured planner clock. It
repeats the last action at an episode boundary, as LeRobot does.

The local LoRA implementation adapts only the action expert's used attention
projections (rank 16 by default, alpha equal to rank). All original Ψ₀
parameters and the VLM stay frozen, as required by `AIM.md`. These local
adapter files are loaded after the strict upstream base checkpoint load;
they are not upstream full-model checkpoints.

`tests/fixtures/psi0_action_expert.json` contains a CPU float32 forward result
generated from the actual upstream action expert at this commit, with small
dimensions and deterministic weights. It covers the entire action expert,
including mixed-length attention masks and checkpoint key names. It does
not establish equivalence of the full released VLM or GPU execution.
