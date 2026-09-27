# Ψ₀ action expert reference

`psi0_action_expert.json` was generated from the actual classes in
`src/psi/models/psi0.py` at upstream commit
`4f3720d45e102b36d7c3e9465ab8062274170518`, using the project's pinned Torch
and Diffusers on CPU in float32. Only the required class definitions were
executed, so generating it did not import the upstream training package.

The model uses action dimension 80, state dimension 45, horizon 3, hidden
dimension 32, view feature dimension 16, pooled dimension 8, two blocks,
four heads, RMS query/key normalization, layerwise fusion, and a learned
state action token and null token. Dropout is disabled.

Every state-dict tensor is generated independently as `randn(shape) * 0.1`.
Its generator seed is the first four bytes of SHA256 of its parameter name,
interpreted as an unsigned little-endian integer. Inputs use a separate Torch
generator with seed 123, in the order actions, views, states, pooled text.
Timesteps are `[333, 777]`; masks contain five and three valid context tokens.
The fixture records all inputs, weight names/shapes, and the upstream output.

The regression test reconstructs these weights and inputs, loads every key
strictly, and compares the local action expert with the saved upstream output
at `rtol=1e-5`, `atol=1e-6`. It requires no upstream checkout or downloaded
model weights. Released-checkpoint, VLM preprocessing, and CUDA comparisons
still require Phase 0 on the training machine.
