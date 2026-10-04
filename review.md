My view: **the project is well structured for a research prototype, and its batching architecture is sensible. Readability is good; actual performance is still unproven.** The clearest scaling limitation is the image acquisition path.

I reviewed the current `carry-box` working tree, the project guidance and milestone documents, and the corresponding runtime, simulation, controller, and inference paths in `dsrf`.

**For understandability, the structure works well.**

- **The execution path is easy to follow.** [runtime.py](/home/sword/Desktop/USC/lab/rlora/src/runtime.py:77) directly shows observation, planning, action execution, and termination. Compared with `dsrf`, understanding a rollout requires following fewer components and state transitions.
- **The component boundaries have clear purposes.** `Psi0Planner` handles preprocessing and inference; `SonicPolicy` handles observation packing and history; `SonicModel` handles ONNX execution; simulation configuration is separate from execution. These are useful divisions with little wrapper overhead.
- **Scope is controlled.** The code avoids importing `dsrf`’s grounding, messaging, virtual forces, and backend selection machinery. That fits the repository’s instructions. `dsrf` supports more behaviors, so its additional complexity serves a different scope.
- **Provenance is documented well.** The isolated Ψ₀ implementation, pinned dependencies, licenses, and compatibility notes make the research code easier to reproduce and maintain.

The main readability weakness is **implicit tensor and execution contracts**. A `torch.Tensor` annotation does not explain shape, joint order, units, device, or stream requirements. Those details matter here: planner state is 43D, actions are 78D, body joints have several orderings, and simulation operations depend on the correct CUDA context. Small docstrings at the public boundaries would help substantially. [RobotState](/home/sword/Desktop/USC/lab/rlora/src/shared/state.py:6) is one obvious place.

Two smaller issues deserve attention:

- In [carry_box.py](/home/sword/Desktop/USC/lab/rlora/src/carry_box.py:66), `reward` actually accumulates placement **time in seconds**, and `SUCCESS_REWARD` is a duration threshold. Naming those quantities by their meaning would make the success rule clearer.
- The Ψ₀ port contains subtle compatibility changes, including bypassing the final language normalization and padding attention output to preserve BF16 rounding. These are explained, but they make this the most delicate code to modify. A focused numerical comparison against the checkpoint’s reference inference would provide more confidence than additional general unit tests. [Model implementation](/home/sword/Desktop/USC/lab/rlora/src/planner/_psi0/model.py:329).

**For performance, several important decisions are already good.**

There is one planner and one SONIC decoder serving the whole batch. The planner’s body tokens go directly to SONIC, avoiding the old trajectory encoding path. SONIC uses persistent input/output buffers and CUDA I/O binding, shares the simulation stream, and reuses a history scratch buffer. Replay actions are uploaded once. These choices avoid duplicated models and repeated observation/action transfers. [SONIC execution](/home/sword/Desktop/USC/lab/rlora/src/controller/sonic/model.py:43).

The planner also avoids unnecessary language logits, retained layer outputs, and KV caching. It uses BF16 and computes the initial context projection once per prediction, outside the denoising loop. These are targeted optimizations that preserve a fairly readable implementation. [Planner inference](/home/sword/Desktop/USC/lab/rlora/src/planner/_psi0/model.py:450).

The main limitations are:

| Area | What I found | Performance implication |
|---|---|---|
| Image capture | Downloads batched poses, then renders each world sequentially | Rendering introduces serial work as `n` grows |
| Image preprocessing | PIL conversion, resize, and crop happen per image on the CPU | Adds work before the batched model invocation |
| Simulator stepping | MJLab calls CUDA `nonzero()` each control step | Host synchronization remains despite chunk-level termination checks |
| Viewer | Synchronizes after every action and downloads state for every world | Viewer runs can substantially distort throughput measurements |
| Episode completion | Finished environments continue executing until the batch exits | Unequal episode lengths reduce useful batch throughput |

The image path in [env.py](/home/sword/Desktop/USC/lab/rlora/src/sim/env.py:71) is the strongest visible scaling concern. Whether it dominates latency requires measurement; the model could still dominate at small batch sizes. [PERF.md](/home/sword/Desktop/USC/lab/rlora/PERF.md) accurately inventories transfers, but it is not a benchmark.

I would **preserve the architecture**, document the tensor/stream contracts and duration units, then measure warmed-up runs at `n = 1, 2, 4` with the viewer disabled. Record rendering/preprocessing time, planner and control latency, total environment steps per second, and peak VRAM before choosing further optimizations.

The existing CPU batch/history test passed. CUDA is unavailable in this workspace, so I cannot establish GPU throughput or numerical checkpoint parity from this review.
