# Milestone 1 — Batched Ψ₀ Inference

## Goal

Build the first version of the new batched robotics runtime:

> **One Ψ₀ model performs batched inference for `n` parallel MJLab environments, with SONIC executing the generated motions.**

For this milestone, a simple task such as **"walk forward"** is sufficient.

The purpose is to validate the inference and simulation architecture before adding LoRA, RL, or dataset generation.

---

## Target architecture

```text
              command: "walk forward"
                       ↓
        observations from n environments
                       ↓
                 Ψ₀ planner
              batched inference
                       ↓
             n action chunks
                       ↓
                SONIC policy
             batched tracking
                       ↓
              n × joint actions
                       ↓
                 MJLab / MuJoCo
                  n environments
                       ↓
                    repeat
```

Everything in the main control path should run in the **same process** where practical.

---

## Project structure

This is a **new project**, not a continuation of the existing `dsrf` architecture.


Suggested initial structure:

```text
src/
├── planner/
│   └── psi0.py
├── tracker/
│   └── sonic/
│       ├── observations.py
│       ├── model.py
│       └── policy.py
├── sim/
│   ├── config.py
│   └── env.py
├── shared/
│   └── g1.py
└── runtime.py

third_party/
└── Psi0/

tests/
AIM.md
MILESTONE_1.md
```

Keep the codebase minimal. Do not port unrelated `dsrf` functionality.

---

## Existing code to reuse

### From `dsrf`

`dsrf` repo is at ../shared/dsrf/. Analyze what is needed. 
You can use the `cp` command and then modify, rather than writing it all yourself.

Reuse only the pieces that are useful for the new runtime:

* G1 joint ordering and default positions from `src/shared/g1.py`
* SONIC observation-layout handling from `src/tracker/sonic/observations.py`
* SONIC ONNX/CUDA inference utilities where useful
* relevant SONIC policy logic from `src/tracker/sonic/tracker.py`
* MJLab G1 environment configuration from `src/sim/config.py`
* basic MJLab environment wrapper ideas from `src/sim/env.py`

Do **not** copy the existing `dsrf` runtime wholesale.

The old runtime is designed around:

```text
motion generator → qpos trajectory → SONIC → simulator
```

and contains assumptions from the older ARDY/VLM architecture.

### Do not port

* ARDY
* kinematic planner
* old VLM agent
* Dora nodes
* grounding
* virtual forces
* Sokoban logic
* task-specific heuristics
* dataset recording
* RL code

---

## Ψ₀

Use the official Ψ₀ repository as an external dependency, preferably under:

```text
third_party/Psi0/
```

Do not manually copy the Ψ₀ model implementation into this repository.

Relevant upstream modules include:

```text
src/psi/models/psi0.py
src/psi/config/transform_psi0_sonic.py
src/psi/deploy/serve_psi0_simple_multi.py
src/psi/deploy/serve_psi0_sonic.py
src/psi/deploy/rtc_inference.py
```

`serve_psi0_simple_multi.py` is especially useful as a reference because it already performs real batched inference across multiple clients.

For this project, however, prefer calling the model **directly in Python** rather than using HTTP once the basic checkpoint works.

---

## Ψ₀ wrapper

Create a small local interface such as:

```python
class Psi0Planner:
    def predict(
        self,
        images,
        states,
        instructions,
    ):
        ...
```

Expected conceptual contract:

```text
images        → batch of n observations
states        → [n, ...]
instructions  → list[str] of length n

output        → [n, T, action_dim]
```

For Milestone 1:

```python
instructions = ["walk forward"] * n
```

All checkpoint loading, preprocessing, normalization, and Ψ₀-specific details should remain behind this wrapper.

---

## Ψ₀ → SONIC interface

The current `dsrf` SONIC path is:

```text
qpos trajectory
      ↓
SONIC reference encoder
      ↓
64-D body token
      ↓
SONIC policy
      ↓
joint actions
```

The SONIC-compatible Ψ₀ checkpoint already predicts a SONIC action representation containing the **64-D body token**.

Therefore the new runtime should not unnecessarily convert:

```text
Ψ₀ token → qpos → SONIC encoder → token
```

Instead use:

```text
Ψ₀
 ↓
SONIC body token
 ↓
SONIC policy
 ↓
joint action
```

Refactor the SONIC code accordingly.

Conceptually separate:

```text
SonicReferenceEncoder
    motion trajectory → SONIC token

SonicPolicy
    SONIC token + robot state/history → joint action
```

Milestone 1 primarily needs `SonicPolicy`.

Any remaining hand/neck dimensions required by the selected Ψ₀ checkpoint should be preserved or given appropriate default handling, but humanoid walking is the only behavior required for this milestone.

---

## Batched MJLab environment

The environment must be designed for batching from the beginning.

Replace the existing single-environment assumption:

```python
SceneCfg(num_envs=1)
```

with configurable:

```python
SceneCfg(num_envs=n)
```

Environment state should remain batched:

```text
root position       [n, 3]
root quaternion     [n, 4]
joint position      [n, 29]
joint velocity      [n, 29]
actions             [n, 29]
```

Avoid indexing environment `0` inside the core API.

Bad:

```python
data.joint_pos[0]
```

Desired:

```python
data.joint_pos
```

---

## Batched SONIC

SONIC must also preserve the environment batch dimension.

Each environment shares the same SONIC weights, but has independent runtime state.

Examples:

```text
last action         [n, 29]
joint history       [n, 10, 29]
velocity history    [n, 10, 29]
gravity history     [n, 10, 3]
SONIC token         [n, 64]
output action       [n, 29]
```

Do not instantiate `n` copies of the SONIC model if a single batched model invocation can be used.

---

## Runtime

The first runtime should be intentionally simple.

Conceptually:

```python
env = MjlabEnv(num_envs=n)
planner = Psi0Planner(...)
sonic = SonicPolicy(batch_size=n)

while running:
    observation = env.observe()

    chunks = planner.predict(
        observation.images,
        observation.state,k
        ["walk forward"] * n,
    )

    for action_step in chunk:
        robot_state = env.robot_state()
        joint_actions = sonic.act(action_step, robot_state)
        env.step(joint_actions)
```

The exact chunk/replanning schedule should follow the selected Ψ₀ SONIC checkpoint and upstream inference behavior rather than inventing a new scheme prematurely.

---

## Implementation order

### 1. Minimal project

Create the new repository and basic package structure.

Install MJLab, SONIC dependencies, and Ψ₀ dependencies.

---

### 2. MJLab + SONIC, `n = 1`

Port the minimal G1 + SONIC code from `dsrf`.

Verify:

```text
known SONIC input/reference
        ↓
SONIC
        ↓
MJLab G1
```

works for one environment.

---

### 3. Standalone Ψ₀ inference

Load the official SONIC-compatible Ψ₀ checkpoint using the upstream code.

Verify one inference call:

```text
image + robot state + "walk forward"
                 ↓
                Ψ₀
                 ↓
            action chunk
```

Check tensor shapes and action layout before integrating with MJLab.

---

### 4. Connect Ψ₀ → SONIC

Create `Psi0Planner`.

Feed the Ψ₀ body-token output directly into the SONIC policy path.

Target:

```text
Ψ₀ → SONIC → MJLab
```

with:

```text
n = 1
```

The G1 should execute a reasonable walking behavior.

---

### 5. Vectorize MJLab

Add configurable `num_envs`.

Remove single-environment `[0]` indexing.

Verify the simulator accepts:

```text
[n, 29]
```

joint actions.

---

### 6. Vectorize SONIC

Make SONIC observations, history, body tokens, and actions batched.

Verify independent state is maintained for each environment.

---

### 7. Batched Ψ₀ inference

Perform **one Ψ₀ forward/inference operation containing multiple environments**.

Do not loop:

```python
for env in envs:
    psi0.predict(...)
```

The desired behavior is:

```python
psi0.predict(batch_of_n_environments)
```

---

### 8. Scale `n`

Test progressively:

```text
n = 1
n = 2
n = 4
n = 8
...
```

Increase until GPU memory or throughput becomes limiting.

---

## Measurements

For each `n`, record at least:

* Ψ₀ inference latency
* effective Ψ₀ throughput
* SONIC inference latency
* overall simulation/control throughput
* GPU memory usage
* whether all environments execute independently and correctly

This milestone is mainly an architecture and scaling experiment.

---

## Success criteria

Milestone 1 is complete when:

* one Ψ₀ checkpoint is loaded once
* one SONIC policy is loaded once
* MJLab contains `n > 1` parallel G1 environments
* observations from multiple environments are passed through Ψ₀ as a real batch
* each environment receives the correct corresponding Ψ₀ output
* SONIC maintains independent history/state for each environment
* SONIC produces batched joint actions
* MJLab executes those actions in parallel
* a simple command such as **"walk forward"** works across the environments
* basic latency, throughput, and VRAM scaling with `n` are measured

---

## Out of scope

Do **not** implement the following during Milestone 1:

* LoRA
* LoRA serving
* reinforcement learning
* PPO / GRPO
* rewards
* dataset generation or recording
* task randomization beyond what is necessary for testing
* complex manipulation
* Sokoban
* external LLM reasoning
* Dora
* multi-task policies
* multiple LoRA adapters

---

## Dora

Do not use Dora inside Milestone 1.

Keep:

```text
Ψ₀ → SONIC → MJLab
```

as one tightly coupled runtime.
Espec
Later, the entire batched runtime can be wrapped as a single Dora worker:

```text
External LLM
     ↓
   Dora
     ↓
┌────────────────────────────┐
│ Batched robotics runtime   │
│ Ψ₀ → SONIC → n × MJLab     │
└────────────────────────────┘
     ↓
   Dora
     ↓
results / task status
```

Dora should eventually provide the **outer orchestration boundary**, not IPC between Ψ₀, SONIC, and MJLab.

---

## Final deliverable

A minimal executable demonstration of:

```text
           one Ψ₀
              │
       batched inference
              │
      ┌───────┼───────┐
      ▼       ▼       ▼
   env 0    env 1   ... env n
      │       │           │
      └────── SONIC ──────┘
              │
          n × G1
              │
            MJLab
```

with all environments executing a simple humanoid walking command on a **single RTX 5090**.
