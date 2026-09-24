# Project intent

- Efficient batched dataset generation using LoRA, RL, and simulation.
- A mid-level planner is responsible only for producing the required humanoid motion.
- Long-context reasoning, task decomposition, and high-level planning are handled externally by an LLM.
- The planner converts the resulting command into short-horizon motion commands.
- The generated motion is tracked and executed in simulation by SONIC.
- We run `n` simulation environments in parallel with different seeds / initial conditions.
- All environments use the same base planner and task-specific LoRA adapter.
- Experience from the `n` environments is collected into a rollout batch.
- The LoRA adapter is updated using RL after each rollout batch / training iteration.
- The base planner remains frozen.

## Specifics

- Planner: Ψ₀ (Psi-0) with SONIC-compatible checkpoint Tracker: SONIC (GEAR-SONIC)
- Simulation: MJLab / MuJoCo
- Hardware: 1× RTX 5090
- LoRA rank: 16 or 32
- Base model: Frozen
- Trainable parameters: LoRA adapter only
- Parallelism: `n` randomized simulation environments
- RL target: Improve the planner's motion generation based on task success and execution quality

## Dataset

Each rollout will record the information required for RL training and evaluation.
This will include, at minimum:

- environment observations
- planner outputs / generated motion commands
- executed robot states or actions
- task rewards
- success / failure and termination information

Additional metadata may also be stored where useful, such as:

- initial object and robot states
- environment randomization values
- task-specific parameters
- generated reference motions

The exact dataset schema is not fixed yet and will depend on the RL algorithm and the information needed for later analysis or reuse.

## Training iteration

A single training iteration is approximately:

1. Freeze the current `Base + LoRA` policy.
2. Run inference over `n` environments.
3. Collect one or more rollouts from each environment
4. Compute task rewards.
4. Form an RL training batch from the collected trajectories.
6. Update the LoRA parameters.
7. Deploy the updated LoRA for the next rollout batch.
