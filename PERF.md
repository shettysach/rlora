# CPU transfers

Inventory of the current runtime, including native MuJoCo RGB rendering.
This records existing behavior; optimization is deferred.
The default HSSD room adds visual mesh/texture assets at startup and a second
mocap root (room plus table); the per-RGB transfer formula below includes both.

Assume `--device cuda`. With a CPU device, the tensor operations below do not
cross a CPU/CUDA boundary. This is a source inspection, not a CUDA profile:
driver transfers and execution-provider placement need profiling to establish
their exact cost. Dependency details refer to the versions in `uv.lock`:
MJLab 1.6.0 at `0fb8a681136be94ffc636a3dd423cabb97d91f10`, Transformers
4.57.1, and Diffusers 0.37.0.

## Reading the inventory

- **D2H:** device to host, usually CUDA tensor to CPU memory.
- **H2D:** host to device, usually CPU tensor or Python data to CUDA memory.
- **CPU copy:** additional movement within host memory.
- **Host synchronization:** CPU waits for device work or reads a scalar/shape;
  even a small payload can stall execution.
- `.numpy()` on a suitable CPU tensor shares its storage. The preceding
  `.cpu()` performs the device transfer when the input is on CUDA.
- `torch.from_numpy()` shares CPU storage; it does not upload to CUDA.
- `.to(device)` does not transfer when device and dtype already match.

## Repeated runtime transfers

| Location | Direction / operation | Payload | Frequency and purpose |
| --- | --- | --- | --- |
| `src/sim/env.py`, `MjlabEnv.rgb`: `data.qpos.cpu().numpy()` | D2H | Entire batch of generalized positions, `[n, nq]` | Once per RGB request, before each policy prediction. Native MuJoCo needs host state. |
| Same function: `data.mocap_pos.cpu().numpy()` | D2H | `[n, nmocap, 3]` | Once per RGB request. Mocap body positions for the host scene. |
| Same function: `data.mocap_quat.cpu().numpy()` | D2H | `[n, nmocap, 4]` | Once per RGB request. Mocap body orientations for the host scene. |
| Same function: `env_origins.cpu().numpy()` | D2H | `[n, 3]` | Once per RGB request, although origins are fixed. Removes batching offsets before rendering. |
| Same function: assignments to host `MjData` | CPU copy / dtype conversion | One world's positions and mocap pose | Once per rendered environment. Copies simulator values into MuJoCo's host arrays; float32 simulation values become float64. |
| Same function: `Renderer.update_scene()` and native rendering | Host scene submission to OpenGL | Scene geometry/transforms; graphics resources as required | Per environment. Native renderer consumes the CPU scene; driver upload details depend on the graphics backend. |
| Same function: `Renderer.render()` | OpenGL framebuffer to CPU | `360 × 640 × 3` uint8 RGB per environment | Per environment. MuJoCo calls `mjr_readPixels` to populate a NumPy image. With hardware rendering this is a graphics-device readback; software rendering has different costs. |
| Same function: `.render().copy()` | CPU copy | One RGB image | Per environment. Copies the returned frame. |
| Same function: `np.stack(images)` | CPU copy | All `n` RGB images | Once per RGB request. Allocates the contiguous image batch. `torch.from_numpy` then shares that allocation. |
| `src/planner/psi0.py`, `predict`: `images.cpu().numpy()` | Currently a CPU view; conditional D2H | Image batch | Once per prediction. `rgb()` already returns a CPU tensor, so this currently performs no CUDA download. Would download if supplied CUDA images. |
| Same function: PIL conversion, resize, center crop | CPU image processing / allocations | Each image | Per prediction. Prepares images for the checkpoint's processor. |
| `src/planner/_psi0/model.py`, `_conditioning`: processor output `.to(self.device)` | H2D | Processed image patches, token IDs, attention mask, image-grid metadata | Once per prediction, with multiple tensors uploaded. Processing/tokenization initially produces host tensors. Pixel payload depends on processor resolution and dtype. |
| `src/runtime.py`, chunk completion: `.all().item()` | D2H scalar and host synchronization | One boolean | Once per executed action chunk, in both replay and policy modes. CPU decides whether to stop. |
| `src/runtime.py`, final report: three `.cpu().tolist()` calls | Three D2H transfers, then CPU list conversion | `terminal_steps`, `succeeded`, `fell`, each `[n]` | Once at the end of a rollout. Prints per-episode outcomes. |

There are **four explicit tensor downloads per RGB batch**, followed by
sequential native rendering and one framebuffer readback per environment.
For float32 simulator state, the four downloads contain
`4 × n × (nq + 7 × nmocap + 3)` bytes. RGB alone contains
`691,200 × n` bytes per batch, before the additional CPU image copies and
processor allocations. These are logical payload sizes, not measured bus traffic.

## Transfers inside planner dependencies

These occur during Qwen conditioning or scheduler setup, rather than in the
local wrapper's explicit `.cpu()` calls. Small metadata reads also synchronize.

| Dependency / function | Direction / operation | Frequency and purpose |
| --- | --- | --- |
| Qwen3-VL vision `rot_pos_emb` | Two explicit D2H scalar reads: grid maximum and total token count via `.item()` | Each vision forward. Determines position-table sizes. Grid scalars used by `arange`, `expand`, and the frame-count condition also require host values. |
| Qwen3-VL vision `fast_pos_embed_interpolate` | Implicit grid scalar reads; H2D uploads of interpolation index and weight tables | Each vision forward. Interpolation tables are constructed on CPU and converted to tensors on the embedding device. Its `indices.tolist()` and `weights.tolist()` operate on CPU tensors, so those particular calls are not D2H. Shape arguments in `split`, `repeat`, and `view` also consume grid scalars. |
| Qwen3-VL vision attention, SDPA path | D2H `lengths.tolist()` | Three list conversions per vision attention block, one each for query/key/value splitting. This branch is skipped when Flash Attention 2 is selected. |
| Qwen3-VL `get_image_features` | D2H grid-derived `split_sizes.tolist()` | Each image-conditioning call. Splits the encoded image features. |
| Qwen3-VL `get_rope_index` | D2H filtered token IDs via `.tolist()`; grid `t/h/w` via three `.item()` calls per image; implicit scalar reads for loop counts and conditions | Per batch element during multimodal position construction. Token IDs are inspected in Python. |
| Same `get_rope_index` | H2D position IDs and rotary-position deltas | CPU builds position arrays; uploads positions per batch element and the batch of deltas. |
| Qwen3-VL token masking / `argwhere` | Host synchronization for dynamically sized CUDA results | During conditioning. Boolean indexing and token-location discovery require result-size information; this does not download the complete activation tensor. |
| Diffusers `FlowMatchEulerDiscreteScheduler.set_timesteps` | H2D sigma schedule | Once per prediction. NumPy constructs the schedule; `torch.from_numpy(sigmas).to(..., device=device)` uploads it. Current default path derives timesteps on-device. An explicitly supplied timestep schedule would cause another upload. |

The planner chooses Flash Attention 2 when available, otherwise SDPA. Exact
implicit synchronization counts depend on that choice, image/token shapes,
and library implementation. This table identifies inspected source paths,
not every internal transfer in CUDA kernels or attention wrappers.

The scheduler's constructor `.item()` calls read CPU tensors. Its
`index_for_timestep().item()` is bypassed in the current inference loop by
`set_begin_index(0)`; it is not an active per-denoising-step download here.

## Simulator synchronization

MJLab's `ManagerBasedRlEnv.step()` calls
`reset_buf.nonzero(as_tuple=False)` every control step. CUDA `nonzero` requires
host synchronization to establish the output size, even though the returned
indices stay on the device. This runs in the current configuration, including
steps where no environment terminates.

Other conditional paths in MJLab can read CUDA scalars or download IDs: pending
reset checks, reset/error reporting, reward/metric logging, and NaN diagnostics.
The current task has no configured reward/termination terms in MJLab, and the
NaN guard is disabled by default. These conditional paths should be revisited
when those features are enabled.

## Optional native viewer

`src/sim/viewer.py` delegates to MJLab's `NativeMujocoViewer`. The runtime calls
`viewer.sync()` after every control step when `--viewer` is enabled; scene mode
also calls it repeatedly while idle, approximately every 20 ms.

| MJLab viewer operation | Direction / payload | When |
| --- | --- | --- |
| `_sync_env_state_to_mjdata`: `qpos.cpu().numpy()` | D2H selected world's positions | Each viewer sync. |
| Same function: `qvel.cpu().numpy()` | D2H selected world's velocities | Each viewer sync. |
| Same function: `ctrl.cpu().numpy()` | D2H selected world's actuator controls | Each viewer sync when `nu > 0`. |
| Same function: `mocap_pos.cpu().numpy()` | D2H selected world's mocap positions | Each viewer sync when `nmocap > 0`. |
| Same function: `mocap_quat.cpu().numpy()` | D2H selected world's mocap quaternions | Each viewer sync when `nmocap > 0`. |
| Same function: `xfrc_applied.cpu().numpy()` | D2H selected world's applied body forces | Each viewer sync. |
| `model_sync.sync_model_fields` | D2H per expanded visual model field, followed by assignment into host `MjModel` | When per-world visual fields are expanded, e.g. by randomization or variants. |
| `_render_other_env_geoms` | Repeats the state/model downloads for every other world | Every viewer sync when `num_envs > 1`. The installed native viewer creates a second host `MjData` and loops over all other worlds; this path does not consult `max_extra_envs=0`. |
| `sync_viewer_to_env` perturbation handling | H2D generalized force array | Conditional mouse-force path. Our passive viewer disables perturbations and does not invoke this path. |

For this scene, the six state downloads therefore repeat for every environment
on each viewer sync, not just the selected one. Viewer host-array assignments
add CPU copies/conversions after the downloads.
Display rendering also submits host scene data to OpenGL. Reward plots and
debug visualizers can add transfers if enabled with corresponding task terms.

## Startup and reset transfers

| Location | Direction / payload | When |
| --- | --- | --- |
| MJLab simulation/entity initialization | H2D compiled model arrays, initial state, scene constants and entity defaults | Environment construction. MuJoCo compiles the host model; MuJoCo Warp initializes device simulation arrays. Per-world variants can introduce additional host staging and uploads. |
| `src/sim/env.py`, constructor | H2D three joint-index maps: body, planner, and hand order | Once per environment wrapper. Python lists become device tensors. |
| `src/sim/env.py`, `reset` | H2D base pose `[n,7]`, joint positions `[n,43]`, box pose `[n,7]` | Whenever external reference poses are supplied. Runtime's `torch.from_numpy` calls create CPU views; these three `.to(self.device)` calls upload them. Subsequent `.clone()` calls are device copies. |
| `src/carry_box.py`, constructor | H2D hand-geometry ID list | Once per task. Geometry names/IDs are inspected in the host model. |
| `src/runtime.py`, replay setup | H2D complete reference action batch `[n,max_steps,78]` | Once in replay mode. `torch.as_tensor(..., device=args.device)` uploads the NumPy batch. |
| `src/planner/psi0.py`, constructor | H2D state/action normalization minima and maxima | Once per planner. Four arrays loaded from JSON become device tensors. |
| `src/planner/_psi0/model.py`, `from_pretrained` | H2D VLM parameters/buffers and action-head parameters/buffers | Once per planner. Safetensors are opened on CPU, loaded into host modules, then modules are moved to the device. Fixed timestep frequencies receive a separate upload to preserve precision. |
| `src/controller/sonic/policy.py`, constructor | H2D default joint positions and two joint-order maps | Once per controller. |
| `src/controller/sonic/model.py`, ONNX session creation | H2D weights/constants for CUDA-assigned nodes | Once per controller, managed internally by ONNX Runtime. |

## Synchronization without bulk CPU transfers

- Runtime `wait_stream()` calls order planner and simulation work on the GPU;
  they do not download tensors or wait for completion on the host.
- Runtime teardown calls `env.cuda_stream.synchronize()`; SONIC teardown runs
  one final synchronized inference and calls `torch.cuda.synchronize()`.
  These host waits drain outstanding work before releasing resources.
- SONIC input/output use ONNX Runtime I/O binding to existing device buffers.
  There is no explicit per-step observation/action CPU round trip. The CPU
  execution provider remains available as a fallback; any node placement or
  internal transfers involving it require provider profiling to verify.
- `wp.to_torch()` in contact handling shares Warp device arrays. Reading
  `data.nacon` as a device scalar tensor for a device comparison does not
  call `.item()` or download it.
- Planner state `.to(self.device)` calls normally see tensors already on the
  target device. Generated actions remain on-device through SONIC and simulation.

## Scope

The tables cover explicit boundaries in repository runtime code and the
inspected dependency paths it invokes. They separate active transfers,
conditional transfers, CPU copies, and synchronization. Exact transfer counts,
latency, allocator traffic, graphics-driver behavior, and ONNX provider fallback
still need a CUDA/graphics profile; no timing claims are made here.
