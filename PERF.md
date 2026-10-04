# CPU transfers and why they occur

Audit of the carry-box runtime and the dependency paths it invokes. Assume
`--device cuda`; CPU tensors do not cross a device boundary on a CPU run.
This is source inspection with CPU numerical/rendering checks, rather than a
CUDA profile. GPU latency, driver traffic, and ONNX node placement are unmeasured.

Inspected dependencies: MJLab 1.6.0 at
`0fb8a681136be94ffc636a3dd423cabb97d91f10`, MuJoCo 3.11.0,
Transformers 4.57.1, Diffusers 0.37.0, and qwen-vl-utils 0.0.14.

## Changes to the image path

- Native rendering downloads only `qpos` on captures between resets. Origins
  and room/table mocap poses are cached on the first capture after reset.
  The scene has no event or action that moves these bodies. `reset()` invalidates
  the cache; adding moving mocap bodies would require refreshing it.
- `Renderer.render(out=...)` writes into slices of one final NumPy batch.
  The extra per-frame `.copy()` and batch `np.stack()` are eliminated. Each
  `rgb()` result owns a new batch, so later captures do not overwrite it.
- Qwen multimodal position IDs are computed from CPU processor output before
  upload. Explicit positions bypass the backbone's token/grid downloads,
  token-location synchronization, and unused rotary-delta upload.
- The last token/mask/grid batch is cached using CPU tensor equality. Unchanged
  metadata reuses device token IDs, boolean attention masks, and position IDs;
  changed tokens, padding, or image grids rebuild the cache. Pixels stay fresh.
  Boolean masks also use one byte per element on their initial upload, rather
  than uploading the processor's eight-byte integers before converting them.
- Image-grid metadata stays on CPU for SDPA. In the pinned implementation it
  controls Python shapes, positional tables, and image splits; attention
  activations remain on GPU. FA2 still uploads the grid to construct CUDA
  cumulative sequence lengths for that attention path.
- Pixel patches are cast to the vision model's BF16 dtype before upload,
  halving their upload payload. Qwen previously performed this cast on GPU.

For this scene, `nq=57` and `nmocap=2`. Explicit tensor downloads per RGB batch
drop from four to **one between resets**, with all four on the first capture
after reset. Float32 pose payload drops from `296 × n` to `228 × n` bytes.
Raw RGB remains `691,200 × n` bytes. Two extra full-batch RGB CPU copies are
removed, totaling `1,382,400 × n` copied bytes per capture. These are logical
payload counts, rather than measured bus traffic or speedups.

With unchanged metadata, local planner input uploads drop from four tensors
to just the pixel tensor in SDPA. Qwen still uploads two interpolation tables
inside its vision model on every prediction; those are listed below.

### Metadata cache cost

A local CPU microbenchmark compares the pinned Qwen `get_rope_index` with the
three CPU equality checks used on a cache hit. Each synthetic world has 256
tokens and one image grid `[1,14,14]`; PyTorch uses one CPU thread. Times are
medians from `torch.utils.benchmark.Timer` with 0.5-second measurement windows.

| Worlds | Position construction | Cache comparison | Metadata upload avoided |
| --- | --- | --- | --- |
| 1 | 0.289 ms | 0.004 ms | 8.25 KiB |
| 8 | 3.033 ms | 0.007 ms | 66 KiB |
| 32 | 8.041 ms | 0.017 ms | 264 KiB |

Upload payload is `33 × n × L` bytes: int64 token IDs, boolean masks, and
three int64 position-ID planes. This measures CPU setup savings only, excluding
GPU transfers and full planner latency. Rendering, preprocessing, tokenization,
vision inference, and denoising still run on every call. The cache has no measured
end-to-end throughput benefit yet.

## Image and planner input boundaries

| Operation | Frequency / direction | Is it needed? |
| --- | --- | --- |
| `MjlabEnv.rgb`: `qpos.cpu().numpy()` | Each prediction; D2H `[n,57]` | **Required by the native renderer.** Host `MjData` needs current robot/box poses. A device renderer could remove this boundary. |
| Origins, mocap positions, mocap quaternions `.cpu().numpy()` | First RGB after reset; three D2H arrays | **Required to initialize the host scene, then cacheable.** Repeated downloads are removed. Replay/scene mode does not initialize this cache. |
| Host `MjData` assignments and rebasing | Each world; CPU copies and float32-to-float64 conversion | **Required by the native renderer's data representation.** Simulation buffers remain unchanged. |
| `mj_forward`, `update_scene`, `mjr_render` | Each world; CPU scene computation and OpenGL submission | **Required by this renderer.** Mesh/texture resources initialize once; dynamic scene submission repeats. Driver upload details are opaque to Python. |
| `Renderer.render`: `mjr_readPixels` into the final batch slice | Each world; graphics framebuffer to CPU | **Required by MuJoCo's public native RGB API.** CUDA/graphics interoperability or a device renderer could remove this readback. |
| Renderer's `out[:] = np.flipud(out)` | Each world under EGL/GLFW/OSMesa; CPU copy | **Needed for the current image orientation.** The library does this internally; custom readback could fuse the flip. |
| `torch.from_numpy(images)` | Each capture; shared CPU storage | **No transfer or copy.** The deleted frame copy and batch stack were unnecessary. |
| Planner `images.cpu().numpy()` | Each prediction; CPU storage view here | **No current D2H.** Native RGB is already on CPU. CUDA images would download because the next step uses PIL. |
| PIL conversion, checkpoint resize and crop | Each image; CPU copies/resampling | **Transformations are needed; CPU execution is a choice.** Device processing could replace them after checking resampling/rounding equivalence. |
| `qwen_vl_utils.fetch_image`: RGB conversion and smart resize | Each image; CPU work/copies | **Part of reference preprocessing.** Aligns size to the patch grid. Even a same-size PIL resize can copy; removing/combining stages needs pixel/processor parity checks. |
| Qwen processor: resize, normalization, temporal expansion, patch packing | Each prediction; CPU array/tensor allocations | **The layout and values are needed.** CPU execution follows from PIL inputs. The fast processor accepts device tensors, but changing only this stage still uploads host images. |
| CPU BF16 pixel cast | Each prediction; CPU conversion | **A deliberate smaller upload.** Uses the vision model's existing input dtype. |
| Processed pixel patches `.to(device)` | Each prediction; H2D | **Required while pixels/preprocessing are on CPU.** A complete device rendering/preprocessing path could remove it. |
| Token IDs and boolean attention mask `.to(device)` | When token/mask/grid metadata changes; H2D | **Required to initialize GPU inputs from CPU tokenization.** Repeated uploads are removed; equality checks use CPU tensors. |
| Multimodal position IDs `.to(device)` | When metadata changes; H2D `[3,n,L]` | **Required for GPU rotary embeddings with CPU-generated positions.** Repeated position construction and upload are removed, with no token round trip. |
| Image-grid metadata | CPU in SDPA; H2D on metadata changes in FA2 | **SDPA upload was unnecessary and is removed.** The current FA2 vision path uses an uploaded grid to construct CUDA sequence lengths; that grid is now reused. |

Native rendering remains sequential. MJLab's `CameraSensor`/`SensorContext`
already offers batched MuJoCo Warp rendering with shared device RGB tensors.
That renderer plus device preprocessing could remove the host pose/readback/
pixel-upload path. It changes rendering/shading and needs image and checkpoint
comparisons on the target GPU. This change preserves the existing renderer
and preprocessing algorithms.

## Qwen dependency transfers and synchronization

Inspected `modeling_qwen3_vl.py`, `masking_utils.py`,
`modeling_flash_attention_utils.py`, and the attention integration modules.

| Operation | Current behavior | Is it needed? |
| --- | --- | --- |
| `get_rope_index`: token `.tolist()`, grid `.item()`, `argwhere`, Python conditions | CPU on metadata changes, before upload | **CUDA downloads were unnecessary and are removed.** Positions depend on metadata, independently of weights. Unchanged metadata also skips reconstruction. |
| Vision `rot_pos_emb`: grid max/count `.item()` and shape scalars | CPU reads in SDPA; GPU scalar synchronization in FA2 | **FA2 reads are avoidable metadata round trips.** Retain/cache CPU shapes separately from CUDA sequence lengths. |
| Vision `fast_pos_embed_interpolate`: grid shape reads | CPU in SDPA; implicit GPU scalar reads in FA2 | **FA2 reads are avoidable.** Shapes are already known before upload. |
| Same function: index/weight tables constructed with `torch.tensor(..., device=...)` | Two H2D tensors per prediction | **Tables must reach GPU; repeated upload is cacheable per grid/dtype/device.** Embedding lookup stays on GPU. Its intermediate `.tolist()` calls read CPU tensors. |
| Vision attention `lengths.tolist()` for Q/K/V splits | Three CPU list conversions per vision block in SDPA | **No current D2H.** Keeping the grid on CPU also keeps split lengths on CPU; activations remain on GPU. |
| `get_image_features`: `split_sizes.tolist()` | CPU in SDPA; D2H in FA2 | **FA2 download is avoidable.** Split sizes are known from CPU grids. |
| Text causal-mask creation: `padding_mask.all()` / `attention_mask.all()` in Python | GPU scalar synchronization once per conditioning call, for SDPA or FA2 respectively | **Implementation overhead.** Padding is known from CPU tokenization. A prepared mask could avoid discovering it on GPU. |
| Image placeholder validation: boolean indexing to check selected embedding size | Dynamic CUDA result-size synchronization | **Validation is useful; synchronization is avoidable using CPU token/grid counts.** No complete activation tensor downloads. |
| Text DeepStack `hidden_states[visual_pos_masks, :]` | Dynamic-size synchronization in each configured DeepStack layer | **Feature insertion is needed; dynamic allocation is a choice.** Fixed index gathers/scatters derived from CPU tokens could avoid it. |
| FA2 text `_get_unpad_data`: `nonzero()` and `max().item()` for padded prompts | Conditional index-size synchronization and D2H max length per attention layer | **Unpadding is needed for this path; host length discovery is avoidable.** CPU tokens already determine lengths/indices. Equal-length unpadded prompts skip this branch. |
| FA2 `_is_packed_sequence` and position-based varlen preparation | Conditional GPU scalar decision for batch size 1; `nonzero()` and max-length `.item()` if the position path is selected | **Metadata overhead.** Positions already exist on CPU. Separate from the SDPA path. |

FA2 vision also passes GPU-derived maximum lengths into its attention helper.
Final scalar handling depends on the installed Flash Attention implementation
and needs GPU profiling. FA2 is optional and is not installed by the manifest;
the available backend is chosen at load. None of these counts imply measured
GPU latency or every internal kernel/driver synchronization.

## Control, termination, and scheduler

| Operation | Frequency / direction | Is it needed? |
| --- | --- | --- |
| Runtime `(terminal_steps != 0).all().item()` | Once per action chunk; D2H boolean/host wait | **Needed for the current Python early-exit decision.** Fixed-budget execution could defer it, changing when the rollout stops. Chunk checks already amortize it. |
| Final steps/success/fall `.cpu().tolist()` | Three D2H arrays once per rollout | **Needed to print results.** Could be packed into one download; outside the steady loop. |
| MJLab RL `step`: `reset_buf.nonzero()` | Bypassed; no per-control-step index discovery | **Removed from the control path.** `MjlabEnv.step` runs action processing, physics substeps, counters, and the final forward refresh directly. Runtime tracks outcomes and performs explicit resets. |
| MJLab pending-reset checks, per-step manager logging and recorder downloads | Bypassed during control; manager reset paths remain | **Unused by this scene.** There are no reward, termination, observation, command, metric, or recording terms, and events only run on reset. Adding per-step terms requires updating the local step path and this audit. |
| Simulation NaN diagnostics | Conditional within `sim.step()` | **Inactive here.** NaN guard is off; the direct physics path retains the simulation's guard behavior. |
| Diffusers `set_timesteps`: CPU NumPy sigma schedule `.to(device)` | H2D once per prediction | **The schedule is needed on GPU; repeated upload is cacheable.** Inference-step count is fixed. The scheduler must also reset its step index; merely skipping this call is incorrect. |
| Diffusers timestep lookup `.item()` | Bypassed by `set_begin_index(0)` | **No active transfer.** Constructor endpoint `.item()` calls read CPU tensors. Denoising uses device sigmas and Python indices. |

## Optional native viewer

`viewer.sync()` runs every control step with `--viewer`, or about every 20 ms
in scene mode. The pinned native viewer downloads the selected world and loops
over all other worlds. `max_extra_envs=0` does not limit that loop.

| Viewer payload | Is it needed? |
| --- | --- |
| `qpos` | **Needed for native display of current poses.** D2H for each displayed world each sync. |
| `qvel` | **General viewer full-state reconstruction.** Ordinary pose-only RGB does not require velocities. |
| `ctrl` | **General viewer full-state reconstruction.** Ordinary pose-only RGB does not require actuator controls. |
| `mocap_pos`, `mocap_quat` | **Needed for poses, but fixed here and cacheable.** Head-camera rendering now caches them. |
| `xfrc_applied` | **General viewer state/force visualization.** Ordinary pose-only RGB does not need it; our passive viewer has no external-force interaction. |
| Expanded visual model fields | **Conditional on randomization/variants.** Changed fields must update the host model; unchanged fields are cacheable. |
| Perturbation generalized-force upload | **Inactive.** Perturbations are disabled and the viewer write-back path is not invoked. |

Six state downloads per world each sync occur in this scene, followed by host
assignments/forward computation and OpenGL submission. This general viewer
behavior is optional; headless policy/replay runs avoid it.

## Startup and reset

| Payload / location | Direction / frequency | Is it needed? |
| --- | --- | --- |
| Warp model/state, MJLab indexing/defaults, actuator scales/offsets and origins | H2D at initialization | **Needed for GPU simulation of a compiled host model.** Model-field expansion can stage existing Warp arrays via `.numpy()` and upload replacements; setup work. |
| Native renderer meshes/textures | Host to graphics device at renderer initialization | **Needed by native rendering.** Driver allocation/transfer details are opaque to Python. |
| Body/planner/hand joint maps in `MjlabEnv` | Three H2D tensors once | **Needed for device indexing in the external joint layout.** Small constants. |
| Reset base pose `[n,7]`, joints `[n,43]`, box pose `[n,7]` | Three H2D arrays per external episode reset | **Needed to initialize GPU state from CPU evaluation data.** Subsequent clones/rebasing are device copies. |
| Task hand-geometry IDs | H2D once | **Needed for device contact filtering.** Names/IDs originate in the host model. |
| Replay actions `[n,max_steps,78]` | H2D once in replay mode | **Needed to execute CPU archive data on GPU.** Already uploaded once and sliced on-device. |
| Four normalization arrays | H2D once per planner | **Needed for GPU normalization using JSON constants.** No repeated upload. |
| VLM/action-head weights and buffers | H2D once per planner | **Needed to load the checkpoint on GPU.** Safetensors open on CPU, then modules move. A separate fixed-frequency upload preserves precision. |
| SONIC default positions and two joint-order maps | H2D once per controller | **Needed for GPU packing/reordering.** Small constants. |
| SONIC ONNX weights/constants | H2D at session initialization for CUDA nodes | **Needed for GPU inference.** ONNX Runtime owns placement/transfers. |

## Operations without CPU round trips

- SONIC I/O binding points to persistent device buffers. The pinned decoder
  contains MatMul/Add/Sigmoid/Mul, constant slices, concatenation, and squeeze
  operations. Local code has no per-step observation/action download. A CPU
  provider's availability does not prove fallback; CUDA placement needs profiling.
- `wp.to_torch()` shares Warp contact arrays. `torch.isin`, comparisons, and
  scatter reductions stay on-device. `nacon` is compared without `.item()`.
- Planner state `.to(device)` normally sees the correct device already.
  Actions/history remain on GPU; their clones, reordering, concatenation, and
  dtype conversions are device work. `.numpy()`/`.tolist()` on CPU tensors are
  host operations, rather than GPU transfers.
- `wait_stream()` orders GPU work without a host wait or download. Synchronized
  teardown/final SONIC inference drains work before releasing resources; those
  host waits are needed by the current resource lifecycle.

## Verification and limits

Native Mesa EGL checks cover world rebasing, unchanged simulation inputs,
independent image storage, and cache refresh after reset. In the full HSSD scene,
two CPU worlds produce exactly the same pixels as the previous rendering path
initially, after a control step, and after reset. Rendering leaves `qpos`, `qvel`,
`ctrl`, `mocap_pos`, `mocap_quat`, and simulation time unchanged.
A tiny real Qwen3-VL model compares updated conditioning with a separate,
unmodified causal wrapper, including unequal image grids and right padding, with exact BF16
output equality. Repeated conditioning checks cover fresh pixels with unchanged
metadata, then cache refresh for changed tokens, padding, and image grids. The
existing SONIC batch/history test also passes.

Direct physics stepping is compared against the pinned MJLab RL step with two
worlds using the actual G1 actuators, box, and table (the visual room is omitted).
CPU simulator state, action histories, planner state, box poses, task outcomes,
and counters match exactly across changing actions and an explicit episode reset.
The local control path is also exercised with Python `nonzero()` calls forbidden.

CUDA/FA2 execution, full checkpoint rollouts, driver copies, allocator traffic,
and throughput require target-GPU profiling. File setup and dataset parsing
are CPU-only work and do not cross a GPU boundary.
