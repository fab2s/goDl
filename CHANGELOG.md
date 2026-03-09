# Changelog

All notable changes to goDl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Record**: inject external metrics (losses, hit rates) into the Collect/Flush pipeline.
- **Trend.Latest**: convenience accessor for the most recent epoch value.
- **Flush timing**: ETA, Elapsed, FlushCount, LastFlushDuration — built-in wall-clock tracking.
- **WriteLog**: human-readable training log with per-epoch metrics, timing, and ETA.
- **FormatDuration**: training-friendly duration formatting (42ms, 1.2s, 2m05s).
- **Checkpoint**: composable training resume — bundles model parameters, optimizer state, scheduler state, and epoch into a single file. Supports `Save(epoch)` / `Load()` / `LoadEpoch(n)`.
- **Stateful interface**: `SaveState(io.Writer) / LoadState(io.Reader)` implemented by all optimizers (SGD, Adam, AdamW), schedulers (StepDecay, Cosine, Warmup, Plateau), and GradScaler.
- **Detachable interface**: `nn.Detachable` for modules holding Variables in struct fields across Forward calls. `nn.Detach(m)` helper.
- **SubModuler interface**: `nn.SubModuler` for composite modules that declare child modules. The framework walks the tree automatically for device placement, training mode, state detachment, and per-forward reset. Built-in implementations on `GRUCell`, `LSTMCell`, and all graph composites (loop, switch, map).
- **CollectParameters**: `nn.CollectParameters(m)` recursively collects deduplicated parameters from a module and its `SubModuler` children. Shared weights are counted once.
- **WalkModules**: `nn.WalkModules(m, visited, fn)` generic tree walker for recursive module operations — used by `Graph.SetDevice`, `SetTraining`, `DetachState`, and `Reset`.
- **DeviceMover interface**: `nn.DeviceMover` for modules with non-parameter tensors (running statistics, buffers) that must move with `SetDevice`. `BatchNorm` implements this for its running mean/variance.
- **Device placement**: `Graph.SetDevice(device)` moves all parameters and state buffers, recurses into sub-graphs and composites via `WalkModules`. `Graph.Device()` getter. Auto-moves input tensors at graph entry when device is set.
- **Loader device**: `LoaderConfig{Device: tensor.DevicePtr(tensor.CUDA)}` moves both input and target batches after stacking.
- **Variable.ToDevice**: moves a variable's data to a target device, preserving `requiresGrad`.
- **tensor.DevicePtr**: convenience helper for `*tensor.Device` in config structs.
- **Tensor.ToInt64**: dtype cast shorthand — enables `caseLabel.GTScalar(0.5).ToInt64()` without CPU round-trip.
- **Tensor.Double**: dtype cast shorthand for Float64, mirroring `Float()` and `Half()`.

### Changed
- **Resettable.Reset**: signature changed from `Reset(batchSize int64)` to `Reset(batchSize int64, device tensor.Device)`. The graph passes the configured device (or the input's device), eliminating the need to sniff device from parameter weights.

### Fixed
- **Deterministic VRAM release**: saved-for-backward tensors are now freed during backward via Go-side atomic reference counting — no `runtime.GC()` calls needed in training loops. `Tensor.Retain()` / `Release()` manage the lifecycle; the engine releases saved tensors immediately after each op's backward pass, old gradient accumulators on replacement, and stale leaf gradients on accumulation. All 23 tensor-saving ops covered. See `docs/design/memory-management.md`.
- **CUDA OOM → GC callback**: when the CUDA caching allocator exhausts its free-block pool, a registered `FreeMemoryCallback` triggers Go's garbage collector to finalize unreachable forward-intermediate tensors. A pending-free queue prevents deadlock with the allocator's recursive mutex. Zero cost in the happy path — fires only under VRAM pressure. See `docs/design/memory-management.md`.
- **CUDA use-after-free in GC callback**: the OOM callback previously drained pending tensor frees on the same call stack as the CGo operation that triggered the allocation — freeing C++ handles that libtorch was still using. Manifested as SIGSEGV with varying symptoms (corrupted vtable, "tensor does not have a device", TLS assertion failures) under VRAM pressure. Fixed by deferring the drain to the next `Free()` call outside the callback and adding `runtime.KeepAlive` to all ~50 CGo call sites in `tensor/` to prevent premature GC of tensor wrappers.
- **Device-aware checkpoint and optimizer state loading**: `LoadParameters`, `SGD.LoadState`, and `Adam.LoadState` now move deserialized tensors to the device of the corresponding parameter. Previously, loaded tensors were always CPU, causing device mismatch errors when resuming training on CUDA.
- **Backward memory retention**: the autograd engine now releases the computation graph during backward — `gradFn`, captured tensors, and intermediate forward results are nil'd out as each node is processed. Previously the entire graph stayed alive until the GC collected the user's loss variable, causing VRAM accumulation across training steps. See `docs/design/memory-management.md`.
- **DetachState memory growth**: `DetachState` is now recursive — walks sub-graphs via `WalkModules` and calls `Detach()` on all `Detachable` modules and nested graphs. Previously only detached the graph's own forward-reference state buffers.
- **Module device mismatch**: GRUCell, LSTMCell, and Dropout now create internal tensors (zero hidden states, dropout masks) on the same device and dtype as the input. Previously these defaulted to CPU, causing device mismatch errors after `SetDevice(CUDA)`.
- **OneHot device**: `tensor.OneHot` now returns a tensor on the same device as the input indices. Previously always created CPU tensors, causing device mismatch in `CrossEntropyLoss` on CUDA.
- **Backward seed device**: the autograd engine now creates the backward seed tensor (`ones`) on the same device as the loss. Previously defaulted to CPU.
- **ClipGradValue device**: `nn.ClipGradValue` now preserves the gradient's device when clamping. Previously could produce CPU gradients from CUDA parameters.
- **BatchNorm lazy device alignment**: `BatchNorm.Forward` moves running statistics to match the input device if they differ, preventing device mismatch when BatchNorm is nested inside user-defined composite modules that aren't direct graph nodes.
- **ETA calculation**: training start is now recorded on first Forward (not first Flush), so epoch 0's duration is included in the per-epoch average. ETA is available after 1 flush instead of requiring 2. `Elapsed()` and `WriteLog` updated consistently.

## [v0.1.0] - 2026-03-07

Initial public release.

### Core Stack
- **Tensor**: Immutable, chainable API with error propagation. CPU and CUDA.
- **Autograd**: Reverse-mode automatic differentiation with full backward for every op.
- **NN Modules**: Linear, Conv2d, ConvTranspose2d, LayerNorm, BatchNorm, Dropout, Embedding, GRUCell, LSTMCell.
- **Activations**: ReLU, Sigmoid, Tanh, GELU, SiLU, Softmax.
- **Losses**: MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, L1Loss, SmoothL1Loss, KLDivLoss.
- **Optimizers**: SGD (with momentum), Adam, AdamW.

### Graph Builder
- Fluent API: From/Through/Build, Split/Merge, Also (residual), Tag/Using (named refs).
- Loop constructs: For (fixed), While (pre-condition), Until (post-condition).
- Routing: Gate (soft, weighted), Switch (hard, selected branch only).
- Map constructs: Each, Over, Slices, with Batched fast path.
- Input (auxiliary graph inputs), TagGroup (auto-suffixed parallel branch names).
- Context-aware execution via ForwardCtx with timeout/cancellation.
- Parallel execution of independent branches via goroutines.

### Training Tools
- LR scheduling: StepDecay, Cosine, Warmup (composable), ReduceOnPlateau.
- Mixed precision: Float16/BFloat16 dtype casting, GradScaler for loss scaling.
- Gradient clipping: ClipGradNorm, ClipGradValue.
- Parameter freezing by tag: Freeze/Unfreeze.
- Checkpointing: SaveParameters/LoadParameters (binary format, file or io.Writer).
- Weight initialization: KaimingUniform/Normal, XavierUniform/Normal.
- Data loading: Dataset/TensorDataset/Loader with parallel prefetch and shuffle.

### Observation & Visualization
- Tag-based metric collection: Collect/Flush/Trend.
- Trend analysis: Slope, Stalled, Improving, Converged.
- Group trends: Trends/TimingTrends with TagGroup expansion.
- DOT/SVG graph visualization with parameter counts and node type shapes.
- Profiling: EnableProfiling, SVGWithProfile (per-node timing, color-coded).
- Training curves: PlotHTML, ExportTrends.

### Testing
- 460 tests, all passing with race detector.
- 40 autograd numerical gradient checks (finite differences).
- 10 module-level gradient checks (input + parameter gradients).
- 11 exact optimizer step verifications.

### Documentation
- PyTorch migration guide with side-by-side examples.
- 8 tutorials from basics to advanced graphs.
- Design documents: roadmap, CUDA dispatch analysis, trajectory thesis.

### Infrastructure
- Docker-based builds: CUDA image and CPU-only image (~2GB vs ~21GB).
- GitHub Actions CI with CPU Docker image.

[Unreleased]: https://github.com/fab2s/goDl/compare/v0.1.0...HEAD
[v0.1.0]: https://github.com/fab2s/goDl/releases/tag/v0.1.0
