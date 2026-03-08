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

### Fixed
- **DetachState memory growth**: `DetachState` is now recursive — walks sub-graphs and calls `Detach()` on all `Detachable` modules. Previously only detached the graph's own forward-reference state buffers, leaving module-level state (hidden vectors, attention locations) attached. This caused unbounded memory growth in models with stateful loop bodies.

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
- 412 tests, all passing with race detector.
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
