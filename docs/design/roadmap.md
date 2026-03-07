# goDl Roadmap

A Go-native deep learning framework built from first principles.

---

## Vision

A complete DL framework in Go that can eventually express anything Python/PyTorch
can — but faster, safer, and with native support for architectures that Python
punishes (recurrent attention, adaptive computation, hypothesis-test loops).

Long-term stretch goal: translate most Python DL scripts into Go equivalents.

---

## Guiding Principles

1. **Own every layer.** No black-box dependencies that can't be replaced. libtorch
   is the exception — it bridges to CUDA kernels — but bindings, API, autograd,
   and graph engine are all native Go.

2. **Idiomatic Go.** Not "PyTorch translated to Go." Leverage Go's strengths:
   interfaces, goroutines, strong typing, explicit error handling, zero-cost
   abstractions.

3. **Safety by default.** Tensor memory management should not require manual
   `Drop()` calls scattered everywhere. The ownership model must be right from
   the start.

4. **Composable primitives.** Small pieces that do one thing, compose into
   complex flows. A GRU cell is just a function. A reading head is a composition
   of functions. A model is a composition of compositions.

5. **Human-readable graphs.** Someone with no Go experience should be able to
   read a graph definition and understand what the model does. The fluent API
   reads as data flow, not as graph construction commands.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  User Code / Model Definitions                          │  Phase 6+
├─────────────────────────────────────────────────────────┤
│  Graph Engine (composition, branching, loops, parallel)  │  Phase 4  ✅
├─────────────────────────────────────────────────────────┤
│  Layers & Optimizers (Linear, Conv, GRU, Adam, SGD)     │  Phase 3  ✅
├─────────────────────────────────────────────────────────┤
│  Autograd Engine (Go-native, reverse-mode AD)           │  Phase 2  ✅
├─────────────────────────────────────────────────────────┤
│  Tensor API (Go-idiomatic wrapper)                      │  Phase 1b ✅
├─────────────────────────────────────────────────────────┤
│  libtorch C Bindings (CGo, minimal, focused)            │  Phase 1a ✅
├─────────────────────────────────────────────────────────┤
│  libtorch / CUDA                                        │  External
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1a: libtorch C Bindings ✅

### Status: Complete

`internal/libtorch/` — CGo bindings to libtorch via a C++ shim layer.

### What's built

- **C++ shim** (`shim.h`, `shim.cpp`) — wraps libtorch C++ API as C functions
- **Go bindings** (`libtorch.go`) — CGo wrappers with error handling
- Tensor creation: `Zeros`, `Ones`, `Rand`, `FromFloat32`, `FromFloat64`
- Core ops: `Add`, `Sub`, `Mul`, `Div`, `Matmul`, `ReLU`, `Sigmoid`, `Tanh`,
  `Exp`, `Log`, `Neg`, `Sqrt`, `Sum`, `SumDim`, `MeanDim`, `Transpose`,
  `Reshape`, `MulScalar`, `AddScalar`, `MaxDim`
- Indexing: `IndexSelect`, `IndexAdd`, `Select`, `SelectScatter`
- Gate support: `Softmax`
- Comparison: `GTScalar`, `OnesLike`, `ZerosLike`
- Device management: CPU/CUDA creation, device transfer
- Memory: explicit free at the binding level

---

## Phase 1b: Tensor API ✅

### Status: Complete

`tensor/` — safe, idiomatic Go tensor type with scope-based memory management.

### What's built

- **Scope-based memory** — `tensor.WithScope(func(s *Scope) { ... })` frees
  intermediates automatically. GC finalizers as safety net.
- **Chainable API** with error propagation — operations chain without
  intermediate error checks; errors propagate through the chain and surface
  at the end.
- Shape, dtype, device tracking
- Data transfer: Go slices ↔ Tensor, CPU ↔ CUDA
- All ops from Phase 1a exposed as chainable tensor methods

### Key design decision: Scope-based memory

Tensor memory lives in libtorch's C allocator. Go's GC doesn't see it.
Scopes make ownership explicit without being as noisy as manual Drop:

```go
tensor.WithScope(func(s *tensor.Scope) {
    t := s.Zeros(3, 4)   // freed when scope exits
    u := t.Matmul(t)     // also freed when scope exits
    return u.Escape()    // explicitly keep this one
})
```

---

## Phase 2: Autograd Engine ✅

### Status: Complete

`autograd/` — Go-native reverse-mode automatic differentiation.

### What's built

- **Variable** type — tensor + gradient + computation graph link
- **Reverse-mode AD** — topological sort of computation graph, reverse walk
- `NoGrad` context for inference
- `Detach` to stop gradient flow
- `ZeroGrad` for parameter gradient reset
- Gradient accumulation for parameters used multiple times
- **Supported ops with backward**: Add, Sub, Mul, Div, Matmul, ReLU, Sigmoid,
  Tanh, Sum, SumDim, MeanDim, Exp, Log, Neg, Sqrt, Transpose, Reshape,
  MulScalar, AddScalar, MaxDim, Softmax, Select, IndexSelect, Conv2d

### Key design decision: Go-native autograd

goDl builds its own autograd rather than using libtorch's. This provides full
control over gradient routing through the graph engine — essential for loops,
branches, forward references, and gated routing.

Each operation records a `gradFn` node with pointers to its inputs and a
backward function. `Variable.Backward()` walks these nodes in reverse
topological order. libtorch handles the forward math; Go handles the graph.

---

## Phase 3: Layers & Optimizers ✅

### Status: Complete

`nn/` — neural network building blocks on top of tensor and autograd.

### What's built

- **Module interface**: `Forward(inputs ...*Variable) *Variable` + `Parameters() []*Parameter`
- **Layers**: `Linear`, `Conv2d`, `LayerNorm`, `BatchNorm`, `Dropout`, `Embedding`, `GRUCell`, `LSTMCell`
- **Activations**: `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`, `Softmax` (as modules)
- **Loss functions**: `MSELoss`, `CrossEntropyLoss`
- **Optimizers**: `SGD`, `Adam`, `AdamW`
- **Parameter** type with gradient tracking
- **Gradient clipping**: `ClipGradNorm`, `ClipGradValue`
- **Weight initialization**: `KaimingUniform`, `KaimingNormal`, `XavierUniform`, `XavierNormal`
- **Checkpoints**: `SaveParameters`, `LoadParameters` (binary format with validation)
- **Data loading**: `Dataset` interface, `TensorDataset`, `Loader` with parallel prefetch and shuffle

### Module interface

```go
// Module is anything with learnable parameters that transforms variables.
// A linear layer is a Module, a full model is a Module, a graph is a Module.
type Module interface {
    Forward(inputs ...*autograd.Variable) *autograd.Variable
    Parameters() []*Parameter
}
```

The variadic Forward enables multi-input modules (merge nodes, attention with
key/query/value, modules receiving cross-wired references via Using).

---

## Phase 4: Graph Engine ✅

### Status: Complete (core)

`graph/` — composable execution graph with a fluent builder API.

### What's built

The graph engine implements a fluent API where graph definitions read as data
flow descriptions:

```go
g, err := graph.From(encoder).
    Through(attention).Using("memory").   // forward ref: recurrent state
    Through(ffn).Tag("memory").           // captured for next iteration
    Through(decoder).
    Build()
```

**Linear chains:**
```go
From(encoder).Through(relu).Through(decoder).Build()
```

**Parallel branches** with automatic goroutine parallelism:
```go
From(encoder).Split(headA, headB, headC).Merge(graph.Add()).Build()
```

**Residual connections:**
```go
From(encoder).Also(transform).Build()  // output = input + transform(input)
```

**Fixed iteration loops** with BPTT:
```go
From(encoder).Loop(refinementStep).For(5).Through(decoder).Build()
```

**Cross-wiring** (Tag/Using):
```go
From(encoder).Tag("features").
    Through(layer).
    Through(crossAttn).Using("features").  // backward ref: same-pass
    Build()
```

**Forward references** (Using before Tag) — state carried between Forward() calls:
```go
From(encoder).
    Through(attention).Using("memory").    // nil on first pass, then previous state
    Through(ffn).Tag("memory").            // captured for next call
    Build()
```

**Gated routing** (Mixture-of-Experts style):
```go
From(encoder).
    Gate(router, expertA, expertB, expertC).
    Build()
```

**Graph-as-Module** — graphs implement Module, enabling hierarchical composition:
```go
inner, _ := From(l1).Through(relu).Build()
outer, _ := From(inner).Through(l2).Build()  // inner graph is a node
```

### Key design details

- **Topological level parallelism**: nodes with no dependencies execute
  concurrently via goroutines. Split branches naturally parallelize.
- **Tag/Using**: Tag names a point in the flow. Using wires it as extra Forward
  args. Order determines semantics — Tag before Using = backward ref (same-pass),
  Using before Tag = forward ref (cross-call state).
- **Forward ref state**: state_read nodes (DAG roots) read from buffers. Writer
  nodes capture output after execution. No DAG cycles — feedback is across time.
  `ResetState()` clears buffers. `DetachState()` breaks gradient chains.
- **Gate**: router owns normalization (softmax/sigmoid/top-k). Gated merge just
  applies weights as-is: `sum_i(weights[...,i] * expert_i)`.
- **onTarget**: tracks which node Using() wires to. Set by Through/Gate/Merge/
  Also/Loop.For. After Split: wires to ALL branch modules.

### Completed since initial plan

- **Conditional execution (Switch)**: hard routing — router selects one branch,
  only that branch executes. `FixedSelector`, `ArgmaxSelector` built-in.
- **Dynamic loops (While/Until)**: condition-based iteration with Module-based
  halt conditions. `ThresholdHalt`, `LearnedHalt` (ACT pattern) built-in.
- **Map**: per-element processing with `Each()`, `Over(tag)`, `Slices(n)`,
  and `Batched()` fast path.
- **Parameter freezing**: `Freeze`/`Unfreeze`/`ZeroFrozenGrads` by tag name.
- **Visualization**: `DOT()` and `SVG()` output with typed node shapes, parameter
  counts, execution levels, and forward-ref state loops.
- **RefValidator**: build-time validation of Using ref contracts.
- **Graph primitives**: `StateAdd`, `SoftmaxRouter`, `SigmoidRouter`, `Reshape`.
- **Documentation**: 8 progressive tutorials, design docs, complete examples.
- **Test coverage**: 351 tests including numerical gradient checks (autograd ops,
  all NN modules, exact optimizer step verification), all passing with race detector.

### Remaining

- **Graph serialization**: save/load entire graph topologies (not just parameters).
- **Graph optimization**: operation fusion, scheduling passes.

---

## Phase 5: Training Refinements ✅

### LR Scheduling ✅

`Scheduler` interface wrapping `LRAdjustable` optimizer. Four schedules:
`StepDecayScheduler` (staircase), `CosineScheduler` (cosine annealing),
`WarmupScheduler` (linear warmup composable with any inner scheduler),
`PlateauScheduler` (reduce on plateau via `Observe(metric)`).
Self-contained in `nn/scheduler.go`.

### Mixed Precision ✅

Full dtype casting pipeline: `Float16`, `BFloat16` support added to
every layer (shim → libtorch → tensor). Tensor methods: `Half()`,
`ToBFloat16()`, `Float()`, `ToDType()`, `AllFinite()`. Training
utilities: `GradScaler` (dynamic loss scaling with growth/backoff),
`CastParameters()`. `Float32Data()`/`Float64Data()` auto-cast from
any dtype for safe data access and checkpointing.

---

## Phase 5b: Interruption & Control Flow

Go has genuine advantages over Python for computation control — goroutine
cancellation, context propagation, and signal handling. This phase exploits
them for patterns that are awkward or impossible in PyTorch.

### Context-Aware Forward ✅

`ForwardCtx(ctx, inputs...)` threads `context.Context` through graph
execution. Loops (For, While, Until) and Maps check `ctx.Err()` between
iterations. The context is also checked between topological levels.

- **Wall-clock timeouts for dynamic loops**: While/Until abort mid-loop
  when a deadline expires, instead of running to `maxIter`.
- **Cooperative cancellation**: cancel a forward pass from another goroutine.
- **Zero overhead**: `context.Background()` checks compile to a nil return
  (~2-5ns per check). No measurable impact on training throughput.
- **Until guarantee preserved**: Until still runs the body at least once
  before checking for cancellation.

Implementation: `execCtx` holder created at build time, shared with loop/map
closures via pointer. `Forward` delegates to `ForwardCtx(context.Background(),
...)`. 8 tests covering all loop types, maps, between-level cancellation, and
partial cancellation.

Genuinely novel — PyTorch has no equivalent because Python's threading model
doesn't support cooperative cancellation inside a forward pass.

### Profiling ✅

Opt-in per-node and per-level execution timing. Zero overhead when
disabled (bool gate before `time.Now()` calls).

- `EnableProfiling()` / `DisableProfiling()` / `Profile()` / `Timing(tag)`
- `LevelTiming.Parallelism()` — SumNodes/WallClock ratio for parallel efficiency
- `OnProfile(fn)` — hook called after each Forward
- Timing trends: `CollectTimings(tags...)`, `FlushTimings()`, `TimingTrend(tag)`
  — mirrors the value observation pipeline, reuses `*Trend` type
- 14 tests covering all profiling paths

### TagGroup & TrendGroup ✅

`TagGroup` names parallel branches with auto-suffixed tags. `TrendGroup`
provides aggregate queries over multiple trends.

- `Split(...).TagGroup("head")` → creates `"head_0"`, `"head_1"`, etc.
- Group registered as `tagGroups map[string][]string` on Graph
- Build-time validation: duplicate group name, suffix collision with existing Tag,
  TagGroup on single stream (guides user to Tag instead)
- `TrendGroup []*Trend` — pure query layer, zero storage
- `g.Trends(tags...)` / `g.TimingTrends(tags...)` expand groups, return TrendGroup
- All/Any variants: `AllImproving`, `AnyImproving`, `AllStalled`, `AnyStalled`,
  `AllConverged`, `AnyConverged`
- Aggregates: `MeanSlope`, `Slopes` ([]float64 for custom logic)
- Suffixed tags work with all existing APIs: Tagged, Collect, Timing,
  ParametersByTag, Freeze/Unfreeze
- 22 tests (10 TagGroup + 12 TrendGroup)

### Early Exit

Models that can skip remaining layers when confidence is high. Different from
Switch (which picks one branch) — this skips the *rest of the chain*:

```go
g, err := graph.From(encoder).
    Through(layer1).EarlyExit(exitHead1, confidenceProbe).
    Through(layer2).EarlyExit(exitHead2, confidenceProbe).
    Through(layer3).
    Build()
```

Real use cases: BranchyNet, early-exit transformers, inference cost reduction.
Each exit point needs an exit head producing the same output shape as the final
layer. The graph builder can wire these naturally as an extension of the
existing conditional execution primitives.

### Graceful Training Interruption

Auto-checkpoint on Ctrl+C using Go's signal handling:

```go
ctx, stop := signal.NotifyContext(ctx, os.Interrupt)
```

Combined with the existing checkpoint system, a `TrainLoop` utility could
handle signals + periodic auto-save. Mostly a training loop concern rather
than a graph concern.

---

## Phase 5c: Numerical Robustness & Higher-Order Gradients

Deep loops, long chains, and advanced loss functions push autograd beyond
simple first-order backward. This phase hardens the training stack for
production-grade stability.

### Loop stability

These are loop-scoped — they extend the existing `Loop` builder without
touching the core autograd engine.

**Gradient checkpointing** — don't store forward intermediates for every
loop iteration; recompute forward during backward. Trades compute for
memory. Critical for `Loop.For(100+)`:

```go
Loop(body).For(100).Checkpoint()  // recompute forward in backward
```

**Truncated BPTT** — detach every K iterations within a single forward
pass. Simple and high-impact for recurrent training:

```go
Loop(body).For(100).Truncate(10)  // detach gradients every 10 steps
```

**Per-iteration gradient monitoring** — extend the profiling/trend
infrastructure to track gradient norms inside loops. Flag instability
automatically via the observation layer.

### Higher-order gradients

Differentiate through the backward pass itself. Today `Backward()`
consumes the graph; for higher-order grads, backward must build new
Variable nodes that can be walked again.

```go
// First-order (existing)
loss.Backward()

// Higher-order: keep graph alive through backward
grads := autograd.Grad(loss, inputs, autograd.CreateGraph(true))
penalty := grads[0].Norm(2)  // differentiable gradient
penalty.Backward()           // second-order backward
```

Unlocks:
- **WGAN-GP** — gradient penalty requires differentiable `∇D(x)`
- **MAML / meta-learning** — differentiate through inner optimization
- **PINNs** — loss functions involving `∂net/∂x`
- **Hessian-vector products** — second-order optimizers

Implementation is incremental: start with ops needed for gradient
penalties (Add, Mul, Matmul, norms), extend to full op coverage over time.

### Custom losses

Built on higher-order grads and numerical stability primitives. Common
patterns that are easy to get wrong: Huber, focal, contrastive, triplet.
The value is in correct log-sum-exp tricks, clamping, and gradient penalty
integration.

---

## Phase 6: Prove It

### Goal

Port the FBRL reading model from Python to goDl. Benchmark against PyTorch.

### Deliverables

- FBRL reading model in goDl (8 parallel reading heads, GRU, foveal attention)
- Direct comparison: same architecture, same data, same GPU
- Benchmark: epoch time, GPU utilization, iteration throughput
- Target: 3-5x over Python batched, 10x+ over Python naive

---

## Phase 7: Multi-GPU Scaling

The composable graph architecture (sub-graphs, Map, Split, Gate, Switch)
is designed for specialized multi-component models. Multi-GPU makes it
practical for real workloads — not LLM-scale, but the vast middle ground
of domain-specific and multi-modal training.

**Level 1: Single-host data parallelism (KISS start)**
- `tensor.To(device)` — move tensors between CPU/GPUs (libtorch shim addition)
- `graph.DataParallel(model, gpus)` — wrapper that manages replicas
- Gradient sync via CPU: copy grads to CPU → average → copy back
- Go goroutines handle per-GPU forward/backward — no process spawning
- PCIe round-trip is the bottleneck, but fine for small-to-medium models
- Training API stays identical: dp.Forward → loss → Backward → Step

**Level 2: NCCL gradient sync (same host, faster)**
- Add NCCL allreduce to shim — one C function binding
- Replace CPU-mediated averaging with direct GPU↔GPU sync (NVLink/PCIe)
- Same DataParallel API, swap the sync backend
- Significant speedup for gradient-heavy models

**Level 3: Multi-host distributed (cloud)**
- Abstract gradient sync behind an interface (already done at Level 1-2)
- Swap NCCL for gRPC-based ring-allreduce between nodes
- Go's native gRPC + goroutines make this natural — no external launchers
- Ring-allreduce: each node sends gradient shard to neighbor, N-1 rounds
- Overlap communication with backward computation for efficiency
- Fault tolerance: checkpoint + restart on node failure

**Level 4: Model parallelism (specialized)**
- Pipeline parallelism: split graph stages across GPUs (graph levels → devices)
- The graph's topological level structure maps naturally to pipeline stages
- Tensor parallelism: split individual layers (e.g. large Linear across GPUs)
- Only needed for models that don't fit on a single GPU

---

## Phase 8+: The Long Road

Stretch goals. Each is a project in itself.

- **Attention mechanisms**: MultiHeadAttention, cross-attention as graph primitives
- **Graph serialization**: save/load entire graph topologies (not just parameters)
- **Model zoo**: standard architectures (ResNet, BERT, GPT) as reference implementations
- **ONNX import/export**: interop with the broader ML ecosystem
- **Python model translation**: parse PyTorch model code, generate goDl equivalent
- **Custom CUDA kernels**: when libtorch ops aren't enough
- **Quantization & inference optimization**

---

## Dependency Map

```
Phase 1a (libtorch bindings) ✅
    │
    v
Phase 1b (Tensor API) ✅
    │
    v
Phase 2 (Autograd) ✅ ──────────────────┐
    │                                    │
    v                                    v
Phase 3 (Layers & Optimizers) ✅   Phase 4 (Graph Engine) ✅
    │                                    │
    └──────────┬─────────────────────────┘
               v
    Phase 5 (Training refinements) ✅ + Phase 5b (Interruption) 🔧
               │
               v
    Phase 5c (Numerical robustness / higher-order grads)
               │
               v
    Phase 6 (FBRL port / benchmark)
               │
               v
    Phase 7 (Multi-GPU scaling)
```

The core stack (Phases 1-4) is complete. Phase 5/5b adds training
refinements and Go-native control flow. Phase 6 proves the thesis
with a real benchmark.

---

## Dev Environment

- Go 1.24+
- libtorch 2.10+ (CUDA 12.6)
- Docker container for reproducible builds (Go + libtorch + CUDA)
- WSL2 with GPU passthrough
- `make test-cpu` / `make test` / `make shell`

---

## References

- **gotch** (`github.com/sugarme/gotch`) — CGo/libtorch binding patterns,
  memory management, C shim layer. Reference, not dependency.
- **gorgonia** (`github.com/gorgonia/gorgonia`) — ExprGraph and TapeMachine
  for autograd inspiration. Notable limitation: no loops or branches by design.
- **PyTorch internals** — Edward Z. Yang's blog posts on autograd, dispatcher,
  and tensor storage.
- **micrograd** (Karpathy) — simplest possible autograd in ~100 lines of Python.
  Ideal for understanding the core AD concept.
- **tinygrad** — minimal tensor + autograd, demonstrates essential op set.
