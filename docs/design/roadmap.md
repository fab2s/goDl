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
│  User Code / Model Definitions                          │  Phase 5+
├─────────────────────────────────────────────────────────┤
│  Graph Engine (composition, branching, loops, parallel)  │  Phase 4  ✅
├─────────────────────────────────────────────────────────┤
│  Layers & Optimizers (Linear, Conv, GRU, Adam, SGD)     │  Phase 3  🔧
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

## Phase 3: Layers & Optimizers 🔧

### Status: In progress

`nn/` — neural network building blocks on top of tensor and autograd.

### What's built

- **Module interface**: `Forward(inputs ...*Variable) *Variable` + `Parameters() []*Parameter`
- **Layers**: `Linear`, `Conv2d`, `LayerNorm`, `BatchNorm`, `Dropout`, `Embedding`, `GRUCell`, `LSTMCell`
- **Activations**: `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`, `Softmax` (as modules)
- **Loss functions**: `MSELoss`, `CrossEntropyLoss`
- **Optimizers**: `SGD`, `Adam`, `AdamW`
- **Parameter** type with gradient tracking

### Remaining

- **Data loading**: parallel data pipelines with prefetching
- **Data loading**: parallel data pipelines with prefetching

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

### Remaining

- **Conditional execution**: hard routing based on tensor values (execute only
  the taken branch). Currently all branches execute; gate provides soft routing.
- **Dynamic loops**: repeat until a condition rather than fixed N.
- **Graph serialization**: save/load entire graph topologies.
- **Graph optimization**: operation fusion, scheduling passes.

---

## Phase 5: Prove It

### Goal

Port the FBRL reading model from Python to goDl. Benchmark against PyTorch.

### Prerequisites

Phase 3 completions needed: data loading pipeline.

### Deliverables

- FBRL reading model in goDl (8 parallel reading heads, GRU, foveal attention)
- Direct comparison: same architecture, same data, same GPU
- Benchmark: epoch time, GPU utilization, iteration throughput
- Target: 3-5x over Python batched, 10x+ over Python naive

---

## Phase 6+: The Long Road

Stretch goals. Each is a project in itself.

- **Model zoo**: standard architectures (ResNet, BERT, GPT) as reference implementations
- **Distributed training**: multi-GPU, multi-node via NCCL
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
Phase 3 (Layers & Optimizers) 🔧   Phase 4 (Graph Engine) ✅
    │                                    │
    └──────────┬─────────────────────────┘
               v
         Phase 5 (FBRL port / benchmark)
```

Phase 3 is the main remaining prerequisite for Phase 5. The graph engine
is ready — it needs more layers to express real architectures.

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
