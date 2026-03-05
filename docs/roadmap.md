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

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  User Code / Model Definitions                          │  Phase 5+
├─────────────────────────────────────────────────────────┤
│  Graph Engine (composition, branching, loops, parallel)  │  Phase 4
├─────────────────────────────────────────────────────────┤
│  Layers & Optimizers (Linear, Conv, GRU, Adam, SGD)     │  Phase 3
├─────────────────────────────────────────────────────────┤
│  Autograd Engine (Go-native, reverse-mode AD)           │  Phase 2
├─────────────────────────────────────────────────────────┤
│  Tensor API (Go-idiomatic wrapper)                      │  Phase 1b
├─────────────────────────────────────────────────────────┤
│  libtorch C Bindings (CGo, minimal, focused)            │  Phase 1a
├─────────────────────────────────────────────────────────┤
│  libtorch / CUDA                                        │  External
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1a: libtorch C Bindings

### Goal

Talk to libtorch from Go. Create, manipulate, and destroy tensors via CGo.

### Concepts covered

- CGo mechanics (Go calling C, pointer passing, memory rules)
- libtorch C API structure (the `aten/` tensor library)
- CUDA device management at the API level
- Tensor memory allocation and deallocation across C/Go boundary

### Deliverables

- `internal/libtorch/` — raw CGo bindings to libtorch's C API
- Tensor creation: `zeros`, `ones`, `rand`, `from_blob` (from Go slice)
- Basic ops: `add`, `mul`, `matmul`, `relu`, `sigmoid`, `tanh`
- Device management: create on CPU, create on CUDA, move between devices
- Memory: explicit `free` at this level (the safe wrapper comes in Phase 1b)
- A test that creates two tensors on CUDA, multiplies them, reads result back

### Key design decisions

**Which libtorch API to bind?**

libtorch exposes two C-level interfaces:
- `torch/csrc/api/` — the C++ API (what gotch uses via a C shim)
- `aten/` — the lower-level tensor library

The gotch project's C shim layer (`libtch/` directory) is worth studying as
reference. The C++ API is more ergonomic; the ATen API gives more control.
The choice depends on which provides the cleanest CGo boundary.

**CGo overhead**

Every CGo call has ~100-200ns overhead. For large tensor ops (matmul on big
matrices), this is negligible. For tiny ops in tight loops, it adds up — the
exact problem the graph engine solves later by batching operations. Phase 1a
should measure this overhead and document it.

**Build configuration and backend selection**

libtorch ships in multiple flavors: CUDA (NVIDIA), ROCm (AMD), CPU-only, and
increasingly XPU (Intel) and MPS (Apple Silicon). The same C API works across
all of them — the difference is which shared libraries are linked at build time.

This must be a first-class concern from day one:
- Build tags or environment variables select the libtorch flavor
- The device abstraction in Go exposes only what the linked backend supports
- CI tests run against at least CPU and CUDA flavors
- Documentation covers how to build for each backend

The goal: switching from NVIDIA to AMD is a build flag, not a code change.
goDl's value lives above the hardware abstraction — the tensor API, autograd,
graph engine, and layers are backend-agnostic by design.

### Reference material
- gotch's `libtch/` directory — C shim approach
- PyTorch's `torch/csrc/` — the C++ API being bound
- libtorch C API headers — the actual function signatures
- PyTorch backend matrix — which backends exist and their maturity levels

---

## Phase 1b: Tensor API

### Goal

A safe, idiomatic Go tensor type wrapping the raw bindings.

### Concepts covered

- Go memory management patterns (finalizers, reference counting, arena patterns)
- Type-safe API design in Go using generics and interfaces
- The tensor abstraction: shape, stride, dtype, device, contiguity

### Deliverables

- `tensor/` package — the public Tensor type
- Safe memory management (see design decision below)
- Shape and dtype tracking
- Slicing, reshaping, transposing (view ops vs. copy ops)
- Data transfer: Go slice <-> Tensor, CPU <-> CUDA
- Printing and debugging (human-readable tensor display)
- Comprehensive tests

### Key design decisions

**Memory management strategy**

The hardest design problem in the project. Options:

```
Option A: Explicit Drop (what gotch does)
    t := tensor.Zeros(3, 4)
    defer t.Drop()  // forget this = memory leak

Option B: GC + Finalizer
    t := tensor.Zeros(3, 4)
    // GC eventually calls C free via runtime.SetFinalizer
    // Problem: GC doesn't know about C memory pressure, may free too late

Option C: Scope-based (like Rust's ownership, via Go closures)
    tensor.WithScope(func(s *tensor.Scope) {
        t := s.Zeros(3, 4)   // freed when scope exits
        u := s.Matmul(t, t)  // also freed when scope exits
        return u.Escape()    // explicitly keep this one
    })

Option D: Arena pattern (Go 1.20+ experimented with this)
    arena := tensor.NewArena()
    defer arena.Free()
    t := arena.Zeros(3, 4)   // freed with arena
```

Current recommendation: **Option C (scope-based) as the primary API, with
Option B as fallback for simple scripts.** Scopes make ownership explicit
without being as noisy as manual Drop. Neither gotch nor gorgonia does this.

Phase 1a uses explicit free (raw bindings). Phase 1b experiments with all
four approaches before committing.

**Dtype representation**

Go generics (since 1.18) allow `Tensor[float32]` vs `Tensor[float64]`. But
libtorch tensors are dynamically typed (dtype is a runtime property). The
question: static typing via generics (compile-time error catching, more
verbose) or dynamic typing with runtime checks (matches libtorch, more
flexible)?

Current recommendation: **Dynamic dtype with a typed accessor API.** The
tensor stores dtype as a runtime value (like libtorch), but data access uses
generic functions: `tensor.DataAs[float32](t)`. This matches PyTorch's model
while adding compile-time safety where it matters most.

---

## Phase 2: Autograd Engine

### Goal

Automatic differentiation in Go. Track operations on tensors, build a backward
graph, compute gradients via reverse-mode AD.

### Concepts covered

- How automatic differentiation works (not just "call .backward()")
- Symbolic vs. reverse-mode AD
- PyTorch autograd internals
- Gradient accumulation, in-place ops, detaching, no-grad contexts

### Deliverables

- `autograd/` package
- `Variable` type — a tensor + its gradient + the operation that created it
- Operation registry — each op knows its forward and backward
- Backward pass: topological sort of the computation graph, then reverse walk
- `NoGrad` context — disable tracking for inference
- Gradient accumulation for parameters used multiple times
- Tests: hand-verified gradients for basic ops (matmul, relu, sigmoid, etc.)

### How autograd works

Every operation records itself:
```
z = matmul(x, y)
```
Internally, this creates a graph node:
```
{ op: "matmul", inputs: [x, y], output: z,
  backward: func(grad_z) -> (grad_x, grad_y) }
```

When `z.Backward()` is called:
1. Topologically sort all nodes from z back to leaves
2. Walk in reverse, calling each backward function
3. Accumulate gradients at each variable

The key insight: libtorch handles forward math (matmul, conv2d, etc.), but
the graph tracking and backward pass computation happen in Go. This provides
full control over gradient routing — essential for the graph engine.

### Key design decisions

**Go-native autograd vs. libtorch autograd**

goDl builds its own autograd. Reasons:
- Full control over gradient flow (stop gradients, scale, reroute)
- The graph engine needs to manage gradients through branches and loops
- libtorch's autograd is optimized for eager execution, not graph-based

Cost: backward functions must be implemented for every supported op. Start with
the essentials (matmul, add, mul, relu, sigmoid, tanh, softmax, conv2d,
cross_entropy) and expand as needed.

**Tape vs. graph representation**

PyTorch uses a DAG of `Function` nodes (graph-based). Some systems use a flat
tape (linear list of operations). The graph representation is more flexible —
it naturally handles operations with multiple outputs, shared inputs, and is
what the graph engine builds on.

Recommendation: **Graph-based from the start.** Each operation is a node with
edges to its inputs. The backward pass walks these edges in reverse topological
order.

---

## Phase 3: Layers & Optimizers

### Goal

Standard neural network building blocks, built on the tensor and autograd
foundation.

### Concepts covered

- Common layer mathematics (Linear = Wx+b, Conv2d = cross-correlation, GRU gating)
- Parameter management patterns
- Optimizer mechanics (SGD momentum, Adam moving averages)
- Weight initialization strategies

### Deliverables

- `nn/` package
- Core layers: `Linear`, `Conv2d`, `GRUCell`, `BatchNorm`, `LayerNorm`, `Dropout`
- Activations: `ReLU`, `Tanh`, `Sigmoid`, `GELU`, `Softmax`
- Loss functions: `CrossEntropy`, `BCE`, `MSE`
- Optimizers: `SGD` (with momentum), `Adam`, `AdamW`
- Parameter management: named parameters, save/load (state_dict equivalent)
- A `Module` interface that all layers implement

### Interface sketch (target, not final)

```go
// Module is anything with learnable parameters that transforms tensors.
// A linear layer is a Module, a full model is a Module, a graph node
// wrapping a Module is a Module.
type Module interface {
    // Forward computes the output. The Context carries autograd state,
    // device info, and training/eval mode.
    Forward(ctx *Context, inputs ...Tensor) ([]Tensor, error)

    // Parameters returns all learnable parameters, keyed by name.
    // Used by optimizers and for save/load.
    Parameters() map[string]*Parameter
}

// Parameter is a tensor that requires gradients.
type Parameter struct {
    Data     Tensor
    Grad     Tensor  // accumulated gradient, nil until first backward
    Name     string
}

// Optimizer updates parameters based on their gradients.
type Optimizer interface {
    Step()              // apply one update to all parameters
    ZeroGrad()          // reset all gradients to zero
}
```

### Why Module.Forward returns ([]Tensor, error)

- **Multiple outputs**: a GRU returns (output, hidden_state). A split operation
  returns N tensors. A slice is more honest than wrapping in a struct every time.
- **Explicit errors**: Go convention. A shape mismatch in Forward should be an
  error, not a panic.
- **Variadic inputs**: same logic. A merge node takes N inputs.

---

## Phase 4: Graph Engine

### Goal

The differentiator. A composable execution graph where nodes are Modules, with
native support for branching, loops, parallelism, and adaptive control flow.

### Concepts covered

- Computation graph design (beyond autograd's backward graph)
- Concurrent execution patterns in Go (goroutines, channels, sync primitives)
- How control flow interacts with gradient computation
- Graph optimization passes (fusion, scheduling)

### Deliverables

- `graph/` package
- `Node` — wraps a Module with graph metadata (edges, ports)
- `Graph` — a collection of connected Nodes (itself a Module — composition!)
- `Edge` — typed connection between node output port and node input port
- Execution engine: topological traversal with goroutine parallelism
- Loop construct: repeat a subgraph N times or until a condition
- Branch construct: conditional execution based on tensor values
- Gradient routing through loops (backprop through time) and branches
- Save/load entire graph topologies

### Interface sketch (target, will evolve)

```go
// Node is a Module placed in a graph, with connectivity metadata.
type Node struct {
    ID       string
    Module   Module
    // ... edges, metadata
}

// Graph is a composition of connected Nodes.
// A Graph is itself a Module — the key composition primitive.
// A Graph can be a node in a parent Graph, enabling arbitrary nesting.
type Graph struct {
    nodes []*Node
    edges []*Edge
    // ... execution order cache, parallel groups
}

// Graph implements Module — this is what makes composition work.
// Forward traverses the graph: execute nodes in topological order,
// route tensors along edges, run parallel groups as goroutines.
func (g *Graph) Forward(ctx *Context, inputs ...Tensor) ([]Tensor, error)

// Builder pattern for constructing graphs.
func NewGraph() *GraphBuilder
func (b *GraphBuilder) Add(name string, m Module) *NodeRef
func (b *GraphBuilder) Connect(from *NodeRef, port int, to *NodeRef, port int)
func (b *GraphBuilder) Loop(n int, nodes ...*NodeRef) *LoopRef
func (b *GraphBuilder) Parallel(nodes ...*NodeRef) *ParallelRef
func (b *GraphBuilder) Build() (*Graph, error)
```

### Design heritage

Several design patterns from workflow/ETL engines apply directly to ML
computation graphs:

| Workflow concept | goDl equivalent | ML purpose |
|---|---|---|
| Flow-as-Node (nested flows) | Graph implements Module | Nested sub-models |
| Payload propagation | Tensor routing along Edges | Forward pass data flow |
| Traversable nodes (yield N values) | Loop construct | Recurrent unrolling, iterative refinement |
| Branch nodes (sub-flow) | Conditional construct | Adaptive computation, data-dependent paths |
| Flow interrupts (break/continue) | Early exit from loops | Stop when confidence > threshold |
| Route-to-node (sendTo) | Skip connections / attention routing | Residual connections, arbitrary wiring |
| Flow registry | Graph.Nodes() + introspection | Visualization, debugging, optimization |
| Lifecycle events | Context hooks | Logging, profiling, gradient clipping |

### The hard problem: gradients through control flow

Looping a subgraph 5 times means the backward pass must unroll those 5 steps
in reverse (backpropagation through time). Conditional branching means only the
taken branch receives gradients. Go-native autograd makes this possible — the
tape records loop iterations and branch decisions as part of the backward graph.

---

## Phase 5: Prove It

### Goal

Port the FBRL reading model from Python to goDl. Benchmark against PyTorch.

### Deliverables

- FBRL reading model in goDl (8 parallel reading heads, GRU, foveal attention)
- Direct comparison: same architecture, same data, same GPU
- Benchmark: epoch time, GPU utilization, iteration throughput
- Target: 3-5x over Python batched, 10x+ over Python naive

---

## Phase 6+: The Long Road

Stretch goals. Each is a project in itself.

- **Model zoo**: standard architectures (ResNet, BERT, GPT) as reference implementations
- **Data loading**: parallel data pipelines with prefetching
- **Distributed training**: multi-GPU, multi-node via NCCL
- **ONNX import/export**: interop with the broader ML ecosystem
- **Python model translation**: parse PyTorch model code, generate goDl equivalent
- **Custom CUDA kernels**: when libtorch ops aren't enough
- **Quantization & inference optimization**

---

## Dependency Map

```
Phase 1a (libtorch bindings)
    │
    v
Phase 1b (Tensor API)
    │
    v
Phase 2 (Autograd) ──────────────────┐
    │                                 │
    v                                 v
Phase 3 (Layers & Optimizers)    Phase 4 (Graph Engine)
    │                                 │
    └──────────┬──────────────────────┘
               v
         Phase 5 (FBRL port / benchmark)
```

Phase 4 can start in parallel with Phase 3 at the interface level. The graph
engine needs the Module interface but not specific layer implementations.

---

## Dev Environment

- Go 1.22+ (for generics maturity)
- libtorch 2.1+ (CPU first, then CUDA)
- Docker container for reproducible builds (libtorch + Go + CUDA)
- WSL2 with GPU passthrough

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
