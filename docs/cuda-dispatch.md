# CUDA Dispatch: Why Go Can Beat Python for Deep Learning

This document explains the full call chain from user code to GPU execution,
where overhead lives at each layer, and why it matters for certain classes of
architectures.

---

## How CUDA kernels work

A CUDA **kernel** is a function that runs on the GPU. A matrix multiply, a ReLU
activation, a convolution — each is a kernel launch. At the hardware level:

1. **Host (CPU)**: Queues a kernel launch command onto a **CUDA stream** (a
   command queue)
2. **GPU scheduler**: Picks up the command, assigns thread blocks to Streaming
   Multiprocessors (SMs)
3. **SMs execute**: Thousands of threads run the operation in parallel
4. **Result lands in GPU memory** — never leaves the GPU unless explicitly
   transferred back

Kernel launches are **asynchronous** — the CPU drops a command on the queue and
moves on. The GPU picks it up whenever it's ready. The CPU's job is to **keep
the queue fed** so the GPU never sits idle.

A bare CUDA kernel launch from C/C++ costs roughly **5-10 microseconds** of
host-side overhead. This is a **CUDA driver/hardware cost** — every kernel
launch pays it regardless of language, framework, or binding strategy. It is
the non-negotiable floor.

---

## Three layers of overhead

Every DL framework adds overhead on top of the CUDA driver cost. Understanding
which layers are inherent to CUDA and which are framework-imposed is critical
for knowing what can actually be optimized away.

```
┌──────────────────────────────────────────────────────────────────┐
│ Layer 1: Language/Framework overhead                              │
│   Python interpreter, GIL, argument parsing, autograd dispatch   │
│   → PyTorch: ~3-5 μs  |  Go/CGo: ~0.3 μs  |  C++: ~0 μs       │
├──────────────────────────────────────────────────────────────────┤
│ Layer 2: libtorch overhead                                       │
│   Argument validation, dtype checks, algorithm selection         │
│   → ~0.3-0.5 μs  (same regardless of calling language)          │
├──────────────────────────────────────────────────────────────────┤
│ Layer 3: CUDA driver overhead                                    │
│   Build command, push to stream, GPU picks it up                 │
│   → ~5-10 μs  (hardware cost, unavoidable)                      │
└──────────────────────────────────────────────────────────────────┘
```

Only Layer 1 changes between Python and Go. Layer 2 (libtorch) and Layer 3
(CUDA driver) are constants. This means **per-operation dispatch savings from
switching languages have a ceiling** — the CUDA driver cost dominates.

The real wins come from reducing the **number** of kernel launches (fusion)
and launching across **multiple streams** in parallel (goroutines).

---

## The PyTorch dispatch stack

When `z = torch.matmul(x, y)` runs in Python, the actual call chain:

```
Python interpreter
  │  ~1-2 μs    Python function call overhead, GIL management
  v
THPVariable_matmul  (auto-generated C binding)
  │  ~0.5 μs   PythonArgParser: unpack Python objects to C++ types
  v
Variable dispatch  (autograd layer)
  │  ~0.5 μs   Check requires_grad, set up backward node
  v
Type dispatch  (virtual method call)
  │  ~0.2 μs   Route to CPU or CUDA backend based on tensor.device
  v
Dtype dispatch  (switch statement)
  │  ~0.1 μs   Route to float32 vs float64 implementation
  v
ATen kernel  (C++ implementation)
  │  ~0.5 μs   Argument validation, algorithm selection     ← libtorch layer
  v
cuBLAS / CUDA kernel launch
  │  ~5-10 μs   Queue kernel on CUDA stream                 ← CUDA driver layer
  v
GPU executes
```

**Total host-side overhead per operation: ~8-15 μs**

For a big matmul (1024x1024), the GPU takes ~100-500 μs. The 15 μs overhead
is 3-15% — tolerable.

For reference: Python executes approximately 32 million operations per second.
An A100 GPU performs 312 trillion FLOPS. In the time Python performs a single
FLOP, an A100 could process 9.75 million.

---

## The Go dispatch stack

From Go via CGo, calling a libtorch C function:

```
Go function call
  │  ~2-5 ns     Compiled dispatch, no interpreter
  v
CGo boundary crossing
  │  ~100-200 ns  Stack switch, save Go state, enter C
  v
libtorch C function                                          ← libtorch layer
  │  ~0.3-0.5 μs  Argument validation, algorithm selection
  v
CUDA kernel launch                                           ← CUDA driver layer
  │  ~5-10 μs      Queue kernel on CUDA stream
  v
GPU executes
```

**Total host-side overhead per operation: ~6-11 μs**

Note that libtorch's ~0.5 μs is present in both stacks — it is not a Python
artifact. Calling libtorch from C++ directly would still pay this cost. The
difference is purely in Layer 1: Go's ~0.3 μs vs Python's ~3-5 μs.

### Overhead comparison

| Component | Python/PyTorch | Go/CGo | C++ direct | Reason |
|-----------|---------------|--------|------------|--------|
| Interpreter | ~1-2 μs | 0 | 0 | Compiled, no interpreter |
| Arg parsing/boxing | ~0.5 μs | 0 | 0 | Static types, no object unwrapping |
| GIL management | ~0.3 μs | 0 | 0 | No GIL |
| Autograd dispatch | ~0.5 μs | ~0.1 μs | ~0.1 μs | Go-native autograd, no virtual chain |
| Type/device dispatch | ~0.3 μs | 0 | 0 | Known at compile time |
| CGo boundary | 0 | ~0.15 μs | 0 | Unique to Go |
| **Framework overhead** | **~3-5 μs** | **~0.3 μs** | **~0.1 μs** | |
| libtorch overhead | ~0.5 μs | ~0.5 μs | ~0.5 μs | Same library, same cost |
| CUDA driver | ~5-10 μs | ~5-10 μs | ~5-10 μs | Hardware cost |
| **Total** | **~9-16 μs** | **~6-11 μs** | **~5.5-10.5 μs** | |

The gap between Go and raw C++ is small (~0.5 μs from CGo boundary + autograd
bookkeeping). The gap between Python and Go is significant (~3-5 μs per op)
but has diminishing returns as a speedup source because the CUDA driver
dominates.

---

## When per-op overhead falls apart: small sequential kernels

Consider an architecture with 8 read heads, each running 9 sequential GRU
steps = 72 sequential operations per forward pass.

A single GRU step at batch_size=32, hidden_dim=256 involves:
- 6 small matrix multiplies (gate weights, input and hidden)
- 3 sigmoid/tanh activations
- Several element-wise ops (Hadamard products, adds)
- ~15 operations total, each on small tensors (32 x 256)

A matmul on a 32x256 tensor takes the GPU roughly **5-15 μs**. The Python
dispatch overhead is also **~15 μs per op**:

```
Per GRU step:
  15 ops x 15 μs GPU time       = ~225 μs useful work
  15 ops x 15 μs Python overhead = ~225 μs waste
  → ~50% GPU utilization at best

Per forward pass (72 steps):
  72 x 15 ops = 1,080 operations
  1,080 x 15 μs overhead = ~16 ms pure overhead
  1,080 x 15 μs GPU work = ~16 ms useful work
```

With Go, the per-op overhead drops to ~10 μs (mostly CUDA driver), improving
GPU utilization to ~60-65%. The per-op saving is real but not transformative
on its own:

```
Python:  1,080 ops x ~15 μs total = ~16 ms dispatch overhead
Go:      1,080 ops x ~10 μs total = ~11 ms dispatch overhead
Saving:  ~5 ms per forward pass (~30% reduction in overhead)
```

**Per-op dispatch savings alone do not deliver the 3-10x speedup target.** The
bigger wins come from parallelism and fusion, described below.

---

## The GIL problem: parallelism

Python's structural problem goes beyond per-op overhead. The **Global
Interpreter Lock** prevents true parallel dispatch. Independent read heads
can't launch kernels simultaneously — only one thread holds the GIL at a time.

Go has no such constraint. Each head can be a goroutine dispatching kernels to
its own CUDA stream:

```
Python (sequential dispatch, one stream):
  Head1.step1 → Head1.step2 → ... → Head2.step1 → Head2.step2 → ...
  72 sequential dispatches, GPU cannot overlap heads

Go (parallel dispatch, 8 streams):
  Goroutine1 (stream1): Head1.step1 → Head1.step2 → ...
  Goroutine2 (stream2): Head2.step1 → Head2.step2 → ...
  ...
  Goroutine8 (stream8): Head8.step1 → Head8.step2 → ...

  GPU scheduler interleaves kernels from all streams across SMs
```

A GTX 1080 Ti has 28 SMs. A single stream of small kernels (32x256 matmul)
might occupy 1-2 SMs per kernel. Eight parallel streams can keep 8-16 SMs
busy simultaneously — a potential **4-8x improvement in GPU utilization**
compared to single-stream dispatch.

The Python workaround is folding heads into the batch dimension (B*8 instead
of B, with 9 dispatches instead of 72). This works but distorts the
architecture — reshaping tensors to fake parallelism, then decomposing to
recover per-head results. Goroutines express the actual algorithm: 8
independent heads running in parallel.

**This is the largest single source of speedup.** Not faster dispatch per op,
but actually using the GPU's parallel hardware.

---

## Kernel fusion: fewer launches, less memory traffic

Beyond dispatch overhead, a separate bottleneck affects small operations:
**memory bandwidth**.

A GPU's compute throughput vastly exceeds its memory bandwidth. Element-wise
operations (ReLU, sigmoid, add) are **bandwidth-bound** — nearly all time is
spent moving data to and from global memory, not computing. Each separate
kernel launch means: read from VRAM → compute → write to VRAM. The next kernel
reads that result back from VRAM.

Operator fusion combines adjacent operations into a single kernel, keeping
intermediate values in registers instead of round-tripping through VRAM:

```
Unfused (3 kernel launches, 6 VRAM accesses):
  [read A, B → matmul → write C] [read C → relu → write D] [read D → add bias → write E]

Fused (1 kernel launch, 2 VRAM accesses):
  [read A, B, bias → matmul + relu + add → write E]
```

This is where the graph engine's value is architectural, not just "faster
language." Fusion opportunities are visible at the graph level: a sequence of
`linear → relu → dropout` can become a single kernel, eliminating intermediate
memory round-trips.

torch.compile does this too, achieving 43% average training speedup on an A100.
But it requires Python tracing, breaks on dynamic control flow, and forces
recompilation when shapes or control flow change. A Go graph engine handles
fusion at the graph level without tracing or compilation.

---

## What torch.compile does (and its limits)

PyTorch recognized the overhead problem and built `torch.compile`. It traces
Python code, captures an operation graph, and optimizes it:

- Eliminates Python dispatch overhead by compiling to GPU code directly
- Fuses adjacent operations (halving memory round-trips in many cases)
- Achieves **43% average training speedup** on an A100 (up to 2x on
  transformers), with 99% graph capture rate across 163 tested models

But torch.compile has fundamental limits:

- **Data-dependent branching breaks the trace.** If a model decides where to
  look based on what it saw, torch.compile cannot capture that as a static
  graph.
- **Loop unrolling is static.** The compiler must know iteration counts at
  trace time. Adaptive computation (varying depth per input) forces
  recompilation.
- **Control flow between compiled regions falls back to eager mode** with full
  Python overhead.
- The 1% of models that fail graph capture includes dynamic control flow —
  exactly the architectures that need this framework most.

---

## What a Go graph engine adds beyond raw speed

The advantage is not just "Go is faster than Python." A Go-native graph engine
can do things torch.compile fundamentally cannot:

**Dynamic loop counts without recompilation.** The graph engine treats loops as
first-class constructs. Iteration count can vary per input without triggering
recompilation.

**Data-dependent branching with gradient routing.** Branch decisions are graph
nodes. The autograd engine records which branch was taken and routes gradients
only through the active path.

**Cross-branch kernel scheduling.** The graph engine sees all heads and their
dependency structure. It schedules kernels across CUDA streams optimally —
analyzing the dependency DAG rather than relying on one-goroutine-per-head.

**Graph-level operator fusion.** When the engine sees `linear → relu → linear →
relu`, it can fuse these into fewer kernel launches without Python tracing.

---

## Honest accounting: where the speedup comes from

Ranked by actual impact, not narrative convenience:

| Source | Impact | Mechanism |
|--------|--------|-----------|
| Goroutine parallelism (N streams) | **Largest** | Fills GPU SMs that single-stream dispatch leaves idle |
| Kernel fusion (graph engine) | **Large** | Fewer launches + eliminates VRAM round-trips |
| No Python dispatch overhead | **Moderate** | ~3-5 μs saved per op, compounds over 1000+ ops |
| No torch.compile recompilation | **Situational** | Matters for dynamic control flow architectures |
| libtorch overhead savings | **None** | Same library, same cost regardless of calling language |
| CUDA driver overhead savings | **None** | Hardware cost, unavoidable by any framework |

For **transformer-style** models (one huge attention matmul), Go's advantage is
marginal — the kernel is big enough that all overhead is noise. The speedup is
significant for **sequential, branching, adaptive** architectures where many
small kernels must fire in rapid succession across multiple independent paths.

### Could libtorch itself be bypassed?

In theory, calling cuBLAS/cuDNN directly would save ~0.5 μs per operation. In
practice, libtorch's value far exceeds this cost:
- Algorithm selection (which cuBLAS routine for a given shape?)
- Memory format handling (contiguous, channels-last, etc.)
- Thousands of pre-optimized kernel implementations
- cuDNN autotuning

Reimplementing this would take years for a marginal per-op saving. The ~0.5 μs
is a worthwhile trade.

---

## Note on numbers

The microsecond figures throughout this document are order-of-magnitude
estimates based on PyTorch internals documentation, CUDA programming guides,
and known CGo overhead measurements. They should be validated with
microbenchmarks on target hardware during Phase 1a development.
