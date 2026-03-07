<p align="center">
  <img src="docs/goDl.png" alt="goDl" width="320">
</p>

<h1 align="center">goDl</h1>

<p align="center">
A Go-native deep learning framework built on libtorch.<br>
Same GPU kernels as PyTorch. No Python. No GIL. Just Go.
</p>

<p align="center">
  <a href="https://github.com/fab2s/goDl/actions/workflows/ci.yml"><img src="https://github.com/fab2s/goDl/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

<p align="center">
  <a href="#the-graph-builder">Graph Builder</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="docs/tutorials/01-tensors.md">Tutorials</a> &bull;
  <a href="#architecture">Architecture</a>
</p>

---

## The Graph Builder

goDl's fluent graph builder lets you describe complex architectures as
readable data flow — no boilerplate, no graph construction commands.

```go
model, err := graph.From(nn.MustLinear(2, 16)).   // input projection
    Through(nn.NewGELU()).                          // activation
    Through(nn.MustLayerNorm(16)).                  // normalization
    Also(nn.MustLinear(16, 16)).                    // residual connection
    Through(nn.MustLinear(16, 2)).                  // output projection
    Build()
```

That's a trainable model. `Also` adds the residual — input flows through the
Linear *and* gets added to its output. `Build()` returns a `graph.Graph` that
implements `nn.Module` — you can nest it inside other graphs.

Things get interesting when architectures get complex:

```go
g, err := graph.From(encoder).Tag("encoded").                // tag for later
    Split(headA, headB, headC).Merge(graph.Mean()).          // multi-head + merge
    Loop(refinementBlock).For(3).Tag("refined").             // iterate 3 times
    Gate(router, expertA, expertB).Using("encoded").         // soft routing with context
    Switch(selector, lightPath, heavyPath).Using("refined"). // hard routing
    Through(graph.StateAdd()).Using("memory").Tag("memory").  // recurrent state
    Loop(decoder).While(haltCondition, 10).                  // adaptive computation
    Through(outputHead).
    Build()
```

Every construct — `Split/Merge`, `Also`, `Loop`, `Gate`, `Switch`, `Map`,
`Tag/Using` — composes cleanly. Sub-graphs nest like any module. Forward
references (`Using` before `Tag`) carry state across calls, enabling recurrent
architectures without special-casing.

The graph executes nodes at the same topological level in parallel via
goroutines — independent branches run concurrently without any extra code.

See the **[Graph Builder Tutorial](docs/tutorials/05-graph-builder.md)** and
the [full showcase](examples/showcase/showcase.go) that exercises every builder
method.

## Quick Start

Requirements: Docker (with NVIDIA Container Toolkit for GPU support).

```bash
git clone https://github.com/fab2s/goDl.git
cd goDl
make image    # build dev container (Go + libtorch + CUDA)
make test     # run all 276 tests (CPU + CUDA)
make test-cpu # run without GPU
make shell    # interactive shell in container
```

### Train a model in 30 lines

```go
// Task: learn cumulative sum — [a, b] → [a, a+b]

// Build the model.
model, err := graph.From(nn.MustLinear(2, 16)).
    Through(nn.NewGELU()).
    Through(nn.MustLayerNorm(16)).
    Also(nn.MustLinear(16, 16)).
    Through(nn.MustLinear(16, 2)).
    Build()

// Set up training.
optimizer := nn.NewAdam(model.Parameters(), 0.01)
model.SetTraining(true)

// Training loop.
for loader.Next() {
    input, target := loader.Batch()

    pred := model.Forward(autograd.NewVariable(input, true))
    loss := nn.MSELoss(pred, autograd.NewVariable(target, false))

    optimizer.ZeroGrad()
    loss.Backward()
    nn.ClipGradNorm(model.Parameters(), 1.0)
    optimizer.Step()
}
```

See [`examples/train/`](examples/train/train_test.go) for the complete
runnable version with data generation and evaluation.

## Features

### Core Stack

| Layer | What it does |
|-------|-------------|
| **Tensor** | Immutable, chainable API with error propagation. CPU and CUDA. |
| **Autograd** | Reverse-mode automatic differentiation. Full backward for every op. |
| **NN Modules** | `Linear`, `Conv2d`, `LayerNorm`, `BatchNorm`, `Dropout`, `Embedding`, `GRUCell`, `LSTMCell` |
| **Activations** | `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`, `Softmax` |
| **Losses** | `MSELoss`, `CrossEntropyLoss` |
| **Optimizers** | `SGD` (with momentum), `Adam`, `AdamW` |
| **LR Scheduling** | `StepDecay`, `Cosine`, `Warmup` (composable), `ReduceOnPlateau` |
| **Mixed Precision** | `Float16`/`BFloat16` dtype casting, `GradScaler` for loss scaling |

### Graph Builder

| Method | What it does |
|--------|-------------|
| `From(m).Through(m)` | Linear chain |
| `Split(m...).Merge(op)` | Parallel branches, merged by `Add()`, `Mean()`, or `Cat(dim)` |
| `Also(m)` | Residual connection: `input + m(input)` |
| `Tag(name)` / `Using(refs...)` | Named references — backward (same pass) or forward (across calls) |
| `Loop(body).For(n)` | Fixed iteration with BPTT |
| `Loop(body).While(cond, max)` | Condition before body (0..max iterations) |
| `Loop(body).Until(cond, max)` | Condition after body (1..max iterations) |
| `Gate(router, experts...)` | Soft routing — all experts execute, weighted combination |
| `Switch(selector, branches...)` | Hard routing — only selected branch executes |
| `Map(body).Each()` | Apply body to each element along dim 0 |
| `Map(body).Over(tag)` | Iterate over a tagged tensor |
| `Map(body).Slices(n)` | Decompose last dim into n slices, map, recompose |
| `.Batched()` | Fast path for Map — full batch in one call |

### Training Tools

| Tool | What it does |
|------|-------------|
| `nn.ClipGradNorm` | L2 norm gradient clipping |
| `nn.ClipGradValue` | Element-wise gradient clamping |
| `g.Freeze(tags...)` / `g.Unfreeze(tags...)` | Freeze parameters by tag name |
| `nn.SaveParameters` / `nn.LoadParameters` | Binary checkpoint format |
| `KaimingUniform/Normal`, `XavierUniform/Normal` | Weight initialization |
| `data.Loader` | Batched data loading with parallel prefetch and shuffle |
| LR schedulers | `StepDecay`, `Cosine`, `Warmup`, `ReduceOnPlateau` (composable) |
| `nn.GradScaler` | Dynamic loss scaling for mixed precision (float16) training |
| `nn.CastParameters` | Cast model parameters to any dtype (`Float16`, `BFloat16`, etc.) |

### Visualization

```go
fmt.Println(g.DOT())          // Graphviz DOT with parameter counts
svg, _ := g.SVG("model.svg")  // render to SVG
```

Node shapes indicate type (input, output, loop, map, switch, activation,
normalization). Parameter counts appear on each node. Forward-ref state
loops are shown as dotted edges.

### Numerical Verification

Every differentiable path is verified against finite-difference gradients:
- 32 autograd op-level checks (every op + compositions)
- 10 module-level checks (every NN module, input + parameter gradients)
- 11 exact optimizer step verifications (SGD, Adam, AdamW)
- 276 tests total, all passing with race detector

## Why Go for Deep Learning?

### The dispatch overhead problem

Python adds ~3-5 us of framework overhead to every GPU operation (interpreter,
GIL, argument parsing, dispatch chain). For large operations like a 1024x1024
matmul, this is noise. For architectures built on many small sequential
operations — recurrent steps, iterative refinement, multi-head attention with
independent heads — this overhead dominates. The GPU starves between kernel
launches.

Python's Global Interpreter Lock prevents parallel kernel dispatch. Independent
model branches must dispatch kernels sequentially from a single thread, even
when the GPU has dozens of idle Streaming Multiprocessors.

`torch.compile` partially addresses this by tracing and fusing operations, but
it breaks on data-dependent control flow and requires recompilation when loop
counts or branch structure change — exactly the dynamic architectures that
need help most.

### Why Go and not C++ or Rust?

**Not C++** because writing and iterating on model architectures in C++ is slow
and error-prone. Go provides compiled-language performance with a much shorter
feedback loop: fast compilation, simple tooling, readable code.

**Not Rust** because Rust's main advantage — the borrow checker — cannot reason
about tensor memory in libtorch's C allocator. Meanwhile Go has concrete
advantages: goroutines are simpler than async for parallel dispatch, compilation
is seconds not minutes, the code reads close to pseudocode, and the tooling
(`go test`, `go build`, `go vet`) just works.

A framework nobody uses solves nothing. Go hits the right trade-off between
performance and accessibility for this domain.

See [docs/design/cuda-dispatch.md](docs/design/cuda-dispatch.md) for the full
dispatch overhead analysis.

## Architecture

```
+-----------------------------------------------------------+
|  User Code / Model Definitions                            |
+-----------------------------------------------------------+
|  graph/    Fluent builder, parallel execution, DOT/SVG    |
+-----------------------------------------------------------+
|  nn/       Modules, losses, optimizers, checkpoints       |
+-----------------------------------------------------------+
|  autograd/ Reverse-mode AD, gradient tracking             |
+-----------------------------------------------------------+
|  tensor/   Immutable chainable API, CPU + CUDA            |
+-----------------------------------------------------------+
|  internal/libtorch/   CGo bindings to libtorch C++        |
+-----------------------------------------------------------+
|  libtorch / CUDA / ROCm / MPS / CPU                      |
+-----------------------------------------------------------+
```

The same GPU kernels that power PyTorch run the actual math. goDl replaces
everything above them: the dispatch path, autograd tracking, operator
composition, and execution scheduling.

Since goDl binds libtorch — not CUDA directly — it inherits libtorch's backend
support: NVIDIA (CUDA), AMD (ROCm), Intel (XPU), Apple Silicon (MPS), and
CPU. Switching hardware is a build flag, not a code change.

## Documentation

### Tutorials

Step-by-step guides from basics to advanced, each with runnable examples:

1. **[Tensors](docs/tutorials/01-tensors.md)** — creation, ops, chaining, error handling
2. **[Autograd](docs/tutorials/02-autograd.md)** — variables, gradients, backward pass
3. **[Modules](docs/tutorials/03-modules.md)** — Linear, Conv2d, normalization, RNN cells
4. **[Training](docs/tutorials/04-training.md)** — losses, optimizers, data loading, full loop
5. **[Graph Builder](docs/tutorials/05-graph-builder.md)** — the fluent API from simple to complex
6. **[Advanced Graphs](docs/tutorials/06-advanced-graphs.md)** — forward refs, loops, gates, switches
7. **[Visualization](docs/tutorials/07-visualization.md)** — DOT/SVG output, reading diagrams
8. **[Utilities](docs/tutorials/08-utilities.md)** — checkpoints, clipping, freezing, initialization

### Design

- [Roadmap](docs/design/roadmap.md) — phased development plan
- [CUDA Dispatch Analysis](docs/design/cuda-dispatch.md) — overhead breakdown and performance thesis
- [Trajectory Thesis](docs/design/trajectory-thesis.md) — geometric intuition behind the project

### Examples

- [`examples/train/`](examples/train/) — complete training loop (data loading, loss, backward, optimizer)
- [`examples/showcase/`](examples/showcase/) — every graph builder method in one graph

## License

goDl is open-sourced software licensed under the [MIT license](./LICENSE).
