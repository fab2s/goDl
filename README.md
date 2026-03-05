# goDl

A Go-native deep learning framework built from first principles on top of
libtorch.

goDl wraps libtorch's optimized C++/CUDA kernels with a Go-native tensor API,
autograd engine, and computation graph — eliminating Python's dispatch overhead
and GIL limitations while keeping the same battle-tested GPU math under the
hood.

The focus is on architectures that Python punishes: recurrent attention,
adaptive computation, hypothesis-test loops — anything with many small
sequential operations, branching control flow, or independent parallel paths.

## Why Go

### Why not just use Python/PyTorch?

Python adds ~3-5 μs of framework overhead to every GPU operation (interpreter,
GIL, argument parsing, dispatch chain). For large operations like a 1024x1024
matmul, this is noise. For architectures built on many small sequential
operations — recurrent steps, iterative refinement, multi-head attention with
independent heads — this overhead dominates. The GPU starves between kernel
launches.

Worse, Python's Global Interpreter Lock prevents parallel kernel dispatch.
Independent model branches (e.g., 8 attention heads) must dispatch kernels
sequentially from a single thread, even when the GPU has dozens of idle
Streaming Multiprocessors.

torch.compile partially addresses this by tracing and fusing operations, but
it breaks on data-dependent control flow and requires recompilation when loop
counts or branch structure change — exactly the dynamic architectures that
need help most.

See [docs/cuda-dispatch.md](docs/cuda-dispatch.md) for the full analysis.

### Why not C++?

C++ would have the least overhead, but writing and iterating on model
architectures in C++ is slow and error-prone. The goal is a framework that
researchers can productively experiment with, not a systems programming
exercise. Go provides compiled-language performance with a much shorter
feedback loop: fast compilation, simple tooling, readable code.

### Why not Rust?

Rust's main advantage — compile-time memory safety via the borrow checker — is
less relevant here. Tensor memory lives in libtorch's C allocator; the borrow
checker cannot reason about it. Both Go and Rust need the same kind of
scope-based or explicit lifecycle management at the FFI boundary.

Meanwhile, Go has concrete advantages for this domain:

- **Goroutines** are simpler than Rust's async model for parallel kernel
  dispatch across CUDA streams. `go func()` vs futures, pinning, Send/Sync
  bounds, and executor selection.
- **Compilation speed.** Go compiles in seconds; Rust in minutes. When
  iterating on model architectures, this compounds.
- **Readability.** DL researchers are not systems programmers. Go reads close
  to pseudocode. Rust's ownership model, lifetimes, and trait system are a
  steep barrier for a community that learned Python reluctantly.
- **Tooling simplicity.** `go test`, `go build`, `go vet` — no feature flags,
  no proc macro debugging, no async runtime choices.

A framework nobody uses solves nothing. Go hits the right trade-off between
performance and accessibility for the DL research community.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  User Code / Model Definitions                          │
├─────────────────────────────────────────────────────────┤
│  Graph Engine (composition, branching, loops, parallel)  │
├─────────────────────────────────────────────────────────┤
│  Layers & Optimizers (Linear, Conv, GRU, Adam, SGD)     │
├─────────────────────────────────────────────────────────┤
│  Autograd Engine (Go-native, reverse-mode AD)           │
├─────────────────────────────────────────────────────────┤
│  Tensor API (Go-idiomatic wrapper)                      │
├─────────────────────────────────────────────────────────┤
│  libtorch C Bindings (CGo)                              │
├─────────────────────────────────────────────────────────┤
│  libtorch / CUDA                                        │
└─────────────────────────────────────────────────────────┘
```

The same GPU kernels that power PyTorch run the actual math. goDl replaces
everything above them: the dispatch path, autograd tracking, operator
composition, and execution scheduling.

Since goDl binds libtorch — not CUDA directly — it inherits libtorch's backend
support: NVIDIA (CUDA), AMD (ROCm), Intel (XPU), Apple Silicon (MPS), and
CPU. Switching hardware is a build flag, not a code change.

## Getting Started

Requirements: Docker with NVIDIA Container Toolkit (for GPU support).

```bash
make image   # build the dev container (Go + libtorch + CUDA)
make test    # run all tests
make shell   # interactive shell in the container
```

See the [Makefile](Makefile) for all available commands.

## Status

Early development. See [docs/roadmap.md](docs/roadmap.md) for the full plan.

Currently developed and tested on a GTX 1060 6GB (Pascal, SM 6.1) — the
minimum NVIDIA GPU that supports CUDA compute capability 6.x. If this runs
well on a 1060, it will fly on anything modern. That said, testing on more
capable hardware would accelerate development significantly. 

If anyone feels like donating a modern GPU to the cause, the latent space would be eternally grateful.

## Documentation

- [Roadmap](docs/roadmap.md) — phased development plan with interface sketches
- [CUDA Dispatch Analysis](docs/cuda-dispatch.md) — detailed overhead breakdown
  and performance thesis
- [Bootstrap](docs/bootstrap.md) — origin story and research context

## License

MIT
