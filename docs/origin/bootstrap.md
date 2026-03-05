# GoML Bootstrap — Seed Document for New Project

Copy this into the new repo as `docs/bootstrap.md` or use it to seed `CLAUDE.md` / memory.

---

## Origin Story

This framework exists because of a concrete problem discovered in the FBRL project (Feedback Recursive Loop — foveal attention research for vision). The core finding: **Python's dispatch overhead creates a selection bias in ML research**, steering the field toward architectures that happen to fit Python's strengths (big parallel matrix multiplies = transformers) and away from architectures that need fast sequential control flow (recurrent attention, iterative refinement, hypothesis-test loops).

### The evidence

FBRL builds vision models where a recurrent GRU controller decides *where to look* on an image, one fixation at a time. This is biologically plausible (human eye movements work this way) and architecturally interesting, but Python makes it painful:

- **v9.2 reading model**: 8 isolated read heads, each running 9 sequential GRU steps = 72 tiny CUDA kernel dispatches per forward pass. At batch_size=32, each kernel processes just 32 samples — GPU sits at 33% utilization, starving between Python-dispatched calls. Epoch time: 380s.
- **Batching workaround**: Folding all 8 heads into the batch dimension (B*8=256 per kernel) reduced dispatches from 72→9. GPU rose to 50%, epoch time dropped to 169s. But this is a hack — reshaping tensors to fake parallelism through the batch dimension, with decomposition loops to recover per-head results.
- **In Go**: each head would be a goroutine, each step a function call, libtorch handles CUDA. No reshape gymnastics. The sequential loop that Python punishes is just... code.

### Selection bias in research

Architectures that fit Python's strengths get explored deeply. Those that need fast sequential control flow get abandoned — not because they don't work, but because they're slow to iterate on:
- **Transformers** (one big parallel attention) → massively explored
- **Recurrent attention** (sequential fixation loops) → niche
- **Tree search during training** (MCTS-style) → almost unexplored
- **Adaptive computation** (variable depth per input) → rare
- **Hypothesis-test loops** (iterative refinement with branching) → nonexistent

The irony: biological vision/cognition is exactly the sequential, feedback-heavy process that Python punishes.

## The Author's Background

The framework author has deep experience in library/framework design from PHP open source work (github.com/fab2s):
- **NodalFlow** — a nodal workflow engine: composable nodes, branching, flow control. Essentially a computation graph for data processing.
- **YaEtl** — ETL framework built on NodalFlow. Extract/Transform/Load = structurally similar to data pipeline → model forward → loss backward.
- **dt0** — immutable DTOs, "8x faster than the alternative." Performance-conscious library design.

The pattern: clean abstractions, composable primitives, explicit data flow, performance at the library level. This maps directly to ML framework design.

## Core Design: Computation Graph

### Why this pattern

From building NodalFlow: "data flows through connected nodes" makes complex workflows easy to write, read, understand, and extend. A computation graph is the natural abstraction for ML:
- **Forward pass**: walk the graph, each node transforms its inputs
- **Backward pass**: walk the graph in reverse, each node computes gradients
- **Composition**: complex architectures are just graph topologies
- **Visualization**: the graph IS the architecture diagram

### Design principles

1. **Composable primitives**: small nodes that do one thing well, compose into complex flows
2. **Explicit data flow**: no hidden state, no magic — trace any tensor through the graph
3. **Type safety at the edges**: nodes declare their input/output tensor shapes
4. **Lazy execution**: build the graph first, execute later (enables optimization passes)
5. **Native loops**: recurrent steps are first-class graph constructs, not Python workarounds
6. **Parallel branches**: independent sub-graphs run concurrently via goroutines

### Why Go

- **Fast dispatch**: no interpreter overhead between operations
- **Goroutines**: natural parallelism for independent graph branches
- **Strong typing**: catch tensor shape mismatches at compile time
- **Zero-cost interfaces**: clean abstractions without runtime penalty
- **libtorch C API**: direct binding to CUDA kernels, no Python middleman
- **Cross-compilation**: single binary deployment, no conda environments
- **CGo**: mature C interop for libtorch bindings

## Architecture Sketch

```go
// Core interface — any operation that transforms tensors
type Node interface {
    Forward(ctx *Context, inputs ...Tensor) []Tensor
    Backward(ctx *Context, gradOutputs ...Tensor) []Tensor
}

// Graph connects nodes with typed edges
type Graph struct {
    nodes []Node
    edges []Edge  // source node/port -> dest node/port
}

// A GRU cell is just a node
type GRUCell struct {
    Wz, Wr, Wh Linear
    hiddenDim  int
}

func (g *GRUCell) Forward(ctx *Context, inputs ...Tensor) []Tensor {
    x, hPrev := inputs[0], inputs[1]
    // ... GRU math
    return []Tensor{hNext}
}

// A reading head is a sub-graph with loops
func NewReadingHead(searchSteps, readSteps int) *Graph {
    g := NewGraph()
    sensor := g.Add(NewPeripheralSensor(20, 28, 2.0))
    gru := g.Add(NewGRUCell(256))
    locHead := g.Add(NewLinear(256, 2))
    g.Loop(searchSteps, sensor, gru, locHead)
    readSensor := g.Add(NewGlimpseSensor(12))
    g.Loop(readSteps, readSensor, gru, locHead)
    return g
}

// 8 heads in parallel — goroutines, not batch-dimension hacks
func NewReadingModel() *Graph {
    g := NewGraph()
    heads := make([]*Graph, 8)
    for i := range heads {
        heads[i] = NewReadingHead(2, 6)
    }
    g.Parallel(heads...)
    return g
}
```

## Implementation Phases

### Phase 1: Tensor + Autograd
The foundation. Must be solid before anything else.
- Go wrapper around libtorch C API (tensor creation, basic ops, CUDA device management)
- Autograd: track operations on tensors, build backward graph, compute gradients
- Key ops: matmul, conv2d, add, relu, tanh, sigmoid, softmax, grid_sample
- Memory management: Go GC + explicit tensor lifecycle (Release/Retain)

### Phase 2: Core Layers
- Linear, Conv2d, GRUCell, BatchNorm
- Activation functions (ReLU, Tanh, Sigmoid)
- Loss functions (CrossEntropy, BCE, MSE)
- Optimizers (Adam, SGD with momentum)
- Parameter management (named parameters, state_dict equivalent)

### Phase 3: Graph Engine
- Node composition, graph building, topological execution
- Parallel execution of independent branches via goroutines
- Loop constructs for recurrent architectures
- Conditional execution (if-then-else in the graph)
- Checkpoint/save/load

### Phase 4: Prove It
- Port FBRL reading model from Python to Go
- Direct comparison: same architecture, same data, same GPU
- Measure: epoch time, GPU utilization, iteration speed
- Target: 3-5x speedup over Python (batched version), potentially 10x over naive Python

## The Target Architecture: Hypothesis-Verification Reading

The ultimate test case for the framework — an architecture that's practically impossible in Python but natural in Go.

### What FBRL v9.2 taught us

Fully isolated read heads (no shared state, no cheating) achieved only 7.9% letter accuracy (2x random). The problem: cold-start heads can't find letters, and even when they do, local reads can't distinguish ambiguous shapes (O vs ll look identical locally).

### How humans actually read

Research shows we can read words with scrambled inner letters ("you can raed tihs"). The brain reads *word shapes*, not individual letters:

1. **Peripheral**: detect word envelope (length, ascender/descender pattern)
2. **Parafoveal**: recognize word shape — often enough for common words
3. **Foveal**: only fixate individual letters for unfamiliar words or ambiguity

The focus/unfocus cycle is **recognition at multiple scales**. Unfocused pass → hypothesis ("this looks like 'hello'"). Focused pass → verify or correct ("wait, that's 'hallo'").

### The architecture this implies

```
META-SCAN (blurred) -> word shape embedding
  "5-letter word, ascender at position 2"

HYPOTHESIS -> candidate word(s) from shape

VERIFICATION (sharp reads) -> confirm/disambiguate
  Only fixate where the hypothesis is uncertain
  Loop until confidence threshold met
```

This is:
- **Sequential**: each step depends on the previous
- **Branching**: where to look depends on what was seen
- **Adaptive depth**: different words need different numbers of fixations
- **Data-dependent control flow**: the graph topology changes per input

Python can't express this efficiently. Go can.

## Key FBRL Results (for reference)

| Experiment | Result | Insight |
|-----------|--------|---------|
| Letters v7 | 100%/100% | Foveal attention works. 7 glimpses, self-scaffolding |
| Letters v8 | 100%/100% | 1 scan + 8 read = 9 glimpses, robust |
| Words v1 | 100% all positions | Prescribed scan + flat read |
| Reading v9.1 | 98.6% letter | Shared GRU cheats — classifies from bridge, not reads |
| Reading v9.2 | 7.9% letter | Isolated heads honest but underpowered |
| Motor v5 | 13.6% test | Re-read via classifier = adversarial, not writing |

### Key design principles from FBRL
- **Canvas geometry enforces strategy**: fovea-to-canvas ratio determines if model can cheat
- **Self-scaffolding**: multi-task losses naturally sequence by difficulty (easy locks in first, scaffolds hard)
- **Void repulsion > center pull**: local "don't stare at nothing" beats global guide
- **Blur prevents shortcutting**: if a sensor can resolve details, the model will bypass downstream processing
- **Isolation proves honesty**: removing shared state reveals whether the model actually uses its read phase

## Existing Go ML Landscape

Before building, evaluate what exists:
- **gotch** — Go bindings for libtorch (most mature, worth studying)
- **gorgonia** — Pure Go ML library (no libtorch dependency, custom autograd)
- **gonum** — Numerical computing in Go (not ML-specific but useful primitives)

The gap: none of these have a computation graph engine designed for recurrent/sequential architectures with native loop and branch constructs. That's the unique value proposition.

## Environment

- **GPU**: GTX 1080 Ti (Pascal, 11GB VRAM) — the development GPU
- **OS**: WSL2/Linux
- **Docker**: for GPU access (same as FBRL setup)
- **FBRL repo**: `/home/peta/src/fab2s/ai/fbrl` — reference implementation in Python/PyTorch
