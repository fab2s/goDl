# Visualizing Graphs

Every `graph.Graph` can export its structure as Graphviz DOT or SVG.
This is the fastest way to verify that your architecture is wired
correctly, especially when Using references, loops, and switches are
involved.

> **Prerequisites**: [The Graph Builder](05-graph-builder.md) and
> [Advanced Graphs](06-advanced-graphs.md) introduce the constructs
> shown in the diagrams.

## Generating output

Two methods on `*graph.Graph`:

```go
g, _ := graph.From(nn.MustLinear(4, 8)).
    Through(nn.NewGELU()).
    Also(nn.MustLinear(8, 8)).
    Through(nn.MustLinear(8, 2)).
    Build()

// DOT string — print, pipe, or paste into an online viewer.
fmt.Println(g.DOT())

// SVG file — requires the `dot` binary from Graphviz.
svg, err := g.SVG("model.svg")

// SVG bytes only — no file written.
svg, err := g.SVG()
```

`DOT()` always works. `SVG()` shells out to the Graphviz `dot` command,
so it returns an error if the binary is not found.

### Installing Graphviz

| OS     | Command                     |
|--------|-----------------------------|
| Ubuntu | `apt install graphviz`      |
| macOS  | `brew install graphviz`     |
| Alpine | `apk add graphviz`          |

If you cannot install Graphviz, paste the `DOT()` output into an online
viewer such as [GraphvizOnline](https://dreampuf.github.io/GraphvizOnline)
for quick iteration.

## Reading the diagrams

### Node shapes

| Shape           | Meaning                                         |
|-----------------|-------------------------------------------------|
| invhouse        | Input node (graph entry point)                  |
| house           | Output node (graph exit point)                  |
| doubleoctagon   | Node that is both input and output              |
| box             | Standard module (Linear, LayerNorm, Conv2d, ...) |
| ellipse         | Activation (GELU, ReLU, Sigmoid, Tanh, ...)     |
| box3d           | Loop (For, While, Until)                        |
| parallelogram   | Map (per-element processing)                    |
| diamond         | Switch router or state-read node                |
| circle          | Merge / add node                                |

### Colors

| Fill color   | Meaning                              |
|--------------|--------------------------------------|
| Blue         | Input nodes                          |
| Green        | Output nodes, normalization layers   |
| Yellow       | State-read nodes (forward refs)      |
| Purple       | Loop nodes                           |
| Orange       | Switch clusters                      |
| Light grey   | Standard modules (Linear, etc.)      |
| Peach        | Activations                          |
| Pink         | Dropout                              |
| Light blue   | Sub-graph (Graph-as-Module) nodes    |

### Node labels

Each node label shows:
1. The module type name (e.g. `Linear`, `GELU`, `Graph (sub)`)
2. Parameter count in brackets, formatted as K/M for readability
   (`[1.2K params]`, `[3.1M params]`)
3. Tag annotations as `#tagName` when the node has been tagged

Loop nodes additionally show their body and condition modules:

```
loop
body: Linear
cond: ThresholdHalt
[256 params]
```

### Edge styles

| Style                | Color   | Meaning                              |
|----------------------|---------|--------------------------------------|
| Solid                | Dark    | Normal data flow                     |
| Dashed               | Blue    | Using reference (backward ref)       |
| Dotted               | Orange  | Forward-ref state loop               |

Dashed Using edges are labeled with the reference name. Dotted state
edges are labeled `state:<tagName>`, showing where state flows from
writer back to reader across Forward() calls.

### Switch clusters

Switch nodes are expanded into sub-clusters showing their internal
structure:

- A **diamond** node for the router (entry point for incoming edges)
- A **box** per branch, labeled `[i] ModuleType` with parameter counts
- A small **circle** as the exit merge point
- Dashed orange internal edges from router to each branch, labeled with
  the branch index

### Execution levels

Nodes are grouped into `level 0`, `level 1`, etc. — dashed boxes that
show which nodes can execute in parallel. Nodes in the same level have
no data dependencies on each other and run concurrently via goroutines.

This makes it easy to spot parallelism: a Split with three branches
puts all three in the same level.

## Example: complex graph

```go
g, _ := graph.From(nn.MustLinear(4, 16)).Tag("input").
    Through(nn.NewGELU()).
    Split(
        nn.MustLinear(16, 16),
        nn.MustLinear(16, 16),
    ).Merge(graph.Mean()).
    Also(nn.MustLinear(16, 16)).Tag("residual").
    Through(nn.NewDropout(0.1)).
    Through(nn.MustLinear(16, 2)).
    Build()

fmt.Println(g.DOT())
```

The output shows:
- Level 0: the input Linear (invhouse shape, blue)
- Level 1: GELU (ellipse, peach)
- Level 2: both Split branches side by side (parallel execution)
- Level 3: Mean merge (circle)
- Level 4: Also with its residual Linear + the add node
- The `#input` and `#residual` annotations on their respective nodes
- Parameter counts on every Linear node

## Tips

- **Iterate with DOT first.** Paste into an online viewer while
  prototyping — no need to install Graphviz locally during design.
- **Check levels for parallelism.** If branches you expect to run in
  parallel end up in different levels, there is an unintended dependency.
- **Verify Using wires.** Dashed blue edges should connect the tagged
  node to the consuming node. Missing or misrouted edges indicate a
  builder bug.
- **State loops.** Dotted orange edges show forward-ref state. They
  should form cycles — from writer back to reader. If you do not see
  them, the forward ref was not resolved.
- **Large models.** For models with many nodes, write SVG and open in a
  browser — SVG scales cleanly and supports pan/zoom.

## What's next

[Utilities](08-utilities.md) covers gradient clipping, checkpoints,
parameter freezing, and weight initialization.
