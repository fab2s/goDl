# The Graph Builder

The fluent graph builder is how you describe model architectures in goDl.
Instead of manually wiring layers together, you write data flow — what
happens to the tensor as it moves through the model.

By the end of this tutorial you'll be able to build models with linear
chains, parallel branches, residual connections, and per-element mapping.

> **Prerequisites**: familiarity with [Modules](03-modules.md) and
> [Training](04-training.md). You don't need to have read them — the
> code here is self-contained — but they explain the building blocks.

## Your first graph

```go
g, err := graph.From(nn.MustLinear(4, 8)).
    Through(nn.NewGELU()).
    Through(nn.MustLinear(8, 2)).
    Build()
```

`From` starts the flow. `Through` appends a module. `Build` finalizes
the graph and returns a `*graph.Graph` that implements `nn.Module` — it
has `Forward` and `Parameters` just like any other module.

> **Note for PyTorch users**: In Python you write
> `model = nn.Sequential(...)` and errors raise exceptions implicitly.
> In Go, `Build()` returns `(*graph.Graph, error)` — errors are explicit
> values you handle yourself. Throughout these tutorials you will see
> `g, err := ... Build()` in real code and `g, _ := ... Build()` in
> short examples where the `_` discards the error for brevity. In
> production code, always check `err`.

```go
input := autograd.NewVariable(inputTensor, true)
output := g.Forward(input)
loss := nn.MSELoss(output, target)
loss.Backward()
```

Gradients flow through the entire graph automatically.

## Residual connections with Also

`Also` adds a skip connection. The input passes through the module *and*
gets added to the module's output:

```go
g, _ := graph.From(nn.MustLinear(8, 8)).
    Through(nn.NewGELU()).
    Also(nn.MustLinear(8, 8)).      // output = input + Linear(input)
    Through(nn.MustLinear(8, 2)).
    Build()
```

This is the standard residual pattern from ResNet. The `Also` node
receives the stream, passes it to its module, and adds the result to
the original stream.

## Parallel branches with Split/Merge

`Split` sends the same input to multiple modules in parallel.
`Merge` combines their outputs:

```go
g, _ := graph.From(nn.MustLinear(4, 8)).
    Split(
        nn.MustLinear(8, 8),  // branch A
        nn.MustLinear(8, 8),  // branch B
        nn.MustLinear(8, 8),  // branch C
    ).Merge(graph.Mean()).     // average the three outputs
    Through(nn.MustLinear(8, 2)).
    Build()
```

Each branch has independent parameters. Branches at the same topological
level execute concurrently via goroutines — no extra code needed.

Built-in merge operations:
- `graph.Add()` — element-wise sum
- `graph.Mean()` — element-wise average
- `graph.Cat(dim)` — concatenate along a dimension

## Naming points with Tag

`Tag` names a point in the flow so you can reference it later:

```go
g, _ := graph.From(encoder).Tag("encoded").
    Through(transformer).
    Through(decoder).
    Build()
```

Tags are used by `Using`, `Gate`, `Switch`, and `Map.Over` to access
values from earlier in the graph. See
[Advanced Graphs](06-advanced-graphs.md) for the full story.

## Per-element processing with Map

`Map` applies a module to each element along dimension 0:

```go
// Process each item in the batch independently through a sub-network.
g, _ := graph.From(encoder).
    Map(nn.MustLinear(8, 8)).Each().
    Through(decoder).
    Build()
```

Three iteration modes:
- `.Each()` — iterate over current stream (dim 0)
- `.Over(tag)` — iterate over a tagged tensor
- `.Slices(n)` — decompose last dim into n slices, map, recompose

For stateless bodies (Linear, activations), add `.Batched()` to skip
element-by-element iteration and pass the full batch in one call:

```go
Map(nn.MustLinear(8, 8)).Batched().Each()  // much faster
```

## Sub-graphs as modules

Since `graph.Graph` implements `nn.Module`, you can use graphs as
building blocks inside other graphs:

```go
// Define a reusable block.
block, _ := graph.From(nn.MustLinear(8, 8)).
    Through(nn.NewGELU()).
    Through(nn.MustLayerNorm(8)).
    Build()

// Use it like any module.
model, _ := graph.From(nn.MustLinear(4, 8)).
    Through(block).           // sub-graph
    Also(block).              // another instance (shared params!)
    Through(nn.MustLinear(8, 2)).
    Build()
```

This is **Graph-as-Module** — the same pattern scales from small blocks
to entire model components.

## Putting it together

Here's a complete model that uses everything from this tutorial:

```go
// Reusable feed-forward block.
ffn := func(dim int64) *graph.Graph {
    g, _ := graph.From(nn.MustLinear(dim, dim)).
        Through(nn.NewGELU()).
        Through(nn.MustLayerNorm(dim)).
        Build()
    return g
}

// Main model.
model, _ := graph.From(nn.MustLinear(4, 16)).
    Through(nn.NewGELU()).
    Split(ffn(16), ffn(16)).Merge(graph.Mean()).  // multi-head
    Also(nn.MustLinear(16, 16)).                   // residual
    Through(nn.NewDropout(0.1)).
    Through(nn.MustLinear(16, 2)).
    Build()

// Train it.
optimizer := nn.NewAdam(model.Parameters(), 0.001)
model.SetTraining(true)

// ... training loop (see Tutorial 04) ...

// Evaluate.
model.SetTraining(false)
output := model.Forward(input)
```

## What's next

This tutorial covered the core builder methods. The
[Advanced Graphs](06-advanced-graphs.md) tutorial covers:
- **Forward references** — recurrent state across calls
- **Loops** — fixed, while, and until with BPTT
- **Gates** — soft routing with learned weights
- **Switches** — hard routing with selectors
- **Visualization** — DOT/SVG output
