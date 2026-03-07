# Training Utilities

This tutorial covers the utilities that sit around the training loop:
gradient clipping, checkpointing, parameter freezing, and weight
initialization.

> **Prerequisites**: [Training](04-training.md) introduces the
> Backward/Step loop. [The Graph Builder](05-graph-builder.md)
> introduces Tags.

## Gradient clipping

Deep models — especially those with loops (While, Until) or long
chains — can suffer from exploding gradients. goDl provides two
clipping strategies. Both are called between `Backward()` and
`optimizer.Step()`.

### ClipGradNorm

Scales all parameter gradients so that the total L2 norm does not
exceed `maxNorm`. Returns the original norm before clipping.

```go
loss.Backward()
origNorm := nn.ClipGradNorm(model.Parameters(), 1.0)
optimizer.Step()
```

If the total gradient norm is already within `maxNorm`, nothing
changes. When it exceeds the limit, every gradient is multiplied by
`maxNorm / totalNorm`, preserving direction while bounding magnitude.

### ClipGradValue

Clamps each individual gradient element to `[-maxVal, maxVal]`.
Returns the maximum absolute gradient value before clipping.

```go
loss.Backward()
maxAbs := nn.ClipGradValue(model.Parameters(), 0.5)
optimizer.Step()
```

Use `ClipGradNorm` as the default — it is less aggressive and preserves
gradient direction. Switch to `ClipGradValue` when you need a hard
per-element bound.

## Checkpoints

Save and restore model parameters with a compact binary format.
The format stores a magic header (`GODL`), version number, parameter
count, then for each parameter: name, shape, and float32 data.

### Saving

```go
f, err := os.Create("model.bin")
if err != nil {
    log.Fatal(err)
}
if err := nn.SaveParameters(f, model.Parameters()); err != nil {
    log.Fatal(err)
}
f.Close()
```

### Loading

```go
f, err := os.Open("model.bin")
if err != nil {
    log.Fatal(err)
}
if err := nn.LoadParameters(f, model.Parameters()); err != nil {
    log.Fatal(err)
}
f.Close()
```

`LoadParameters` validates that the parameter count, names, and shapes
match exactly. If the model architecture has changed since the
checkpoint was written, loading fails with a descriptive error.

### Details

- Parameters are always serialized as float32, regardless of the
  runtime dtype. This means checkpoints are portable across CPU and
  CUDA.
- Parameters are matched positionally — the model that loads must
  produce `Parameters()` in the same order as the model that saved.
- The `io.Writer` / `io.Reader` interface means you can write to any
  destination: files, buffers, network connections.

### Periodic checkpoints during training

```go
for epoch := 0; epoch < numEpochs; epoch++ {
    // ... training loop ...

    // Save every 10 epochs.
    if (epoch+1)%10 == 0 {
        f, _ := os.Create(fmt.Sprintf("checkpoint_epoch_%d.bin", epoch+1))
        nn.SaveParameters(f, model.Parameters())
        f.Close()
    }
}
```

## Parameter freezing

When fine-tuning, you often want to freeze part of a model (no gradient
updates) while training the rest. Graph-level freezing works through
Tags — name the nodes you want to control, then freeze or unfreeze by
tag.

### Setup

Tag the nodes when building the graph:

```go
g, _ := graph.From(nn.MustLinear(4, 16)).Tag("encoder").
    Through(nn.NewGELU()).
    Through(nn.MustLinear(16, 16)).Tag("middle").
    Through(nn.NewGELU()).
    Through(nn.MustLinear(16, 2)).Tag("head").
    Build()
```

### Freezing and unfreezing

```go
// Freeze the encoder — its parameters won't be updated.
g.Freeze("encoder")

// Freeze multiple tags at once.
g.Freeze("encoder", "middle")

// Unfreeze later for full fine-tuning.
g.Unfreeze("encoder")
```

### Training loop with frozen parameters

Freezing does not prevent gradient computation (upstream layers still
need gradients to flow through). Instead, `ZeroFrozenGrads` zeroes out
the frozen parameter gradients before the optimizer step:

```go
optimizer := nn.NewAdam(g.Parameters(), 0.001)

for epoch := 0; epoch < numEpochs; epoch++ {
    optimizer.ZeroGrad()
    output := g.Forward(input)
    loss := nn.MSELoss(output, target)
    loss.Backward()

    g.ZeroFrozenGrads()   // zero frozen grads before step
    optimizer.Step()
}
```

The order matters: `Backward()` first (computes all gradients),
`ZeroFrozenGrads()` next (zeroes the ones you want frozen), then
`Step()` (applies the remaining non-zero gradients).

### Accessing parameters by tag

```go
encoderParams := g.ParametersByTag("encoder")
fmt.Printf("Encoder has %d parameters\n", len(encoderParams))
```

This returns the parameters of the node at the given tag, or nil if the
tag does not exist.

## Weight initialization

goDl modules use sensible defaults — `Linear` initializes weights with
Kaiming uniform (suitable for ReLU) and bias with uniform. But you can
override this when needed.

### Built-in initializers

| Function                                  | Distribution             | Best for              |
|-------------------------------------------|--------------------------|-----------------------|
| `nn.KaimingUniform(shape, fanIn)`         | U(-bound, bound)         | ReLU activations      |
| `nn.KaimingNormal(shape, fanIn)`          | N(0, std)                | ReLU activations      |
| `nn.XavierUniform(shape, fanIn, fanOut)`  | U(-bound, bound)         | Sigmoid / Tanh        |
| `nn.XavierNormal(shape, fanIn, fanOut)`   | N(0, std)                | Sigmoid / Tanh        |

Where:
- Kaiming: `bound = sqrt(6 / fanIn)`, `std = sqrt(2 / fanIn)`
- Xavier: `bound = sqrt(6 / (fanIn + fanOut))`, `std = sqrt(2 / (fanIn + fanOut))`

### Custom initialization

Replace parameter data after constructing the module. Every `Parameter`
embeds `*autograd.Variable`, which has `SetData`:

```go
layer, _ := nn.NewLinear(128, 64)

// Re-initialize weight with Xavier normal (e.g., for a Tanh layer).
w, _ := nn.XavierNormal([]int64{64, 128}, 128, 64)
layer.Parameters()[0].SetData(w)
```

### Initializing all layers in a model

Walk `Parameters()` and replace data based on naming conventions:

```go
for _, p := range model.Parameters() {
    shape := p.Data().Shape()
    if len(shape) == 2 {
        // Weight matrix: use Xavier.
        fanIn, fanOut := shape[1], shape[0]
        w, _ := nn.XavierUniform(shape, fanIn, fanOut)
        p.SetData(w)
    }
    // Bias vectors (1-D) keep their default init.
}
```

### When to change initialization

- **Default (Kaiming uniform)** works well for ReLU-based networks.
  No action needed for most models.
- **Xavier** is better when using Sigmoid or Tanh activations — it
  accounts for both fan-in and fan-out to keep variance stable.
- **Custom** initialization matters most for very deep networks or
  unusual architectures where the default leads to vanishing or
  exploding activations at init time.

## Trend-based training control

After collecting metrics and flushing epoch summaries (see
[Training — Observing Training](04-training.md#observing-training)),
query trends to make training decisions.

### Trend queries

Each `Flush` adds one data point (the epoch mean) to the tag's
history. `Trend` returns a queryable view over that history:

```go
trend := g.Trend("loss")
trend.Mean()              // mean across all flushed epochs
trend.Slope(5)            // OLS slope over last 5 epochs
trend.Improving(5)        // is slope negative? (loss decreasing)
trend.Stalled(5, 1e-4)    // is |slope| below tolerance?
trend.Converged(5, 1e-5)  // is variance below tolerance?
```

All query methods accept a window parameter: positive values limit the
analysis to the last N data points, zero or negative uses all.

### LR decay on plateau

```go
if g.Trend("loss").Stalled(10, 1e-4) {
    currentLR := optimizer.LR()
    optimizer.SetLR(currentLR * 0.1)
}
```

### Conditional freezing

Unfreeze a section of the model only when training shows progress:

```go
if g.Trend("loss").Improving(3) {
    g.Unfreeze("decoder")
}
```

### Early stopping

```go
if g.Trend("loss").Converged(5, 1e-5) {
    break
}
```

### Sub-graph observation

When a graph contains sub-graphs (Graph-as-Module), use `Sub` to
observe their internal tags. The sub-graph's Forward runs automatically
as part of the parent — no separate Forward call needed:

```go
// Build a sub-graph with its own tags.
refiner, _ := graph.From(step).Tag("residual").Build()

// Compose into root graph.
g, _ := graph.From(encoder).
    Through(refiner).Tag("refine").
    Through(lossModule).Tag("loss").
    Build()

// Training loop — only Forward on root.
for epoch := range epochs {
    for loader.Next() {
        g.Forward(input)
        g.Collect("loss")                    // root tag
        g.Sub("refine").Collect("residual")  // sub-graph tag
    }
    g.Flush()
    g.Sub("refine").Flush()
}
```

### ResetTrend

Clear epoch history when you need a fresh start (e.g. after unfreezing
a section of the model and wanting to track the new training phase):

```go
g.Unfreeze("encoder")
g.ResetTrend("loss")    // clear history for one tag
g.ResetTrend()          // or clear all
```

## Putting it together

A complete fine-tuning script using all utilities — clipping, freezing,
checkpoints, initialization, and trend-based training control:

```go
// Build model with tagged sections.
g, _ := graph.From(nn.MustLinear(4, 64)).Tag("encoder").
    Through(nn.NewGELU()).
    Through(nn.MustLinear(64, 64)).Tag("body").
    Through(nn.NewGELU()).
    Through(nn.MustLinear(64, 2)).Tag("head").
    Build()

// Load pretrained weights.
f, _ := os.Open("pretrained.bin")
nn.LoadParameters(f, g.Parameters())
f.Close()

// Re-initialize the head with Xavier (switching to Tanh downstream).
headParams := g.ParametersByTag("head")
if len(headParams) > 0 {
    shape := headParams[0].Data().Shape()
    w, _ := nn.XavierNormal(shape, shape[1], shape[0])
    headParams[0].SetData(w)
}

// Freeze encoder, train body + head.
g.Freeze("encoder")
optimizer := nn.NewAdam(g.Parameters(), 0.001)
g.SetTraining(true)

// Log epoch summaries.
g.OnFlush(func(flushed map[string]float64) {
    fmt.Printf("epoch loss: %.6f\n", flushed["head"])
})

for epoch := 0; epoch < 100; epoch++ {
    loader.Reset()
    for loader.Next() {
        inT, tgtT := loader.Batch()
        input := autograd.NewVariable(inT, true)
        target := autograd.NewVariable(tgtT, false)

        optimizer.ZeroGrad()
        output := g.Forward(input)
        loss := nn.MSELoss(output, target)
        loss.Backward()

        nn.ClipGradNorm(g.Parameters(), 1.0)
        g.ZeroFrozenGrads()
        optimizer.Step()

        g.Collect("head")
    }
    g.Flush()

    // Unfreeze encoder once loss is improving.
    if epoch > 5 && g.Trend("head").Improving(3) {
        g.Unfreeze("encoder")
    }

    // Early stop when converged.
    if g.Trend("head").Converged(5, 1e-5) {
        break
    }
}

// Save fine-tuned model.
g.SetTraining(false)
f, _ = os.Create("finetuned.bin")
nn.SaveParameters(f, g.Parameters())
f.Close()
```

---

Previous tutorials: [07-Visualization](07-visualization.md) |
[06-Advanced Graphs](06-advanced-graphs.md) |
[05-Graph Builder](05-graph-builder.md) |
[04-Training](04-training.md) |
[03-Modules](03-modules.md) |
[02-Autograd](02-autograd.md) |
[01-Tensors](01-tensors.md)
