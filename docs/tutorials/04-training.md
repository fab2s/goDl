# Tutorial 4: Training

This tutorial puts everything together: loss functions, optimizers, data loading, and the training loop. It builds on [Tutorial 3: Modules](03-modules.md).

## Loss Functions

```go
// Mean Squared Error: mean((pred - target)^2)
// Both inputs must have the same shape. Returns a scalar.
loss := nn.MSELoss(pred, target)

// Cross-Entropy from raw logits (not probabilities).
// pred: [batch, classes] logits. target: [batch, classes] one-hot.
// Uses log-sum-exp trick for numerical stability.
loss := nn.CrossEntropyLoss(logits, oneHotTarget)
```

Both return a scalar `*autograd.Variable` ready for `Backward()`.

## Optimizers

All optimizers implement the `Optimizer` interface:

```go
type Optimizer interface {
    Step()      // apply one parameter update
    ZeroGrad()  // reset all parameter gradients
}
```

### SGD

```go
// Vanilla SGD (momentum=0)
opt := nn.NewSGD(model.Parameters(), 0.01, 0)

// SGD with momentum
opt := nn.NewSGD(model.Parameters(), 0.01, 0.9)
```

### Adam

```go
// Default betas (0.9, 0.999), eps=1e-8
opt := nn.NewAdam(model.Parameters(), 0.001)
```

### AdamW

Adam with decoupled weight decay (Loshchilov & Hutter, 2017):

```go
// weightDecay=0.01 is typical
opt := nn.NewAdamW(model.Parameters(), 0.001, 0.01)
```

## Gradient Clipping

Prevent exploding gradients by clipping after backward and before the optimizer step:

```go
// Scale gradients so total L2 norm <= maxNorm
nn.ClipGradNorm(model.Parameters(), 1.0)

// Clamp each gradient element to [-maxVal, maxVal]
nn.ClipGradValue(model.Parameters(), 0.5)
```

## Data Loading

### Dataset

The `Dataset` interface provides random access to individual samples:

```go
type Dataset interface {
    Len() int
    Get(index int) (input, target *tensor.Tensor, err error)
}
```

`TensorDataset` wraps pre-loaded batched tensors. Each sample is a slice along dimension 0:

```go
inputs, _ := tensor.FromFloat32(data, []int64{numSamples, featureDim})
targets, _ := tensor.FromFloat32(labels, []int64{numSamples, outputDim})
ds := data.NewTensorDataset(inputs, targets)
```

### Loader

`Loader` iterates over a dataset in batches using a scanner-style API:

```go
loader := data.NewLoader(ds, data.LoaderConfig{
    BatchSize:  32,
    Shuffle:    true,
    NumWorkers: 0,     // 0 = sequential, >0 = parallel with goroutines
    PrefetchN:  0,     // batches buffered ahead (for parallel mode)
    DropLast:   false,  // drop incomplete final batch
})
defer loader.Close()

for loader.Next() {
    input, target := loader.Batch()
    // ... training step ...
}
if err := loader.Err(); err != nil {
    log.Fatal(err)
}

loader.Reset()  // start a new epoch (reshuffles if Shuffle=true)
```

## Device Placement

By default, all tensors and parameters live on CPU. To train on CUDA,
use `SetDevice` on the graph and optionally `Device` on the loader.

### Moving the model

`SetDevice` moves all parameters and state buffers to the target
device. It uses `WalkModules` to recurse into sub-graphs, composite
modules (loops, switches, maps), and any user module implementing
`SubModuler` — automatically reaching nested `DeviceMover` modules
like `BatchNorm`.

```go
model, _ := buildModel()

if tensor.CUDAAvailable() {
    model.SetDevice(tensor.CUDA)
}

// Create optimizer AFTER SetDevice — optimizer state tensors are
// allocated lazily on first Step, matching the parameter device.
optimizer := nn.NewAdam(model.Parameters(), 0.001)
```

### Moving data

The loader can move batches to a device after stacking:

```go
loader := data.NewLoader(ds, data.LoaderConfig{
    BatchSize: 32,
    Shuffle:   true,
    Device:    tensor.DevicePtr(tensor.CUDA), // both input and target
})
```

When `Device` is nil (the default), no move happens — existing code
is unaffected.

### Auto-move inputs at graph entry

If the graph has a device set via `SetDevice` and a Forward input is
on a different device, the graph moves it automatically. This is
useful for ad-hoc inference without a loader:

```go
model.SetDevice(tensor.CUDA)

// CPU tensor — graph auto-moves to CUDA before execution.
cpuInput, _ := tensor.FromFloat32(data, shape)
output := model.Forward(autograd.NewVariable(cpuInput, false))
```

For training loops, prefer the loader's `Device` option — it moves
both input and target tensors, avoiding the target-side mismatch that
auto-move alone cannot handle.

### Full CUDA training pattern

```go
model, _ := buildModel()
model.SetDevice(tensor.CUDA)
optimizer := nn.NewAdam(model.Parameters(), 0.001)

loader := data.NewLoader(ds, data.LoaderConfig{
    BatchSize: 32,
    Shuffle:   true,
    Device:    tensor.DevicePtr(tensor.CUDA),
})
defer loader.Close()

model.SetTraining(true)
for loader.Next() {
    inT, tgtT := loader.Batch() // already on CUDA
    pred := model.Forward(autograd.NewVariable(inT, true))
    loss := nn.MSELoss(pred, autograd.NewVariable(tgtT, false))

    optimizer.ZeroGrad()
    loss.Backward()
    nn.ClipGradNorm(model.Parameters(), 1.0)
    optimizer.Step()
}
```

### Device query

`Device()` returns the configured device, or nil if `SetDevice` was
never called:

```go
if d := model.Device(); d != nil {
    fmt.Println("model on", *d) // "cuda" or "cpu"
}
```

## The Training Loop

The standard pattern is: **forward -> loss -> zeroGrad -> backward -> clip -> step**.

```go
model.SetTraining(true)

for loader.Next() {
    inT, tgtT := loader.Batch()
    input := autograd.NewVariable(inT, true)
    target := autograd.NewVariable(tgtT, false)

    // 1. Forward
    pred := model.Forward(input)

    // 2. Loss
    loss := nn.MSELoss(pred, target)

    // 3. Zero gradients
    optimizer.ZeroGrad()

    // 4. Backward
    loss.Backward()

    // 5. Clip gradients
    nn.ClipGradNorm(model.Parameters(), 1.0)

    // 6. Update parameters
    optimizer.Step()
}
```

## Observing Training

Tag the nodes you want to monitor when building the graph:

```go
model, _ := graph.From(nn.MustLinear(2, 16)).
    Through(nn.NewGELU()).
    Through(nn.MustLinear(16, 2)).Tag("output").
    Build()
```

### Log

Print current tagged values after any Forward call:

```go
model.Forward(input)
model.Log("output")  // output: 0.2341
model.Log()          // all tagged values
```

The default prints to stderr. Replace it with `OnLog` for custom handling:

```go
model.OnLog(func(values map[string]*autograd.Variable) {
    // structured logging, file output, etc.
})
```

### Collect, Record, and Flush

For epoch-level metrics, collect scalar values during the batch loop
and flush at epoch boundaries. `Collect` captures tagged graph node
outputs; `Record` injects external metrics (losses, hit rates — anything
computed outside the graph) into the same pipeline:

```go
for epoch := range epochs {
    loader.Reset()
    for loader.Next() {
        inT, tgtT := loader.Batch()
        input := autograd.NewVariable(inT, true)
        target := autograd.NewVariable(tgtT, false)

        pred := model.Forward(input)
        loss := nn.MSELoss(pred, target)

        optimizer.ZeroGrad()
        loss.Backward()
        optimizer.Step()

        model.Collect("output")              // from graph tag
        model.Record("loss", loss.Item())    // external metric
    }
    model.Flush()                 // batch mean → epoch history
}
```

`Collect` appends the scalar value of each tagged node to a batch
buffer. `Record` pushes raw `float64` values into the same buffer.
`Flush` computes the mean of the buffer, stores it in the epoch
history, and clears the buffer. The epoch history is then queryable
as a trend — see [Tutorial 8: Utilities](08-utilities.md#trend-based-training-control).

Since losses are typically computed outside the graph (they need both
graph outputs and external targets), `Record` is the natural way to
track them alongside graph-internal metrics.

## Stateful Graphs — DetachState

If your model uses forward references (recurrent state carried between
Forward calls), you **must** call `DetachState()` between training
steps. Without it, the autograd computation graph grows without bound —
each step's state buffer holds gradFn chains linking back through all
previous steps. Memory grows O(N) with the number of training steps.

```go
model.SetTraining(true)

for loader.Next() {
    inT, tgtT := loader.Batch()
    input := autograd.NewVariable(inT, true)
    target := autograd.NewVariable(tgtT, false)

    pred := model.Forward(input)
    loss := nn.MSELoss(pred, target)

    optimizer.ZeroGrad()
    loss.Backward()
    nn.ClipGradNorm(model.Parameters(), 1.0)

    model.DetachState() // break gradient chains on state buffers
    optimizer.Step()
}
```

`DetachState` is recursive — it uses `WalkModules` to traverse the
full module tree (including `SubModuler` children) and calls `Detach()`
on any module implementing `nn.Detachable`. A single call on the
outermost graph handles the entire model hierarchy.

**When is it needed?** Only for graphs with forward references — where
`Using("x")` appears before `Tag("x")` in the builder chain. Plain
feedforward graphs (no recurrent state) don't need it.

For full details on the mechanism and the `Detachable` interface, see
[Advanced Graphs — Managing state](06-advanced-graphs.md#managing-state).

### The PyTorch equivalent

In PyTorch, the same problem appears with RNN hidden states. The
standard pattern is to detach the hidden state between training steps:

```python
# PyTorch — manual hidden state detachment
hidden = model.init_hidden(batch_size)
for batch in dataloader:
    hidden = hidden.detach()  # or hidden.detach_() for in-place
    output, hidden = model(batch.input, hidden)
    loss = criterion(output, batch.target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

`hidden.detach()` creates a new tensor with the same data but no
gradient history — exactly what goDl's `DetachState` does to state
buffers. The difference is that PyTorch requires you to track and
detach each hidden state variable yourself, while goDl handles it
with a single `DetachState()` call on the graph.

PyTorch also offers `retain_graph=True` on `backward()` to keep the
computation graph alive for multiple backward passes, but this is
orthogonal — it's about reusing the graph, not about breaking cross-step
chains.

## Checkpoints — Training Resume

The `nn.Checkpoint` type bundles model parameters, optimizer state,
scheduler state, and epoch number into a single atomic file.

### Setup

```go
model, _ := buildModel()
optimizer := nn.NewAdam(model.Parameters(), 0.001)
scheduler := nn.Cosine(optimizer, 100)

ckpt := nn.NewCheckpoint("checkpoints/mymodel").
    Model(model).
    Add("optimizer", optimizer).
    Add("scheduler", scheduler)
```

The path prefix determines file location. Checkpoint files are named
`{prefix}_{epoch:06d}.ckpt` — for example, `checkpoints/mymodel_000010.ckpt`.

### Save during training

```go
for epoch := range numEpochs {
    // ... training loop ...

    if epoch%10 == 0 {
        if err := ckpt.Save(epoch); err != nil {
            log.Fatal(err)
        }
    }
}
```

### Resume from latest

```go
ckpt := nn.NewCheckpoint("checkpoints/mymodel").
    Model(model).
    Add("optimizer", optimizer).
    Add("scheduler", scheduler)

startEpoch, err := ckpt.Load()
if err != nil {
    // No checkpoint found — start from scratch.
    startEpoch = 0
}

for epoch := startEpoch; epoch < numEpochs; epoch++ {
    // ... training loop ...
}
```

`Load()` finds the most recent checkpoint file (by filename sort),
restores all state, and returns the saved epoch number. All named
components must match between save and load — mismatched names or
counts produce an error.

Checkpoint loading is device-aware: parameters and optimizer state
(momentum buffers, moment estimates) are automatically moved to the
device they were on before the load. This means you can call
`SetDevice` before loading, and the restored tensors end up on the
correct device:

```go
model.SetDevice(tensor.CUDA)
optimizer := nn.NewAdam(model.Parameters(), 0.001)

ckpt := nn.NewCheckpoint("checkpoints/mymodel").
    Model(model).
    Add("optimizer", optimizer)

startEpoch, err := ckpt.Load()  // params + optimizer state land on CUDA
```

### What gets saved

| Component | What's persisted |
|-----------|-----------------|
| Model (`Model(m)`) | All parameter tensors (names, shapes, values) |
| SGD | Learning rate, velocity tensors (if momentum > 0) |
| Adam / AdamW | Learning rate, step counter, m and v moment estimates |
| StepDecay / Cosine | Tick counter |
| Warmup | Tick counter + inner scheduler state |
| Plateau | Best value, wait counter, started flag |
| GradScaler | Scale factor, steps-since-growth counter |

Any type implementing the `nn.Stateful` interface can be added:

```go
type Stateful interface {
    SaveState(w io.Writer) error
    LoadState(r io.Reader) error
}
```

## Eval Mode

Switch to eval mode for inference. This affects Dropout (becomes identity) and BatchNorm (uses running statistics):

```go
model.SetTraining(false)
autograd.NoGrad(func() {
    output := model.Forward(input)
    // No graph built, no gradient tracking overhead.
})
```

## Context-Aware Forward

`ForwardCtx` threads Go's `context.Context` through the graph. Loops
and maps check for cancellation between iterations. This enables
wall-clock timeouts, cancellation from another goroutine, and deadline
enforcement — things Python cannot express inside a forward pass.

### Timeouts for dynamic loops

While and Until loops accept a maximum iteration count, but in serving
you often need a time bound, not just an iteration cap:

```go
ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
defer cancel()

result := model.ForwardCtx(ctx, input)
if err := result.Err(); err != nil {
    // context.DeadlineExceeded — loop ran out of time.
    // Fall back to a simpler model or return partial result.
}
```

The loop returns the state computed so far. If the context expires
before the first iteration, `ForwardCtx` returns the context error
immediately.

### Training with a deadline

In test training or hyperparameter search, you may want to kill a
run that takes too long per batch:

```go
for loader.Next() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

    pred := model.ForwardCtx(ctx, autograd.NewVariable(input, true))
    cancel()

    if err := pred.Err(); err != nil {
        log.Printf("forward timed out, skipping batch: %v", err)
        continue
    }

    loss := nn.MSELoss(pred, target)
    optimizer.ZeroGrad()
    loss.Backward()
    optimizer.Step()
}
```

### Cancellation from another goroutine

You can cancel a long-running forward pass externally:

```go
ctx, cancel := context.WithCancel(context.Background())

go func() {
    // Cancel after receiving a signal, a user request, etc.
    <-stopCh
    cancel()
}()

result := model.ForwardCtx(ctx, input)
```

### What gets checked

Context is checked at three points during execution:

1. **Between topological levels** — if the graph has multiple sequential
   stages, cancellation is detected between them.
2. **Between loop iterations** — For, While, and Until all check before
   each iteration. Until preserves its at-least-once guarantee by
   checking only after the first iteration.
3. **Between map elements** — `Map.Each()` and `Map.Slices(n)` check
   before each element. `Batched()` maps run in a single call and are
   not interruptible.

When no context is needed, `Forward` is a zero-overhead wrapper
around `ForwardCtx(context.Background(), ...)` — the background
context never cancels, so `ctx.Err()` returns nil immediately (~2ns).

### Combined with trends

Context-aware forward composes naturally with the observation layer.
For example, abandon training early if loss isn't improving within
a time budget:

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
defer cancel()

for epoch := range maxEpochs {
    loader.Reset()
    for loader.Next() {
        inT, tgtT := loader.Batch()
        input := autograd.NewVariable(inT, true)
        target := autograd.NewVariable(tgtT, false)

        pred := model.ForwardCtx(ctx, input)
        if pred.Err() != nil {
            break // time budget exhausted
        }

        loss := nn.MSELoss(pred, target)
        optimizer.ZeroGrad()
        loss.Backward()
        optimizer.Step()

        model.Collect("output")
    }
    model.Flush()

    if model.Trend("output").Converged(5, 1e-5) {
        break // converged — stop early
    }
}
```

## Tag Groups and Trend Groups

When you have parallel branches from `Split`, `TagGroup` names them
all at once with auto-suffixed tags. `Trends` expands groups for
aggregate queries — useful for multi-head architectures:

```go
// Build a multi-head model.
g, _ := graph.From(encoder).
    Split(headA, headB, headC).TagGroup("head").
    Merge(graph.Mean()).
    Through(outputHead).Tag("loss").
    Build()

// Training loop with group observation.
for epoch := range maxEpochs {
    for loader.Next() {
        input, target := loader.Batch()
        pred := g.Forward(autograd.NewVariable(input, true))
        loss := nn.MSELoss(pred, autograd.NewVariable(target, false))
        optimizer.ZeroGrad()
        loss.Backward()
        optimizer.Step()

        g.Collect("loss", "head_0", "head_1", "head_2")
    }
    g.Flush()

    // Group queries expand "head" → head_0, head_1, head_2.
    if g.Trends("head").AllImproving(5) {
        fmt.Println("all heads improving")
    }
    if g.Trends("head").AnyStalled(5, 1e-4) {
        scheduler.Decay()
    }
    // Per-trend slopes for custom logic.
    slopes := g.Trends("head").Slopes(5)
    fmt.Printf("head slopes: %v\n", slopes)
}
```

`TrendGroup` methods mirror single-`Trend` queries:
- `AllImproving(w)` / `AnyImproving(w)` — slope direction
- `AllStalled(w, tol)` / `AnyStalled(w, tol)` — slope near zero
- `AllConverged(w, tol)` / `AnyConverged(w, tol)` — variance below tolerance
- `MeanSlope(w)` — average slope across the group
- `Slopes(w)` — individual slopes as `[]float64`

Timing trends work the same way via `g.TimingTrends("head")`:

```go
g.EnableProfiling()
// ... training ...
if g.TimingTrends("head").MeanSlope(5) > 0.001 {
    fmt.Println("heads getting slower — possible memory issue")
}
```

## Complete Example: Learning Cumulative Sum

This example trains a small graph to learn cumulative sum: given `[a, b]`, predict `[a, a+b]`. It is adapted directly from `examples/train/train_test.go`.

```go
package main

import (
    "fmt"
    "math"
    "math/rand/v2"

    "github.com/fab2s/goDl/autograd"
    "github.com/fab2s/goDl/data"
    "github.com/fab2s/goDl/graph"
    "github.com/fab2s/goDl/nn"
    "github.com/fab2s/goDl/tensor"
)

// buildModel creates:
//   Linear(2,16) -> GELU -> LayerNorm -> Also(Linear) -> Linear(16,2)
//
// Also() adds a residual connection (skip connection).
func buildModel() (*graph.Graph, error) {
    return graph.From(nn.MustLinear(2, 16)).
        Through(nn.NewGELU()).
        Through(nn.MustLayerNorm(16)).
        Also(nn.MustLinear(16, 16)).
        Through(nn.MustLinear(16, 2)).
        Build()
}

// makeDataset generates n samples: input [a,b] -> target [a, a+b].
func makeDataset(n int) *data.TensorDataset {
    inputs := make([]float32, n*2)
    targets := make([]float32, n*2)
    for i := range n {
        a := rand.Float32()*2 - 1
        b := rand.Float32()*2 - 1
        inputs[i*2] = a
        inputs[i*2+1] = b
        targets[i*2] = a
        targets[i*2+1] = a + b
    }
    inT, _ := tensor.FromFloat32(inputs, []int64{int64(n), 2})
    tgtT, _ := tensor.FromFloat32(targets, []int64{int64(n), 2})
    return data.NewTensorDataset(inT, tgtT)
}

func main() {
    // Build model.
    model, err := buildModel()
    if err != nil {
        panic(err)
    }

    // Create dataset and loader.
    ds := makeDataset(200)
    loader := data.NewLoader(ds, data.LoaderConfig{
        BatchSize: 20,
        Shuffle:   true,
    })
    defer loader.Close()

    // Create optimizer.
    optimizer := nn.NewAdam(model.Parameters(), 0.01)

    // Training loop.
    model.SetTraining(true)

    var firstLoss, lastLoss float64
    epochs := 50
    for epoch := range epochs {
        loader.Reset()
        epochLoss := 0.0
        batches := 0

        for loader.Next() {
            inT, tgtT := loader.Batch()
            input := autograd.NewVariable(inT, true)
            target := autograd.NewVariable(tgtT, false)

            // Forward.
            pred := model.Forward(input)
            if err := pred.Err(); err != nil {
                panic(err)
            }

            // Loss.
            loss := nn.MSELoss(pred, target)
            if err := loss.Err(); err != nil {
                panic(err)
            }

            // Backward.
            optimizer.ZeroGrad()
            if err := loss.Backward(); err != nil {
                panic(err)
            }

            // Clip gradients to prevent explosion.
            nn.ClipGradNorm(model.Parameters(), 1.0)

            // Update.
            optimizer.Step()

            lossVal, _ := loss.Data().Float32Data()
            epochLoss += float64(lossVal[0])
            batches++
        }
        if err := loader.Err(); err != nil {
            panic(err)
        }

        avgLoss := epochLoss / float64(batches)
        if epoch == 0 {
            firstLoss = avgLoss
        }
        lastLoss = avgLoss

        if epoch%10 == 0 || epoch == epochs-1 {
            fmt.Printf("epoch %3d  loss=%.6f\n", epoch, avgLoss)
        }
    }

    // Verify convergence.
    if lastLoss >= firstLoss*0.5 {
        fmt.Printf("WARNING: training did not converge (first=%.6f last=%.6f)\n",
            firstLoss, lastLoss)
    }

    // Eval.
    model.SetTraining(false)
    testInput, _ := tensor.FromFloat32([]float32{0.5, 0.3}, []int64{1, 2})
    var pred *autograd.Variable
    autograd.NoGrad(func() {
        pred = model.Forward(autograd.NewVariable(testInput, false))
    })
    vals, _ := pred.Data().Float32Data()

    fmt.Printf("input=[0.5, 0.3] -> pred=%v (want ~ [0.5, 0.8])\n", vals)

    if math.Abs(float64(vals[0])-0.5) > 0.15 || math.Abs(float64(vals[1])-0.8) > 0.15 {
        fmt.Println("Predictions are outside expected range.")
    } else {
        fmt.Println("Model learned cumulative sum successfully.")
    }
}
```

### Key points illustrated

1. **Model construction** -- the graph builder chains `From -> Through -> Also -> Through -> Build`.
2. **Data pipeline** -- `TensorDataset` + `Loader` with scanner-style iteration.
3. **Training loop** -- forward, loss, zeroGrad, backward, clipGrad, step.
4. **Observation** -- `Collect` snapshots metrics per batch, `Flush` promotes to epoch history.
5. **Eval mode** -- `SetTraining(false)` + `NoGrad` for inference.
6. **Gradient clipping** -- `ClipGradNorm` between backward and step.

---

Next: [Tutorial 5: The Graph Builder](05-graph-builder.md)

Previous: [01-Tensors](01-tensors.md) | [02-Autograd](02-autograd.md) | [03-Modules](03-modules.md)
