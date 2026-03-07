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

## Eval Mode

Switch to eval mode for inference. This affects Dropout (becomes identity) and BatchNorm (uses running statistics):

```go
model.SetTraining(false)
autograd.NoGrad(func() {
    output := model.Forward(input)
    // No graph built, no gradient tracking overhead.
})
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
4. **Eval mode** -- `SetTraining(false)` + `NoGrad` for inference.
5. **Gradient clipping** -- `ClipGradNorm` between backward and step.

---

Next: [Tutorial 5: The Graph Builder](05-graph-builder.md)

Previous: [01-Tensors](01-tensors.md) | [02-Autograd](02-autograd.md) | [03-Modules](03-modules.md)
