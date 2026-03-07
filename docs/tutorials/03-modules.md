# Tutorial 3: Modules

The `nn` package provides neural network layers, activations, and the `Module` interface that unifies them all. Modules compose naturally -- a model is a Module that contains other Modules.

This tutorial builds on [Tutorial 2: Automatic Differentiation](02-autograd.md).

## The Module Interface

Every layer in goDl implements this interface:

```go
type Module interface {
    Forward(inputs ...*autograd.Variable) *autograd.Variable
    Parameters() []*Parameter
}
```

`Forward` takes one or more input variables and returns an output variable. `Parameters` returns all learnable weights. Modules with no learnable parameters (like activations) return nil.

## Linear

Fully connected layer: `y = x @ W^T + b`.

```go
// Returns (*Linear, error)
linear, err := nn.NewLinear(784, 128)

// Panic-on-error variant for graph construction with known dimensions
linear := nn.MustLinear(784, 128)
```

Weights are Kaiming-initialized (suitable for ReLU). Input shape: `[batch, inFeatures]`. Output shape: `[batch, outFeatures]`.

```go
output := linear.Forward(input)  // [batch, 784] -> [batch, 128]
```

## Conv2d

2D convolution over `[N, C, H, W]` inputs.

```go
conv, err := nn.NewConv2d(3, 64, 3)  // inChannels=3, outChannels=64, kernel=3x3
```

Configure stride, padding, dilation, and groups via exported fields:

```go
conv.Padding = [2]int64{1, 1}   // same-padding for 3x3 kernel
conv.Stride = [2]int64{2, 2}    // stride 2 downsampling
conv.Dilation = [2]int64{1, 1}  // default
conv.Groups = 1                  // default (set to inChannels for depthwise)
```

Weights are Kaiming-initialized.

## Normalization

### LayerNorm

Normalizes the last dimension. Commonly used in transformers.

```go
ln, err := nn.NewLayerNorm(512)
ln := nn.MustLayerNorm(512)  // panic-on-error variant

output := ln.Forward(input)  // [batch, 512] -> [batch, 512]
```

### BatchNorm

Normalizes over the batch dimension. Uses running statistics at inference.

```go
bn, err := nn.NewBatchNorm(128)
output := bn.Forward(input)  // [batch, 128] -> [batch, 128]
```

BatchNorm behaves differently during training (batch statistics) vs. inference (running statistics). See [Train/Eval Mode](#traineval-mode) below.

## Dropout

Randomly zeroes elements during training. Uses inverted dropout so no scaling is needed at inference.

```go
drop := nn.NewDropout(0.1)  // 10% drop probability
output := drop.Forward(input)
```

During inference, Dropout becomes an identity function. See [Train/Eval Mode](#traineval-mode).

## Embedding

Lookup table mapping integer indices to dense vectors.

```go
emb, err := nn.NewEmbedding(10000, 64)  // vocab=10000, dim=64
```

Input is a Variable wrapping an Int64 tensor. Supports arbitrary input shapes:

```go
// [batch, seqLen] -> [batch, seqLen, 64]
output := emb.Forward(indices)
```

## Recurrent Cells

### GRUCell

Single GRU timestep. Hidden state is nil on the first call (auto-initializes to zeros):

```go
gru, err := nn.NewGRUCell(128, 256)

h := gru.Forward(x)        // first step: h initialized to zeros
h = gru.Forward(x2, h)     // subsequent steps: pass previous h
```

Input `x`: `[batch, inputSize]`. Hidden `h`: `[batch, hiddenSize]`.

### LSTMCell

Single LSTM timestep. State packs hidden and cell states into one tensor via concatenation along the last dimension:

```go
lstm, err := nn.NewLSTMCell(128, 256)

state := lstm.Forward(x)            // first step: h,c initialized to zeros
state = lstm.Forward(x2, state)     // state is [batch, 2*hiddenSize]
```

The packed state format (`cat(h, c)` along dim=1) allows LSTMCell to work with the graph builder's forward reference mechanism, which passes a single Variable between calls.

## Activations

Activation functions are also modules, making them composable in the graph builder:

```go
nn.NewReLU()        // max(0, x)
nn.NewSigmoid()     // 1 / (1 + exp(-x))
nn.NewTanh()        // hyperbolic tangent
nn.NewGELU()        // Gaussian Error Linear Unit (tanh approximation)
nn.NewSiLU()        // x * sigmoid(x), also called Swish
nn.NewSoftmax(dim)  // softmax along given dimension
```

All activations have no learnable parameters.

## Train/Eval Mode

Some modules (Dropout, BatchNorm) behave differently during training vs. inference. They implement the `TrainToggler` interface:

```go
type TrainToggler interface {
    SetTraining(training bool)
}
```

Use the helper function to set training mode on any module (no-op if the module does not implement TrainToggler):

```go
nn.SetTraining(dropout, false)   // eval mode
nn.SetTraining(dropout, true)    // back to training
```

When using the graph builder, `Graph.SetTraining(bool)` propagates to all nodes recursively.

## Composing Modules Manually

Without the graph builder, you compose modules in plain Go:

```go
type MLP struct {
    fc1 *nn.Linear
    fc2 *nn.Linear
    relu *nn.ReLU
}

func NewMLP() (*MLP, error) {
    fc1, err := nn.NewLinear(784, 128)
    if err != nil { return nil, err }
    fc2, err := nn.NewLinear(128, 10)
    if err != nil { return nil, err }
    return &MLP{fc1: fc1, fc2: fc2, relu: nn.NewReLU()}, nil
}

func (m *MLP) Forward(inputs ...*autograd.Variable) *autograd.Variable {
    x := m.fc1.Forward(inputs[0])
    x = m.relu.Forward(x)
    return m.fc2.Forward(x)
}

func (m *MLP) Parameters() []*nn.Parameter {
    var params []*nn.Parameter
    params = append(params, m.fc1.Parameters()...)
    params = append(params, m.fc2.Parameters()...)
    return params
}
```

This works fine for simple models. For anything involving residual connections, parallel branches, loops, or conditional execution, the graph builder API (`graph` package) is more expressive and handles the wiring automatically.

---

Next: [Tutorial 4: Training](04-training.md)
