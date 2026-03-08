# PyTorch → goDl Migration Guide

A side-by-side reference for PyTorch users learning goDl.

## Imports

In PyTorch, `import torch` gives you almost everything. Go uses explicit imports —
each package has a clear responsibility. Here's what you'll typically need:

```go
import (
    "github.com/fab2s/goDl/tensor"   // tensor creation and ops (like torch)
    "github.com/fab2s/goDl/autograd"  // gradient tracking (like torch.autograd)
    "github.com/fab2s/goDl/nn"        // modules, losses, optimizers (like torch.nn + torch.optim)
    "github.com/fab2s/goDl/graph"     // graph builder (goDl-specific, no PyTorch equivalent)
    "github.com/fab2s/goDl/data"      // datasets and loaders (like torch.utils.data)
)
```

**You won't need all five in every file.** Here's what goes where:

| If you're doing... | You need |
|---------------------|----------|
| Defining a model with the graph builder | `nn`, `graph` |
| Writing a training loop | `nn`, `autograd`, `tensor` |
| Creating raw tensors or data pipelines | `tensor`, `data` |
| Everything (main training script) | all five |

**How it maps to PyTorch**:

| PyTorch | goDl | What's in it |
|---------|------|-------------|
| `torch.*` | `tensor` | Creation (`Zeros`, `Rand`, `Arange`...), math ops, shape ops |
| `torch.autograd` | `autograd` | `Variable`, `NewVariable`, `NoGrad`, gradient tracking |
| `torch.nn` | `nn` | Modules (`Linear`, `Conv2d`...), activations, losses |
| `torch.optim` | `nn` | Optimizers (`Adam`, `SGD`, `AdamW`), LR schedulers |
| `torch.utils.data` | `data` | `Dataset`, `Loader` |
| *(no equivalent)* | `graph` | Fluent computation graph builder |

> **Tip**: Go editors with `goimports` auto-add imports when you use a symbol.
> Just type `tensor.Zeros` and your editor inserts the import.

## Core Concepts

| PyTorch | goDl | Notes |
|---------|------|-------|
| `torch.Tensor` | `*tensor.Tensor` | Immutable, chainable with error propagation |
| `torch.autograd` | `autograd.Variable` | Wraps Tensor, tracks gradients |
| `torch.nn.Module` | `nn.Module` interface | `Forward(...*Variable) *Variable` + `Parameters()` |
| `model.train()` | `nn.SetTraining(m, true)` | Or `m.SetTraining(true)` if module implements TrainToggler |
| `with torch.no_grad():` | `autograd.NoGrad(func() { ... })` | Disables gradient tracking in block |

## Tensor Creation

```python
# PyTorch
x = torch.zeros(2, 3)
x = torch.ones(2, 3)
x = torch.rand(2, 3)
x = torch.randn(2, 3)
x = torch.full((2, 3), 7.0)
x = torch.eye(4)
x = torch.arange(0, 10, 2)
x = torch.arange(5)
x = torch.tensor([1.0, 2.0, 3.0])
x = torch.tensor([0, 1, 2], dtype=torch.int64)
x = torch.linspace(0, 1, 10)
```

```go
// goDl
x, _ := tensor.Zeros([]int64{2, 3})
x, _ := tensor.Ones([]int64{2, 3})
x, _ := tensor.Rand([]int64{2, 3})
x, _ := tensor.RandN([]int64{2, 3})
x, _ := tensor.Full([]int64{2, 3}, 7.0)
x, _ := tensor.Eye(4)
x, _ := tensor.Arange(0, 10, 2)
x, _ := tensor.ArangeEnd(5)
x, _ := tensor.FromFloat32([]float32{1.0, 2.0, 3.0}, []int64{3})
x, _ := tensor.FromInt64([]int64{0, 1, 2}, []int64{3})
x, _ := tensor.Linspace(0, 1, 10)
```

## Tensor Operations

### Arithmetic

```python
# PyTorch
c = a + b           # element-wise add
c = a - b           # element-wise sub
c = a * b           # element-wise mul
c = a / b           # element-wise div
c = a @ b           # matrix multiply
c = x * 2.0         # scalar multiply
c = x + 1.0         # scalar add
c = x / 3.0         # scalar divide
c = -x              # negation
```

```go
// goDl
c := a.Add(b)
c := a.Sub(b)
c := a.Mul(b)
c := a.Div(b)
c := a.Matmul(b)
c := x.MulScalar(2.0)
c := x.AddScalar(1.0)
c := x.DivScalar(3.0)
c := x.Neg()
```

### Math Functions

```python
# PyTorch
y = torch.exp(x)
y = torch.log(x)
y = torch.sqrt(x)
y = torch.abs(x)
y = torch.pow(x, 2.0)
y = torch.clamp(x, -1.0, 1.0)
```

```go
// goDl
y := x.Exp()
y := x.Log()
y := x.Sqrt()
y := x.Abs()
y := x.Pow(2.0)
y := x.Clamp(-1.0, 1.0)
```

### Activations

```python
# PyTorch
y = torch.relu(x)
y = torch.sigmoid(x)
y = torch.tanh(x)
y = torch.softmax(x, dim=1)
```

```go
// goDl
y := x.ReLU()
y := x.Sigmoid()
y := x.Tanh()
y := x.Softmax(1)
```

### Reductions

```python
# PyTorch
s = x.sum()
s = x.sum(dim=1, keepdim=True)
m = x.mean()
m = x.mean(dim=1, keepdim=True)
v = x.max(dim=1, keepdim=True).values
v = x.min(dim=1, keepdim=True).values
idx = x.argmax(dim=1)
```

```go
// goDl
s := x.Sum()
s := x.SumDim(1, true)
m := x.Mean()
m := x.MeanDim(1, true)
v := x.MaxDim(1, true)
v := x.MinDim(1, true)
idx := x.ArgMax(1, false)  // returns Int64 tensor
```

### Shape Operations

```python
# PyTorch
y = x.reshape(2, 3)
y = x.squeeze(0)
y = x.unsqueeze(0)
y = x.flatten(1)
y = x.permute(0, 2, 1)
y = x.transpose(0, 1)
y = x.expand(4, 3)
```

```go
// goDl
y := x.Reshape([]int64{2, 3})
y := x.Squeeze(0)
y := x.Unsqueeze(0)
y := x.Flatten(1)
y := x.Permute(0, 2, 1)
y := x.Transpose(0, 1)
y := x.Expand([]int64{4, 3})
```

### Indexing and Slicing

```python
# PyTorch
y = x[0]                       # select first along dim 0
y = x[:, 1:3]                  # narrow: dim=1, start=1, length=2
y = x.index_select(0, indices) # gather rows
y = torch.cat([a, b, c], dim=0)
y = torch.stack([a, b, c], dim=0)
```

```go
// goDl
y := x.Select(0, 0)
y := x.Narrow(1, 1, 2)
y := x.IndexSelect(0, indices)
y := tensor.CatAll([]*tensor.Tensor{a, b, c}, 0)
y := tensor.Stack([]*tensor.Tensor{a, b, c}, 0)
```

### Comparisons and Conditionals

```python
# PyTorch
mask = x > 0
mask = x >= 0
mask = x < 0
mask = x <= 0
y = torch.where(mask, a, b)
```

```go
// goDl — returns float masks (1.0/0.0) for chainable math
mask := x.GTScalar(0)
mask := x.GEScalar(0)
mask := x.LTScalar(0)
mask := x.LEScalar(0)
y := mask.Where(a, b)
```

### Data Access

```python
# PyTorch
val = loss.item()          # scalar → float
data = x.numpy()           # → numpy array
data = x.tolist()          # → Python list
```

```go
// goDl
val := loss.Item()                // scalar → float64
data, _ := x.Float32Data()       // → []float32
data, _ := x.Float64Data()       // → []float64
data, _ := x.Int64Data()         // → []int64
```

### One-Hot Encoding

```python
# PyTorch
one_hot = F.one_hot(labels, num_classes=10)
```

```go
// goDl
oneHot := tensor.OneHot(labels, 10, 0)  // 0 = infer batch size
```

## Autograd

```python
# PyTorch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # [2.0, 4.0]
```

```go
// goDl
xt, _ := tensor.FromFloat32([]float32{1.0, 2.0}, []int64{2})
x := autograd.NewVariable(xt, true)
y := x.Pow(2).Sum()
y.Backward()
fmt.Println(x.Grad())  // [2.0, 4.0]
```

All tensor ops that have autograd support: Add, Sub, Mul, Div, Matmul, MulScalar, AddScalar,
DivScalar, Neg, Exp, Log, Sqrt, Abs, Pow, Clamp, ReLU, Sigmoid, Tanh, Softmax, Sum, SumDim,
MeanDim, Mean, MaxDim, Reshape, Transpose, Permute, Squeeze, Unsqueeze, Flatten, Select,
Narrow, Cat, IndexSelect, Expand, Conv2d, ConvTranspose2d, AdaptiveAvgPool2d, GridSample.

## Neural Network Layers

```python
# PyTorch
layer = nn.Linear(784, 128)
layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)
layer = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
layer = nn.LayerNorm(128)
layer = nn.BatchNorm1d(128)
layer = nn.Dropout(p=0.5)
layer = nn.Embedding(1000, 128)
cell = nn.GRUCell(128, 256)
cell = nn.LSTMCell(128, 256)
```

```go
// goDl
layer, _ := nn.NewLinear(784, 128)
layer, _ := nn.NewConv2d(3, 64, 3, nn.Conv2dOpts{Padding: [2]int64{1, 1}})
layer, _ := nn.NewConvTranspose2d(64, 3, 4, nn.ConvTranspose2dOpts{Stride: [2]int64{2, 2}, Padding: [2]int64{1, 1}})
layer, _ := nn.NewLayerNorm(128)
layer, _ := nn.NewBatchNorm(128)
layer := nn.NewDropout(0.5)
layer, _ := nn.NewEmbedding(1000, 128)
cell, _ := nn.NewGRUCell(128, 256)
cell, _ := nn.NewLSTMCell(128, 256)

// "Must" constructors panic on error (convenient for init):
layer := nn.MustLinear(784, 128)
layer := nn.MustConv2d(3, 64, 3, nn.Conv2dOpts{})
```

## Activations (as Modules)

```python
# PyTorch
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.GELU()
nn.SiLU()
nn.Softmax(dim=1)
```

```go
// goDl
nn.NewReLU()
nn.NewSigmoid()
nn.NewTanh()
nn.NewGELU()
nn.NewSiLU()
nn.NewSoftmax(1)
```

## Loss Functions

```python
# PyTorch
loss = F.mse_loss(pred, target)
loss = F.cross_entropy(logits, labels)       # labels: int indices
loss = F.cross_entropy(logits, one_hot)      # labels: one-hot
loss = F.binary_cross_entropy_with_logits(pred, target)
loss = F.l1_loss(pred, target)
loss = F.smooth_l1_loss(pred, target, beta=1.0)
loss = F.kl_div(log_probs, targets, reduction='batchmean')
```

```go
// goDl
loss := nn.MSELoss(pred, target)
loss := nn.CrossEntropyLoss(logits, labels)       // auto-detects int indices
loss := nn.CrossEntropyLoss(logits, oneHot)        // or one-hot — both work
loss := nn.BCEWithLogitsLoss(pred, target)
loss := nn.L1Loss(pred, target)
loss := nn.SmoothL1Loss(pred, target, 1.0)
loss := nn.KLDivLoss(logProbs, targets)
```

## Optimizers

```python
# PyTorch
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

opt.zero_grad()
loss.backward()
opt.step()
```

```go
// goDl
opt := nn.NewSGD(model.Parameters(), 0.01, 0.9)
opt := nn.NewAdam(model.Parameters(), 0.001)
opt := nn.NewAdamW(model.Parameters(), 0.001, 0.01)

opt.ZeroGrad()
loss.Backward()
opt.Step()
```

## Learning Rate Scheduling

```python
# PyTorch
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
scheduler.step()
```

```go
// goDl
scheduler := nn.StepDecay(opt, 30, 0.1)
scheduler := nn.Cosine(opt, 100)
scheduler.Step()

// Composable warmup:
scheduler = nn.Warmup(nn.Cosine(opt, 100), 10)

// Plateau (reduce on loss plateau):
scheduler := nn.Plateau(opt, 10, 0.1)
scheduler.Observe(lossValue)
```

## Gradient Clipping

```python
# PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

```go
// goDl
nn.ClipGradNorm(model.Parameters(), 1.0)
nn.ClipGradValue(model.Parameters(), 0.5)
```

## Saving and Loading

```python
# PyTorch — model only
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

```go
// goDl — model only
nn.SaveParametersFile("model.bin", model.Parameters())
nn.LoadParametersFile("model.bin", model.Parameters())

// Or with io.Writer/Reader for custom storage:
nn.SaveParameters(writer, model.Parameters())
nn.LoadParameters(reader, model.Parameters())
```

### Full training resume (model + optimizer + scheduler)

```python
# PyTorch — manual state dict management
torch.save({
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
}, "checkpoint.pt")

ckpt = torch.load("checkpoint.pt")
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
scheduler.load_state_dict(ckpt['scheduler'])
start_epoch = ckpt['epoch']
```

```go
// goDl — composable Checkpoint type
ckpt := nn.NewCheckpoint("checkpoints/model").
    Model(model).
    Add("optimizer", optimizer).
    Add("scheduler", scheduler)

// Save
ckpt.Save(epoch)

// Load (finds latest automatically)
startEpoch, err := ckpt.Load()
```

## Detaching Hidden State

In recurrent models, hidden state carries gradient chains across
training steps. Without detachment, the computation graph grows
without bound. Both frameworks require explicit detachment, but
the mechanism differs.

```python
# PyTorch — manual per-variable detachment
hidden = model.init_hidden(batch_size)
for batch in dataloader:
    hidden = hidden.detach()  # new tensor, no grad history
    output, hidden = model(batch.input, hidden)
    loss = criterion(output, batch.target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# For multiple hidden states (e.g., LSTM h and c):
h, c = h.detach(), c.detach()
```

```go
// goDl — single call detaches everything
for loader.Next() {
    output := model.Forward(input)
    loss := nn.MSELoss(output, target)
    loss.Backward()
    model.DetachState() // detaches all state buffers + Detachable modules
    optimizer.Step()
    optimizer.ZeroGrad()
}
```

Key differences:

| Aspect | PyTorch | goDl |
|--------|---------|------|
| Granularity | Per-variable: `h = h.detach()` | Per-graph: `model.DetachState()` |
| Discovery | You must know which variables to detach | Automatic — walks all state buffers and Detachable modules |
| Sub-models | Manual for each sub-module's state | Recursive — one call handles the full hierarchy |
| Module support | N/A (no equivalent interface) | `nn.Detachable` — modules declare their own detach logic |

In PyTorch, forgetting to detach a single hidden state variable causes
the same unbounded memory growth. In goDl, `DetachState` on the
outermost graph covers everything.

### When is it needed?

Only for models with cross-step state — forward references in goDl,
hidden states in PyTorch. Plain feedforward models don't need it.

## Weight Initialization

```python
# PyTorch
nn.init.kaiming_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)
```

```go
// goDl
w, _ := nn.KaimingUniform(shape, fanIn)
w, _ := nn.XavierNormal(shape, fanIn, fanOut)
```

## Training Loop Pattern

```python
# PyTorch
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

```go
// goDl
nn.SetTraining(model, true)
for epoch := range numEpochs {
    for loader.Next() {
        inputs, targets := loader.Batch()
        optimizer.ZeroGrad()
        outputs := model.Forward(inputs)
        loss := nn.CrossEntropyLoss(outputs, targets)
        loss.Backward()
        nn.ClipGradNorm(model.Parameters(), 1.0)
        optimizer.Step()
    }
    scheduler.Step()
    fmt.Printf("Epoch %d: loss=%.4f\n", epoch, loss.Item())
}
```

## Error Handling

goDl uses error propagation instead of exceptions. Errors chain through operations:

```go
// Check at the end of a chain — no need to check every step:
result := x.Matmul(w).Add(b).ReLU()
if err := result.Err(); err != nil {
    // handle error
}

// Variables work the same way:
loss := model.Forward(input)
if err := loss.Err(); err != nil {
    return fmt.Errorf("forward: %w", err)
}
```

## Graph Builder (Advanced)

goDl's unique feature: a fluent API for building computation graphs declaratively.

```go
// Simple sequential model:
g := graph.New().
    From(nn.MustLinear(784, 128)).
    Through(nn.NewReLU()).
    Through(nn.MustLinear(128, 10)).
    Build()

// With residual connections:
g := graph.New().
    From(nn.MustLinear(128, 128)).
    Also(nn.NewReLU(), nn.MustLinear(128, 128)).  // skip connection
    Build()

// Parallel branches:
g := graph.New().
    From(nn.MustLinear(128, 128)).
    Split(branch1, branch2).
    Merge(graph.Cat()).
    Build()

// Recurrent (loop):
g := graph.New().
    From(initModule).
    Loop(stepModule).For(10).Using("image").
    Through(headModule).
    Build()
```

The graph implements `nn.Module`, so it works with optimizers, checkpointing, and everything else.
