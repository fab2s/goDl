# Tutorial 2: Automatic Differentiation

The `autograd` package provides reverse-mode automatic differentiation. Variables wrap tensors with gradient tracking. When you perform operations on variables, a computation graph is built behind the scenes. Calling `Backward()` walks that graph in reverse, accumulating gradients at each leaf variable.

This tutorial builds on [Tutorial 1: Tensors](01-tensors.md).

## Variables

A `Variable` wraps a tensor and optionally tracks gradients:

```go
import (
    "github.com/fab2s/goDl/autograd"
    "github.com/fab2s/goDl/tensor"
)

t, _ := tensor.FromFloat32([]float32{2.0}, []int64{1})

// requiresGrad=true: operations on this variable build a computation graph
x := autograd.NewVariable(t, true)

// requiresGrad=false: just a constant, no tracking
c := autograd.NewVariable(someTensor, false)
```

Variables created by the user are **leaf variables**. Variables produced by operations are non-leaf (intermediate) nodes in the computation graph.

## Forward Pass: Building the Graph

Operations on variables mirror the tensor API. Each operation records its inputs and backward function:

```go
wT, _ := tensor.FromFloat32([]float32{3.0}, []int64{1})
xT, _ := tensor.FromFloat32([]float32{2.0}, []int64{1})

w := autograd.NewVariable(wT, true)
x := autograd.NewVariable(xT, true)

// y = x * w, then reduce to scalar
y := x.Mul(w).Sum()
// The graph now records: Sum <- Mul <- (x, w)
```

The full set of differentiable operations includes: `Add`, `Sub`, `Mul`, `Div`, `Matmul`, `ReLU`, `Sigmoid`, `Tanh`, `Exp`, `Log`, `Neg`, `Sqrt`, `Sum`, `SumDim`, `MeanDim`, `MulScalar`, `AddScalar`, `Softmax`, `Transpose`, `Reshape`, `Narrow`, `Cat`, `Select`, `IndexSelect`, `Conv2d`.

## Backward Pass: Computing Gradients

Call `Backward()` on a scalar variable to compute gradients for all leaf variables that contributed to it:

```go
err := y.Backward()
if err != nil {
    log.Fatal(err)
}
```

`Backward()` requires a scalar (single-element) output. For non-scalar outputs, use `BackwardWithGrad(gradOutput)` to provide the starting gradient tensor.

After backward, leaf variables hold their accumulated gradients:

```go
fmt.Println(w.Grad())  // dy/dw -- the gradient tensor
fmt.Println(x.Grad())  // dy/dx
```

## Complete Example: Manual Gradient Check

```go
// y = x * w, where x=2, w=3
// dy/dw = x = 2
// dy/dx = w = 3

xT, _ := tensor.FromFloat32([]float32{2.0}, []int64{1})
wT, _ := tensor.FromFloat32([]float32{3.0}, []int64{1})

x := autograd.NewVariable(xT, true)
w := autograd.NewVariable(wT, true)

y := x.Mul(w).Sum()
y.Backward()

wGrad, _ := w.Grad().Float32Data()  // [2.0] -- dy/dw = x
xGrad, _ := x.Grad().Float32Data()  // [3.0] -- dy/dx = w
```

## ZeroGrad

Gradients accumulate across multiple backward passes. Reset them before each training step:

```go
w.ZeroGrad()  // reset gradient to nil
```

In practice you will call `optimizer.ZeroGrad()` which does this for all parameters (see [Tutorial 4](04-training.md)).

## Detach

Stop gradient flow by detaching a variable. This creates a new leaf variable sharing the same tensor data but with no gradient tracking:

```go
detached := v.Detach()
// Operations on detached do not build a graph
```

## RetainGrad

By default, only leaf variables retain their gradients after backward. To keep the gradient of an intermediate variable:

```go
intermediate := x.Matmul(w)
intermediate.RetainGrad()

loss := intermediate.Sum()
loss.Backward()

fmt.Println(intermediate.Grad())  // available because RetainGrad was called
```

## NoGrad: Disabling Tracking for Inference

Wrap inference code in `NoGrad` to skip graph construction. This saves memory and computation:

```go
autograd.NoGrad(func() {
    output := model.Forward(input)
    // No computation graph is built, even if inputs require gradients.
})
```

`NoGrad` blocks can nest safely.

## Error Propagation

Like tensors, variables propagate errors through chains:

```go
result := x.Matmul(w).Add(b)
if err := result.Err(); err != nil {
    log.Fatal(err)
}
```

If any operation produces an error variable, all subsequent operations become no-ops and the error flows to the end of the chain.

---

Next: [Tutorial 3: Modules](03-modules.md)
