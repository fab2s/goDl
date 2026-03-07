# Tutorial 1: Tensors

Tensors are the fundamental data type in goDl -- n-dimensional arrays of numbers backed by libtorch. This tutorial covers creation, operations, error handling, and memory management.

## Creating Tensors

All creation functions return `(*tensor.Tensor, error)` since there is no existing tensor to chain from.

```go
import "github.com/fab2s/goDl/tensor"

// From Go data -- data is copied into libtorch
t, err := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})

// Filled tensors
zeros, err := tensor.Zeros([]int64{3, 4})
ones, err := tensor.Ones([]int64{3, 4})

// Random tensors
uniform, err := tensor.Rand([]int64{2, 3})   // values in [0, 1)
normal, err := tensor.RandN([]int64{2, 3})    // standard normal distribution

// Integer tensor (for indices, e.g. Embedding lookups)
idx, err := tensor.FromInt64([]int64{0, 3, 7}, []int64{3})
```

### Options

Pass options to control dtype and device:

```go
t, err := tensor.Ones([]int64{4}, tensor.WithDType(tensor.Float64))
t, err := tensor.Zeros([]int64{3, 3}, tensor.WithDevice(tensor.CUDA))
```

## Shape Inspection

```go
t, _ := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})

t.Shape()  // []int64{2, 3}
t.Ndim()   // 2
t.Numel()  // 6
t.DType()  // tensor.Float32
t.Device() // tensor.CPU
```

## Chainable Operations

Operations return new tensors -- originals are never modified. They are designed to chain fluently:

```go
a, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{2, 2})
b, _ := tensor.FromFloat32([]float32{5, 6, 7, 8}, []int64{2, 2})
c, _ := tensor.Ones([]int64{2, 2})

result := a.Add(b).Matmul(c).ReLU()
```

### Error Propagation

If any operation in a chain fails, the error propagates through the rest of the chain. All subsequent operations become no-ops. Check once at the end:

```go
result := a.Add(b).Matmul(c)
if err := result.Err(); err != nil {
    log.Fatal(err)
}
```

A tensor in an error state (including after `Release()`) carries the error through any operation applied to it.

### Arithmetic

```go
a.Add(b)           // element-wise a + b
a.Sub(b)           // element-wise a - b
a.Mul(b)           // element-wise a * b (Hadamard product)
a.Div(b)           // element-wise a / b
a.Matmul(b)        // matrix multiplication

a.MulScalar(0.5)   // multiply every element by a scalar
a.AddScalar(1.0)   // add a scalar to every element
```

### Activations and Math

```go
t.ReLU()           // max(0, x)
t.Sigmoid()        // 1 / (1 + exp(-x))
t.Tanh()           // hyperbolic tangent
t.Exp()            // element-wise e^x
t.Log()            // element-wise ln(x)
t.Sqrt()           // element-wise square root
t.Neg()            // element-wise negation
t.Softmax(dim)     // softmax along dimension
```

### Reductions

```go
t.Sum()                   // reduce all elements to scalar
t.SumDim(dim, keepdim)    // reduce along one dimension
t.MeanDim(dim, keepdim)   // mean along one dimension
t.MaxDim(dim, keepdim)    // max values along one dimension
```

### Shape Manipulation

```go
t.Reshape([]int64{6, 1})      // new shape, same data
t.Transpose(0, 1)             // swap two dimensions
```

### Slicing and Joining

```go
t.Narrow(dim, start, length)  // extract a contiguous slice along dim
t.Select(dim, index)          // pick one index along dim, removing that dim
t.Cat(other, dim)             // concatenate two tensors along dim
t.IndexSelect(dim, indices)   // gather slices at given indices

// Stack multiple tensors along a new dimension
tensor.Stack([]*tensor.Tensor{a, b, c}, 0)
```

## Extracting Data

Copy tensor data back to Go slices:

```go
t, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})

data, err := t.Float32Data()  // []float32{1, 2, 3}
```

`Float64Data()` is also available. The tensor is moved to CPU if necessary (without modifying the original).

## Memory Management

Tensors are backed by C++ memory managed through libtorch. The Go garbage collector handles cleanup automatically via finalizers -- you do not need to free tensors manually in normal code.

For hot paths where you want tighter control over memory:

```go
t, _ := tensor.Zeros([]int64{1000, 1000})
// ... use t ...
t.Release()  // free immediately, don't wait for GC
// t is now in an error state; further ops on it return an error
```

For debugging memory leaks in tests, `tensor.ActiveTensors()` returns the count of tensors that have not yet been released.

## Device Transfer

```go
gpuTensor := t.ToCUDA()   // move to GPU
cpuTensor := t.ToCPU()    // move back to CPU
custom := t.ToDevice(tensor.CUDA)
```

---

Next: [Tutorial 2: Automatic Differentiation](02-autograd.md)
