package tensor

import (
	"github.com/fab2s/goDl/internal/libtorch"
)

// Operations are chainable methods. If the receiver carries an error,
// the operation is a no-op and the error propagates:
//
//	result := a.Matmul(b).Add(c).ReLU()
//	if err := result.Err(); err != nil { ... }
//
// Operations return new tensors — the originals are never modified.

// --- Binary operations ---

// Add returns the element-wise sum of t and other.
func (t *Tensor) Add(other *Tensor) *Tensor {
	if !t.valid() {
		return t
	}
	if !other.valid() {
		return other
	}
	raw, err := libtorch.Add(t.raw, other.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Mul returns the element-wise product of t and other.
func (t *Tensor) Mul(other *Tensor) *Tensor {
	if !t.valid() {
		return t
	}
	if !other.valid() {
		return other
	}
	raw, err := libtorch.Mul(t.raw, other.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Matmul returns the matrix product of t and other.
func (t *Tensor) Matmul(other *Tensor) *Tensor {
	if !t.valid() {
		return t
	}
	if !other.valid() {
		return other
	}
	raw, err := libtorch.Matmul(t.raw, other.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Sub returns the element-wise difference t - other.
func (t *Tensor) Sub(other *Tensor) *Tensor {
	if !t.valid() {
		return t
	}
	if !other.valid() {
		return other
	}
	raw, err := libtorch.Sub(t.raw, other.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// --- Unary operations ---

// ReLU applies the rectified linear unit activation: max(0, x).
func (t *Tensor) ReLU() *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.ReLU(t.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Sigmoid applies the sigmoid activation function: 1 / (1 + exp(-x)).
func (t *Tensor) Sigmoid() *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Sigmoid(t.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Tanh applies the hyperbolic tangent activation function.
func (t *Tensor) Tanh() *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Tanh(t.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Transpose swaps two dimensions.
func (t *Tensor) Transpose(dim0, dim1 int) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Transpose(t.raw, dim0, dim1)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Sum reduces all elements to a scalar tensor.
func (t *Tensor) Sum() *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Sum(t.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// SumDim reduces along a single dimension.
func (t *Tensor) SumDim(dim int, keepdim bool) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.SumDim(t.raw, dim, keepdim)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// OnesLike creates a tensor of ones with the same shape, dtype, and device.
func (t *Tensor) OnesLike() *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.OnesLike(t.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// MulScalar multiplies every element by a scalar value.
func (t *Tensor) MulScalar(scalar float64) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.MulScalar(t.raw, scalar)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// GTScalar returns a float mask: 1.0 where element > scalar, else 0.0.
func (t *Tensor) GTScalar(scalar float64) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.GTScalar(t.raw, scalar)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Reshape returns a tensor with the given shape.
func (t *Tensor) Reshape(shape []int64) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Reshape(t.raw, shape)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// --- Device operations ---

// ToDevice moves the tensor to the specified device. Returns a new tensor.
func (t *Tensor) ToDevice(device Device) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := t.raw.ToDevice(device.toLibtorch())
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// ToCPU is shorthand for ToDevice(CPU).
func (t *Tensor) ToCPU() *Tensor {
	return t.ToDevice(CPU)
}

// ToCUDA is shorthand for ToDevice(CUDA).
func (t *Tensor) ToCUDA() *Tensor {
	return t.ToDevice(CUDA)
}
