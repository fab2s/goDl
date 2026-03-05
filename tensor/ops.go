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
