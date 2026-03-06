package tensor

import (
	"fmt"
	"runtime"
	"sync/atomic"

	"github.com/fab2s/goDl/internal/libtorch"
)

// Tensor is an n-dimensional array of numbers, backed by libtorch.
//
// Tensors carry an error state for chainable operations. If any operation
// in a chain fails, subsequent operations become no-ops and the error
// propagates to the end:
//
//	result := x.Matmul(w).Add(b).ReLU()
//	if err := result.Err(); err != nil { ... }
//
// Tensors are safe to use after the Go garbage collector collects them —
// a finalizer releases the underlying C memory. For tighter control in
// hot loops, use Release() or Scopes.
type Tensor struct {
	raw *libtorch.Tensor // nil if error tensor or already released
	err error            // non-nil if this tensor represents an error
}

// tracking allocated tensor count for debugging/testing
var activeTensors atomic.Int64

// ActiveTensors returns the number of tensors that haven't been released.
// Useful for detecting leaks in tests.
func ActiveTensors() int64 {
	return activeTensors.Load()
}

// wrap creates a Tensor from a raw libtorch tensor and sets up the
// GC finalizer. This is the only place where raw tensors enter the
// public API.
func wrap(raw *libtorch.Tensor) *Tensor {
	t := &Tensor{raw: raw}
	activeTensors.Add(1)
	runtime.SetFinalizer(t, func(t *Tensor) {
		t.release()
	})
	return t
}

// errTensor creates a Tensor that carries an error. All operations on it
// are no-ops that propagate the error.
func errTensor(err error) *Tensor {
	return &Tensor{err: err}
}

// release frees the underlying C tensor. Safe to call multiple times.
func (t *Tensor) release() {
	if t.raw != nil {
		t.raw.Free()
		t.raw = nil
		activeTensors.Add(-1)
	}
}

// Release explicitly frees the tensor's underlying memory. After calling
// Release, the tensor is in an error state and operations on it will
// return an error.
//
// This is optional — the GC finalizer handles cleanup automatically.
// Use Release in hot loops or when memory pressure is a concern.
func (t *Tensor) Release() {
	t.release()
	// Clear the finalizer since we've already cleaned up.
	runtime.SetFinalizer(t, nil)
}

// Err returns the error carried by this tensor, or nil if the tensor
// is valid. Check this after a chain of operations:
//
//	result := a.Add(b).Matmul(c)
//	if err := result.Err(); err != nil { ... }
func (t *Tensor) Err() error {
	if t.err != nil {
		return t.err
	}
	if t.raw == nil {
		return fmt.Errorf("tensor: use after release")
	}
	return nil
}

// valid returns true if the tensor can be used in operations.
func (t *Tensor) valid() bool {
	return t.err == nil && t.raw != nil
}

// Raw returns the underlying libtorch tensor handle.
// This is exported for use by sibling packages (autograd) and should
// not be used by end users.
func (t *Tensor) Raw() *libtorch.Tensor {
	return t.raw
}

// WrapRaw creates a managed Tensor from a raw libtorch tensor.
// Exported for use by sibling packages (autograd).
func WrapRaw(raw *libtorch.Tensor) *Tensor {
	return wrap(raw)
}

// --- Creation functions ---
// These return (*Tensor, error) since there is nothing to chain from.

// Zeros creates a tensor filled with zeros.
func Zeros(shape []int64, opts ...Option) (*Tensor, error) {
	o := applyOptions(opts)
	raw, err := libtorch.Zeros(shape, o.dtype.toLibtorch(), o.device.toLibtorch())
	if err != nil {
		return nil, err
	}
	return wrap(raw), nil
}

// Ones creates a tensor filled with ones.
func Ones(shape []int64, opts ...Option) (*Tensor, error) {
	raw, err := libtorch.Ones(shape, applyOptions(opts).dtype.toLibtorch(),
		applyOptions(opts).device.toLibtorch())
	if err != nil {
		return nil, err
	}
	return wrap(raw), nil
}

// Rand creates a tensor with uniform random values in [0, 1).
func Rand(shape []int64, opts ...Option) (*Tensor, error) {
	o := applyOptions(opts)
	raw, err := libtorch.Rand(shape, o.dtype.toLibtorch(), o.device.toLibtorch())
	if err != nil {
		return nil, err
	}
	return wrap(raw), nil
}

// FromFloat32 creates a tensor from a Go slice. Data is copied.
func FromFloat32(data []float32, shape []int64, opts ...Option) (*Tensor, error) {
	o := applyOptions(opts)
	raw, err := libtorch.FromFloat32(data, shape, o.device.toLibtorch())
	if err != nil {
		return nil, err
	}
	return wrap(raw), nil
}

// FromFloat64 creates a tensor from a Go slice. Data is copied.
func FromFloat64(data []float64, shape []int64, opts ...Option) (*Tensor, error) {
	o := applyOptions(opts)
	raw, err := libtorch.FromFloat64(data, shape, o.device.toLibtorch())
	if err != nil {
		return nil, err
	}
	return wrap(raw), nil
}

// --- Metadata ---

// Shape returns the full shape as a slice.
func (t *Tensor) Shape() []int64 {
	if !t.valid() {
		return nil
	}
	return t.raw.Shapes()
}

// Ndim returns the number of dimensions.
func (t *Tensor) Ndim() int {
	if !t.valid() {
		return 0
	}
	return t.raw.Ndim()
}

// Numel returns the total number of elements.
func (t *Tensor) Numel() int64 {
	if !t.valid() {
		return 0
	}
	return t.raw.Numel()
}

// DType returns the element type.
func (t *Tensor) DType() DType {
	if !t.valid() {
		return 0
	}
	return DType(t.raw.DType())
}

// Device returns where the tensor lives (CPU or CUDA).
func (t *Tensor) Device() Device {
	if !t.valid() {
		return CPU
	}
	return Device(t.raw.Device())
}

// --- Data access ---

// Float32Data copies the tensor data into a Go float32 slice.
// The tensor is moved to CPU if necessary (without modifying the original).
func (t *Tensor) Float32Data() ([]float32, error) {
	if err := t.Err(); err != nil {
		return nil, err
	}
	return t.raw.Float32Data()
}

// Float64Data copies the tensor data into a Go float64 slice.
func (t *Tensor) Float64Data() ([]float64, error) {
	if err := t.Err(); err != nil {
		return nil, err
	}
	return t.raw.Float64Data()
}

// String returns a human-readable summary of the tensor.
func (t *Tensor) String() string {
	if t.err != nil {
		return fmt.Sprintf("Tensor(<error: %s>)", t.err)
	}
	if t.raw == nil {
		return "Tensor(<released>)"
	}
	return fmt.Sprintf("Tensor(shape=%v, dtype=%s, device=%s)",
		t.Shape(), t.DType(), t.Device())
}
