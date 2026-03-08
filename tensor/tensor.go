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
// Tensors are reference-counted. The autograd engine calls Retain/Release
// to manage saved-for-backward tensors deterministically, freeing C++
// memory (including VRAM) as soon as backward finishes with each tensor —
// without waiting for Go's garbage collector. A GC finalizer remains as
// a safety net for any tensor not explicitly released.
type Tensor struct {
	raw  *libtorch.Tensor // nil if error tensor or already released
	refs int64            // atomic reference count; 1 from wrap(), 0 for error tensors
	err  error            // non-nil if this tensor represents an error
}

// tracking allocated tensor count for debugging/testing
var activeTensors atomic.Int64

// ActiveTensors returns the number of tensors that haven't been released.
// Useful for detecting leaks in tests.
func ActiveTensors() int64 {
	return activeTensors.Load()
}

// wrap creates a Tensor from a raw libtorch tensor, initializes its
// reference count to 1, and sets up a GC finalizer as a safety net.
// This is the only place where raw tensors enter the public API.
func wrap(raw *libtorch.Tensor) *Tensor {
	t := &Tensor{raw: raw, refs: 1}
	activeTensors.Add(1)
	runtime.SetFinalizer(t, (*Tensor).release)
	return t
}

// errTensor creates a Tensor that carries an error. All operations on it
// are no-ops that propagate the error.
func errTensor(err error) *Tensor {
	return &Tensor{err: err}
}

// Retain increments the reference count, keeping the underlying C++
// tensor alive until a matching Release is called. Used by autograd
// to save tensors for backward without depending on GC timing.
func (t *Tensor) Retain() {
	if t.raw == nil {
		return
	}
	atomic.AddInt64(&t.refs, 1)
}

// release decrements the reference count and frees the C++ tensor
// when it reaches zero. Called by the GC finalizer as a safety net
// and by Release for explicit lifecycle management.
func (t *Tensor) release() {
	if t.raw == nil {
		return
	}
	if atomic.AddInt64(&t.refs, -1) == 0 {
		t.raw.Free()
		t.raw = nil
		activeTensors.Add(-1)
		runtime.SetFinalizer(t, nil)
	}
}

// Release decrements the reference count and frees the underlying
// C++ memory when no references remain. After the last Release,
// the tensor is in an error state and operations on it will return
// an error.
//
// For tensors not managed by autograd, this behaves identically to
// the previous immediate-free semantics (refcount goes from 1 to 0).
// The GC finalizer remains as a safety net for unreleased tensors.
func (t *Tensor) Release() {
	t.release()
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

// RandN creates a tensor with values from a standard normal distribution.
func RandN(shape []int64, opts ...Option) (*Tensor, error) {
	o := applyOptions(opts)
	raw, err := libtorch.RandN(shape, o.dtype.toLibtorch(), o.device.toLibtorch())
	if err != nil {
		return nil, err
	}
	return wrap(raw), nil
}

// Linspace creates a 1D tensor with evenly spaced values from start to end (inclusive).
func Linspace(start, end float64, steps int64, opts ...Option) (*Tensor, error) {
	o := applyOptions(opts)
	raw, err := libtorch.Linspace(start, end, steps, o.dtype.toLibtorch(), o.device.toLibtorch())
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

// FromInt64 creates an Int64 tensor from a Go slice. Data is copied.
// Useful for index tensors (e.g., Embedding lookups).
func FromInt64(data []int64, shape []int64, opts ...Option) (*Tensor, error) {
	o := applyOptions(opts)
	raw, err := libtorch.FromInt64(data, shape, o.device.toLibtorch())
	if err != nil {
		return nil, err
	}
	return wrap(raw), nil
}

// Arange creates a 1D tensor with values [start, start+step, start+2*step, ...) < end.
func Arange(start, end, step float64, opts ...Option) (*Tensor, error) {
	o := applyOptions(opts)
	raw, err := libtorch.Arange(start, end, step, o.dtype.toLibtorch(), o.device.toLibtorch())
	if err != nil {
		return nil, err
	}
	return wrap(raw), nil
}

// ArangeEnd creates a 1D tensor with values [0, 1, 2, ..., end-1]. Shorthand for Arange(0, end, 1).
func ArangeEnd(end float64, opts ...Option) (*Tensor, error) {
	return Arange(0, end, 1, opts...)
}

// Full creates a tensor filled with a single value.
func Full(shape []int64, value float64, opts ...Option) (*Tensor, error) {
	t, err := Ones(shape, opts...)
	if err != nil {
		return nil, err
	}
	result := t.MulScalar(value)
	if err := result.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

// OneHot converts a 1D int64 index tensor [B] to a float32 one-hot tensor [B, C].
// Each row has a 1.0 at the index position and 0.0 elsewhere.
// C is the number of classes, B is the batch size (inferred from indices if 0).
func OneHot(indices *Tensor, nClasses, batchSize int64) *Tensor {
	if err := indices.Err(); err != nil {
		return errTensor(err)
	}
	if batchSize == 0 {
		batchSize = indices.Shape()[0]
	}
	data, err := indices.Float32Data()
	if err != nil {
		return errTensor(err)
	}
	buf := make([]float32, batchSize*nClasses)
	for i := int64(0); i < batchSize; i++ {
		idx := int64(data[i])
		if idx >= 0 && idx < nClasses {
			buf[i*nClasses+idx] = 1.0
		}
	}
	t, terr := FromFloat32(buf, []int64{batchSize, nClasses})
	if terr != nil {
		return errTensor(terr)
	}
	// Match the device of the input indices.
	return t.ToDevice(indices.Device())
}

// Eye creates an n×n identity matrix (float32).
func Eye(n int64, opts ...Option) (*Tensor, error) {
	buf := make([]float32, n*n)
	for i := int64(0); i < n; i++ {
		buf[i*n+i] = 1.0
	}
	return FromFloat32(buf, []int64{n, n}, opts...)
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

// Int64Data copies the tensor data into a Go int64 slice.
// If the tensor is not int64, it is cast first.
func (t *Tensor) Int64Data() ([]int64, error) {
	if err := t.Err(); err != nil {
		return nil, err
	}
	return t.raw.Int64Data()
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
