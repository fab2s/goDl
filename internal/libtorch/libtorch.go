// Package libtorch provides raw CGo bindings to libtorch via a C shim.
//
// This is the lowest level of goDl — it exposes C tensor handles and
// operations directly. Higher-level packages (tensor/, autograd/) wrap
// these with safe Go types.
//
// Every function that can fail returns an error. Errors originate from
// libtorch C++ exceptions caught in the shim layer.
package libtorch

// CGo preamble: tell the Go compiler how to build and link the C++ shim.
//
// The shim.cpp file is compiled by CGo as C++ (via the .cpp extension).
// CGo automatically compiles .c/.cpp files in the same directory.
//
// Base LDFLAGS link against core libtorch libraries. Backend-specific
// libraries (CUDA, ROCm) are added via build tags in separate files:
//   - libtorch_cuda.go  — adds -ltorch_cuda -lc10_cuda
//   - libtorch_cpu.go   — CPU-only, no extra libs
//
// The LIBTORCH_PATH environment variable must point to the libtorch
// installation. The Dockerfile sets this automatically.

/*
#cgo CXXFLAGS: -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1
#cgo LDFLAGS: -lstdc++ -ltorch -ltorch_cpu -lc10
#include "shim.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// DType represents a tensor's element type.
type DType int

const (
	Float32 DType = C.GODL_FLOAT32
	Float64 DType = C.GODL_FLOAT64
	Int32   DType = C.GODL_INT32
	Int64   DType = C.GODL_INT64
)

// Device represents where a tensor lives (CPU or CUDA).
type Device int

const (
	CPU  Device = C.GODL_CPU
	CUDA Device = C.GODL_CUDA
)

// Tensor is an opaque handle to a libtorch tensor.
// It must be freed with Free() when no longer needed.
type Tensor struct {
	handle C.TorchTensor
}

// checkErr converts a C error string to a Go error, freeing the C string.
// Returns nil if the C string is NULL (success).
func checkErr(cerr *C.char) error {
	if cerr == nil {
		return nil
	}
	msg := C.GoString(cerr)
	C.godl_free_string(cerr)
	return fmt.Errorf("libtorch: %s", msg)
}

// --- Tensor creation ---

// Zeros creates a tensor filled with zeros.
func Zeros(shape []int64, dtype DType, device Device) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_zeros(
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int(len(shape)),
		C.int(dtype),
		C.int(device),
		&handle,
	)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Ones creates a tensor filled with ones.
func Ones(shape []int64, dtype DType, device Device) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_ones(
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int(len(shape)),
		C.int(dtype),
		C.int(device),
		&handle,
	)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Rand creates a tensor filled with uniform random values in [0, 1).
func Rand(shape []int64, dtype DType, device Device) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_rand(
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int(len(shape)),
		C.int(dtype),
		C.int(device),
		&handle,
	)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// FromFloat32 creates a tensor from a Go float32 slice.
// The data is copied — the Go slice can be modified or GC'd after this call.
func FromFloat32(data []float32, shape []int64, device Device) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_from_blob(
		unsafe.Pointer(&data[0]),
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int(len(shape)),
		C.int(Float32),
		C.int(device),
		&handle,
	)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// FromFloat64 creates a tensor from a Go float64 slice.
func FromFloat64(data []float64, shape []int64, device Device) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_from_blob(
		unsafe.Pointer(&data[0]),
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int(len(shape)),
		C.int(Float64),
		C.int(device),
		&handle,
	)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// FromInt64 creates a tensor from a Go int64 slice.
func FromInt64(data []int64, shape []int64, device Device) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_from_blob(
		unsafe.Pointer(&data[0]),
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int(len(shape)),
		C.int(Int64),
		C.int(device),
		&handle,
	)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// --- Tensor lifecycle ---

// Free releases the underlying libtorch tensor. Must be called exactly once.
func (t *Tensor) Free() {
	if t.handle != nil {
		C.godl_free_tensor(t.handle)
		t.handle = nil
	}
}

// --- Tensor metadata ---

// Ndim returns the number of dimensions.
func (t *Tensor) Ndim() int {
	return int(C.godl_ndim(t.handle))
}

// Shape returns the size of the given dimension.
func (t *Tensor) Shape(dim int) int64 {
	return int64(C.godl_shape(t.handle, C.int(dim)))
}

// Shapes returns the full shape as a slice.
func (t *Tensor) Shapes() []int64 {
	ndim := t.Ndim()
	shape := make([]int64, ndim)
	for i := 0; i < ndim; i++ {
		shape[i] = t.Shape(i)
	}
	return shape
}

// DType returns the tensor's element type.
func (t *Tensor) DType() DType {
	return DType(C.godl_dtype(t.handle))
}

// Device returns which device the tensor lives on.
func (t *Tensor) Device() Device {
	return Device(C.godl_device(t.handle))
}

// Numel returns the total number of elements.
func (t *Tensor) Numel() int64 {
	return int64(C.godl_numel(t.handle))
}

// --- Data access ---

// Float32Data copies the tensor data into a Go float32 slice.
// The tensor is moved to CPU if necessary.
func (t *Tensor) Float32Data() ([]float32, error) {
	n := t.Numel()
	buf := make([]float32, n)
	cerr := C.godl_copy_data(
		t.handle,
		unsafe.Pointer(&buf[0]),
		C.int64_t(n*4), // 4 bytes per float32
	)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return buf, nil
}

// Float64Data copies the tensor data into a Go float64 slice.
func (t *Tensor) Float64Data() ([]float64, error) {
	n := t.Numel()
	buf := make([]float64, n)
	cerr := C.godl_copy_data(
		t.handle,
		unsafe.Pointer(&buf[0]),
		C.int64_t(n*8), // 8 bytes per float64
	)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return buf, nil
}

// --- Basic operations ---

// Add returns a + b (element-wise).
func Add(a, b *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_add(a.handle, b.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Mul returns a * b (element-wise).
func Mul(a, b *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_mul(a.handle, b.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Matmul returns the matrix product of a and b.
func Matmul(a, b *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_matmul(a.handle, b.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// ReLU applies rectified linear unit activation.
func ReLU(a *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_relu(a.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Sigmoid applies the sigmoid activation function.
func Sigmoid(a *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_sigmoid(a.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Tanh applies the hyperbolic tangent activation function.
func Tanh(a *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_tanh_op(a.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// --- Additional operations (used by autograd backward) ---

// Sub returns a - b (element-wise).
func Sub(a, b *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_sub(a.handle, b.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Transpose swaps two dimensions.
func Transpose(t *Tensor, dim0, dim1 int) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_transpose(t.handle, C.int(dim0), C.int(dim1), &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Sum reduces all elements to a single scalar tensor.
func Sum(t *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_sum(t.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// SumDim reduces along a single dimension.
func SumDim(t *Tensor, dim int, keepdim bool) (*Tensor, error) {
	var handle C.TorchTensor
	kd := C.int(0)
	if keepdim {
		kd = 1
	}
	cerr := C.godl_sum_dim(t.handle, C.int(dim), kd, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// OnesLike creates a tensor of ones with the same shape, dtype, and device.
func OnesLike(t *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_ones_like(t.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// MulScalar multiplies every element by a scalar.
func MulScalar(t *Tensor, scalar float64) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_mul_scalar(t.handle, C.double(scalar), &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// GTScalar returns a float mask where each element is 1.0 if > scalar, else 0.0.
func GTScalar(t *Tensor, scalar float64) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_gt_scalar(t.handle, C.double(scalar), &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Reshape returns a tensor with the given shape.
func Reshape(t *Tensor, shape []int64) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_reshape(
		t.handle,
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int(len(shape)),
		&handle,
	)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Exp returns element-wise exponential.
func Exp(t *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_exp(t.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Log returns element-wise natural logarithm.
func Log(t *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_log(t.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// RandN creates a tensor with values from a standard normal distribution.
func RandN(shape []int64, dtype DType, device Device) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_randn(
		(*C.int64_t)(unsafe.Pointer(&shape[0])),
		C.int(len(shape)),
		C.int(dtype),
		C.int(device),
		&handle,
	)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// AddScalar adds a scalar to every element.
func AddScalar(t *Tensor, scalar float64) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_add_scalar(t.handle, C.double(scalar), &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Neg returns element-wise negation.
func Neg(t *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_neg(t.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// MaxDim returns the max values along a dimension.
func MaxDim(t *Tensor, dim int, keepdim bool) (*Tensor, error) {
	var handle C.TorchTensor
	kd := C.int(0)
	if keepdim {
		kd = 1
	}
	cerr := C.godl_max_dim(t.handle, C.int(dim), kd, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Softmax applies softmax along a dimension.
func Softmax(t *Tensor, dim int) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_softmax(t.handle, C.int(dim), &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Select picks a single index along a dimension, removing that dimension.
func Select(t *Tensor, dim int, index int64) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_select(t.handle, C.int(dim), C.int64_t(index), &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// ZerosLike creates a tensor of zeros with the same shape, dtype, and device.
func ZerosLike(t *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_zeros_like(t.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// SelectScatter returns a copy of input with the slice at (dim, index) replaced by src.
func SelectScatter(input, src *Tensor, dim int, index int64) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_select_scatter(input.handle, src.handle, C.int(dim), C.int64_t(index), &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// --- Slicing and concatenation ---

// Narrow extracts a slice along dim starting at start with the given length.
func Narrow(t *Tensor, dim int, start, length int64) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_narrow(t.handle, C.int(dim), C.int64_t(start), C.int64_t(length), &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// NarrowScatter returns input with the narrow slice at (dim, start) replaced by src.
func NarrowScatter(input, src *Tensor, dim int, start int64) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_narrow_scatter(input.handle, src.handle, C.int(dim), C.int64_t(start), &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Cat2 concatenates two tensors along dim.
func Cat2(a, b *Tensor, dim int) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_cat2(a.handle, b.handle, C.int(dim), &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// --- Reduction ---

// MeanDim computes the mean along a single dimension.
func MeanDim(t *Tensor, dim int, keepdim bool) (*Tensor, error) {
	var handle C.TorchTensor
	kd := C.int(0)
	if keepdim {
		kd = 1
	}
	cerr := C.godl_mean_dim(t.handle, C.int(dim), kd, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// --- Indexing ---

// IndexSelect gathers slices along dim at the given indices.
func IndexSelect(t *Tensor, dim int, index *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_index_select(t.handle, C.int(dim), index.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// IndexAdd returns t with src added at positions given by index along dim.
func IndexAdd(t *Tensor, dim int, index *Tensor, src *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_index_add(t.handle, C.int(dim), index.handle, src.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// --- Element-wise math ---

// Sqrt returns element-wise square root.
func Sqrt(t *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_sqrt(t.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Div returns a / b (element-wise).
func Div(a, b *Tensor) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_div(a.handle, b.handle, &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// --- Convolution ---

// Conv2d performs a 2D convolution. bias may be nil.
// stride, padding, dilation must each be 2-element slices.
func Conv2d(input, weight, bias *Tensor, stride, padding, dilation []int64, groups int64) (*Tensor, error) {
	var biasHandle C.TorchTensor
	if bias != nil {
		biasHandle = bias.handle
	}
	var handle C.TorchTensor
	cerr := C.godl_conv2d(
		input.handle, weight.handle, biasHandle,
		(*C.int64_t)(unsafe.Pointer(&stride[0])),
		(*C.int64_t)(unsafe.Pointer(&padding[0])),
		(*C.int64_t)(unsafe.Pointer(&dilation[0])),
		C.int64_t(groups),
		&handle,
	)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// Conv2dBackward computes gradients for a 2D convolution.
// Always returns gradInput and gradWeight. Returns gradBias only if computeBias is true.
func Conv2dBackward(gradOutput, input, weight *Tensor, stride, padding, dilation []int64, groups int64, computeBias bool) (gradInput, gradWeight, gradBias *Tensor, err error) {
	var giHandle, gwHandle, gbHandle C.TorchTensor
	cb := C.int(0)
	if computeBias {
		cb = 1
	}
	cerr := C.godl_conv2d_backward(
		gradOutput.handle, input.handle, weight.handle,
		(*C.int64_t)(unsafe.Pointer(&stride[0])),
		(*C.int64_t)(unsafe.Pointer(&padding[0])),
		(*C.int64_t)(unsafe.Pointer(&dilation[0])),
		C.int64_t(groups),
		cb,
		&giHandle, &gwHandle, &gbHandle,
	)
	if err := checkErr(cerr); err != nil {
		return nil, nil, nil, err
	}
	gradInput = &Tensor{handle: giHandle}
	gradWeight = &Tensor{handle: gwHandle}
	if computeBias {
		gradBias = &Tensor{handle: gbHandle}
	}
	return gradInput, gradWeight, gradBias, nil
}

// --- Device operations ---

// ToDevice moves a tensor to the specified device. Returns a new tensor.
func (t *Tensor) ToDevice(device Device) (*Tensor, error) {
	var handle C.TorchTensor
	cerr := C.godl_to_device(t.handle, C.int(device), &handle)
	if err := checkErr(cerr); err != nil {
		return nil, err
	}
	return &Tensor{handle: handle}, nil
}

// --- Utility ---

// CUDAAvailable returns true if CUDA is available.
func CUDAAvailable() bool {
	return C.godl_cuda_is_available() != 0
}

// CUDADeviceCount returns the number of available CUDA devices.
func CUDADeviceCount() int {
	return int(C.godl_cuda_device_count())
}
