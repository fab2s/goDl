package tensor

import (
	"fmt"

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

// Exp returns element-wise exponential.
func (t *Tensor) Exp() *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Exp(t.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Log returns element-wise natural logarithm.
func (t *Tensor) Log() *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Log(t.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// AddScalar adds a scalar to every element.
func (t *Tensor) AddScalar(scalar float64) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.AddScalar(t.raw, scalar)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Neg returns element-wise negation.
func (t *Tensor) Neg() *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Neg(t.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// MaxDim returns max values along a dimension.
func (t *Tensor) MaxDim(dim int, keepdim bool) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.MaxDim(t.raw, dim, keepdim)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Softmax applies softmax along a dimension.
func (t *Tensor) Softmax(dim int) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Softmax(t.raw, dim)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Select picks a single index along a dimension, removing that dimension.
func (t *Tensor) Select(dim int, index int64) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Select(t.raw, dim, index)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// ZerosLike creates a tensor of zeros with the same shape, dtype, and device.
func (t *Tensor) ZerosLike() *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.ZerosLike(t.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// SelectScatter returns a copy of t with the slice at (dim, index) replaced by src.
func (t *Tensor) SelectScatter(src *Tensor, dim int, index int64) *Tensor {
	if !t.valid() {
		return t
	}
	if !src.valid() {
		return src
	}
	raw, err := libtorch.SelectScatter(t.raw, src.raw, dim, index)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Narrow extracts a slice along dim: t[dim, start:start+length].
func (t *Tensor) Narrow(dim int, start, length int64) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Narrow(t.raw, dim, start, length)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// NarrowScatter returns t with the narrow slice at (dim, start) replaced by src.
func (t *Tensor) NarrowScatter(src *Tensor, dim int, start int64) *Tensor {
	if !t.valid() {
		return t
	}
	if !src.valid() {
		return src
	}
	raw, err := libtorch.NarrowScatter(t.raw, src.raw, dim, start)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Cat concatenates two tensors along dim.
func (t *Tensor) Cat(other *Tensor, dim int) *Tensor {
	if !t.valid() {
		return t
	}
	if !other.valid() {
		return other
	}
	raw, err := libtorch.Cat2(t.raw, other.raw, dim)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// MeanDim computes the mean along a single dimension.
func (t *Tensor) MeanDim(dim int, keepdim bool) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.MeanDim(t.raw, dim, keepdim)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// IndexSelect gathers slices along dim at the given indices (Int64 tensor).
func (t *Tensor) IndexSelect(dim int, index *Tensor) *Tensor {
	if !t.valid() {
		return t
	}
	if !index.valid() {
		return index
	}
	raw, err := libtorch.IndexSelect(t.raw, dim, index.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// IndexAdd returns t with src added at positions given by index along dim.
func (t *Tensor) IndexAdd(dim int, index *Tensor, src *Tensor) *Tensor {
	if !t.valid() {
		return t
	}
	if !index.valid() {
		return index
	}
	if !src.valid() {
		return src
	}
	raw, err := libtorch.IndexAdd(t.raw, dim, index.raw, src.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Sqrt returns element-wise square root.
func (t *Tensor) Sqrt() *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.Sqrt(t.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Div returns element-wise division t / other.
func (t *Tensor) Div(other *Tensor) *Tensor {
	if !t.valid() {
		return t
	}
	if !other.valid() {
		return other
	}
	raw, err := libtorch.Div(t.raw, other.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// --- Convolution ---

// Conv2d performs a 2D convolution. bias may be nil.
// Input shape: [N, C_in, H, W]. Weight shape: [C_out, C_in/groups, kH, kW].
// Bias shape: [C_out] or nil.
func (t *Tensor) Conv2d(weight, bias *Tensor, stride, padding, dilation []int64, groups int64) *Tensor {
	if !t.valid() {
		return t
	}
	if !weight.valid() {
		return weight
	}
	var biasRaw *libtorch.Tensor
	if bias != nil {
		if !bias.valid() {
			return bias
		}
		biasRaw = bias.raw
	}
	raw, err := libtorch.Conv2d(t.raw, weight.raw, biasRaw, stride, padding, dilation, groups)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Conv2dBackward computes gradients for a 2D convolution.
// Returns (gradInput, gradWeight, gradBias). gradBias is nil if computeBias is false.
func Conv2dBackward(gradOutput, input, weight *Tensor, stride, padding, dilation []int64, groups int64, computeBias bool) (gradInput, gradWeight, gradBias *Tensor) {
	giRaw, gwRaw, gbRaw, err := libtorch.Conv2dBackward(
		gradOutput.raw, input.raw, weight.raw,
		stride, padding, dilation, groups, computeBias,
	)
	if err != nil {
		e := errTensor(err)
		return e, e, e
	}
	gradInput = wrap(giRaw)
	gradWeight = wrap(gwRaw)
	if gbRaw != nil {
		gradBias = wrap(gbRaw)
	}
	return gradInput, gradWeight, gradBias
}

// --- Expand (broadcast) ---

// Expand broadcasts the tensor to a larger shape. -1 keeps the existing size.
// The result shares data where possible (like a view). No gradient support needed
// — expand is a tensor creation/shaping op used to build sampling grids.
func (t *Tensor) Expand(shape []int64) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := t.raw.Expand(shape)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// --- Transposed convolution ---

// ConvTranspose2d performs a 2D transposed convolution (deconvolution). bias may be nil.
// Input shape: [N, C_in, H, W]. Weight shape: [C_in, C_out/groups, kH, kW].
func (t *Tensor) ConvTranspose2d(weight, bias *Tensor, stride, padding, outputPadding, dilation []int64, groups int64) *Tensor {
	if !t.valid() {
		return t
	}
	if !weight.valid() {
		return weight
	}
	var biasRaw *libtorch.Tensor
	if bias != nil {
		if !bias.valid() {
			return bias
		}
		biasRaw = bias.raw
	}
	raw, err := libtorch.ConvTranspose2d(t.raw, weight.raw, biasRaw, stride, padding, outputPadding, dilation, groups)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// ConvTranspose2dBackward computes gradients for a 2D transposed convolution.
func ConvTranspose2dBackward(gradOutput, input, weight *Tensor, stride, padding, outputPadding, dilation []int64, groups int64, computeBias bool) (gradInput, gradWeight, gradBias *Tensor) {
	giRaw, gwRaw, gbRaw, err := libtorch.ConvTranspose2dBackward(
		gradOutput.raw, input.raw, weight.raw,
		stride, padding, outputPadding, dilation, groups, computeBias,
	)
	if err != nil {
		e := errTensor(err)
		return e, e, e
	}
	gradInput = wrap(giRaw)
	gradWeight = wrap(gwRaw)
	if gbRaw != nil {
		gradBias = wrap(gbRaw)
	}
	return gradInput, gradWeight, gradBias
}

// --- Adaptive average pooling ---

// AdaptiveAvgPool2d performs 2D adaptive average pooling.
// Input shape: [N, C, H, W]. Pools to outputSize [H_out, W_out].
func (t *Tensor) AdaptiveAvgPool2d(outputSize []int64) *Tensor {
	if !t.valid() {
		return t
	}
	raw, err := libtorch.AdaptiveAvgPool2d(t.raw, outputSize)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// AdaptiveAvgPool2dBackward computes the gradient for adaptive avg pool.
func AdaptiveAvgPool2dBackward(gradOutput, input *Tensor) *Tensor {
	raw, err := libtorch.AdaptiveAvgPool2dBackward(gradOutput.raw, input.raw)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// --- Grid sampling ---

// GridSample performs 2D grid sampling with bilinear interpolation.
// Input shape: [N, C, H, W]. Grid shape: [N, H_out, W_out, 2].
// Grid coordinates are in [-1, 1] when alignCorners is true.
// mode: 0=bilinear, 1=nearest, 2=bicubic.
// paddingMode: 0=zeros, 1=border, 2=reflection.
func (t *Tensor) GridSample(grid *Tensor, mode, paddingMode int, alignCorners bool) *Tensor {
	if !t.valid() {
		return t
	}
	if !grid.valid() {
		return grid
	}
	raw, err := libtorch.GridSample(t.raw, grid.raw, mode, paddingMode, alignCorners)
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// GridSampleBackward computes gradients for grid sampling.
// Returns (gradInput, gradGrid).
func GridSampleBackward(gradOutput, input, grid *Tensor, mode, paddingMode int, alignCorners bool) (gradInput, gradGrid *Tensor) {
	giRaw, ggRaw, err := libtorch.GridSampleBackward(
		gradOutput.raw, input.raw, grid.raw,
		mode, paddingMode, alignCorners,
	)
	if err != nil {
		e := errTensor(err)
		return e, e
	}
	return wrap(giRaw), wrap(ggRaw)
}

// --- Stacking ---

// Stack concatenates tensors along a new dimension.
// All tensors must have the same shape.
func Stack(tensors []*Tensor, dim int) *Tensor {
	if len(tensors) == 0 {
		return errTensor(fmt.Errorf("tensor: Stack requires at least one tensor"))
	}
	// Unsqueeze each tensor at dim via Reshape, then Cat.
	base := tensors[0]
	if !base.valid() {
		return base
	}
	shape := base.Shape()
	newShape := make([]int64, len(shape)+1)
	copy(newShape, shape[:dim])
	newShape[dim] = 1
	copy(newShape[dim+1:], shape[dim:])

	result := base.Reshape(newShape)
	for i := 1; i < len(tensors); i++ {
		t := tensors[i]
		if !t.valid() {
			return t
		}
		result = result.Cat(t.Reshape(newShape), dim)
	}
	return result
}

// --- Dtype casting ---

// ToDType casts the tensor to a different element type. Returns a new tensor.
// No-op if the tensor is already the target dtype.
func (t *Tensor) ToDType(dtype DType) *Tensor {
	if !t.valid() {
		return t
	}
	if t.DType() == dtype {
		return t
	}
	raw, err := libtorch.ToDType(t.raw, dtype.toLibtorch())
	if err != nil {
		return errTensor(err)
	}
	return wrap(raw)
}

// Half casts the tensor to float16. Shorthand for ToDType(Float16).
func (t *Tensor) Half() *Tensor {
	return t.ToDType(Float16)
}

// BFloat16 casts the tensor to bfloat16. Shorthand for ToDType(BFloat16).
func (t *Tensor) ToBFloat16() *Tensor {
	return t.ToDType(BFloat16)
}

// Float casts the tensor to float32. Shorthand for ToDType(Float32).
func (t *Tensor) Float() *Tensor {
	return t.ToDType(Float32)
}

// AllFinite returns true if all elements are finite (no inf, no nan).
func (t *Tensor) AllFinite() bool {
	if !t.valid() {
		return false
	}
	ok, err := libtorch.AllFinite(t.raw)
	if err != nil {
		return false
	}
	return ok
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
