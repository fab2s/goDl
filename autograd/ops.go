package autograd

import "github.com/fab2s/goDl/tensor"

// --- Binary operations ---

// Add returns the element-wise sum of v and other.
// Backward: grad_v = gradOutput, grad_other = gradOutput.
func (v *Variable) Add(other *Variable) *Variable {
	if !v.valid() {
		return v
	}
	if !other.valid() {
		return other
	}
	result := v.data.Add(other.data)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if needsGrad(v, other) {
		vShape := v.data.Shape()
		otherShape := other.data.Shape()
		fn = &gradFn{
			name:   "AddBackward",
			inputs: []*Variable{v, other},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				gradV := unbroadcast(grad, vShape)
				gradOther := unbroadcast(grad, otherShape)
				return []*tensor.Tensor{gradV, gradOther}
			},
		}
	}
	return newVar(result, fn)
}

// Sub returns the element-wise difference v - other.
// Backward: grad_v = gradOutput, grad_other = -gradOutput.
func (v *Variable) Sub(other *Variable) *Variable {
	if !v.valid() {
		return v
	}
	if !other.valid() {
		return other
	}
	result := v.data.Sub(other.data)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if needsGrad(v, other) {
		vShape := v.data.Shape()
		otherShape := other.data.Shape()
		fn = &gradFn{
			name:   "SubBackward",
			inputs: []*Variable{v, other},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				gradV := unbroadcast(grad, vShape)
				gradOther := unbroadcast(grad.MulScalar(-1), otherShape)
				return []*tensor.Tensor{gradV, gradOther}
			},
		}
	}
	return newVar(result, fn)
}

// Mul returns the element-wise product of v and other.
// Backward: grad_v = gradOutput * other, grad_other = gradOutput * v.
func (v *Variable) Mul(other *Variable) *Variable {
	if !v.valid() {
		return v
	}
	if !other.valid() {
		return other
	}
	result := v.data.Mul(other.data)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if needsGrad(v, other) {
		vData := v.data
		otherData := other.data
		vShape := v.data.Shape()
		otherShape := other.data.Shape()
		fn = &gradFn{
			name:   "MulBackward",
			inputs: []*Variable{v, other},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				gradV := unbroadcast(grad.Mul(otherData), vShape)
				gradOther := unbroadcast(grad.Mul(vData), otherShape)
				return []*tensor.Tensor{gradV, gradOther}
			},
		}
	}
	return newVar(result, fn)
}

// Matmul returns the matrix product of v and other.
// For 2D tensors: v [M,K] @ other [K,N] = result [M,N].
// Backward: grad_v = gradOutput @ otherᵀ, grad_other = vᵀ @ gradOutput.
func (v *Variable) Matmul(other *Variable) *Variable {
	if !v.valid() {
		return v
	}
	if !other.valid() {
		return other
	}
	result := v.data.Matmul(other.data)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if needsGrad(v, other) {
		vData := v.data
		otherData := other.data
		vNdim := v.data.Ndim()
		otherNdim := other.data.Ndim()
		fn = &gradFn{
			name:   "MatmulBackward",
			inputs: []*Variable{v, other},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				var gradV, gradOther *tensor.Tensor
				switch {
				case vNdim >= 2 && otherNdim >= 2:
					// 2D+: grad_v = grad @ other^T, grad_other = v^T @ grad
					otherT := otherData.Transpose(otherNdim-2, otherNdim-1)
					gradV = grad.Matmul(otherT)
					vT := vData.Transpose(vNdim-2, vNdim-1)
					gradOther = vT.Matmul(grad)
				case vNdim == 1 && otherNdim == 2:
					// v [K] @ other [K,N] → result [N]
					otherT := otherData.Transpose(0, 1)
					gradV = grad.Matmul(otherT)
					vUnsqueeze := vData.Reshape([]int64{vData.Shape()[0], 1})
					gradUnsqueeze := grad.Reshape([]int64{1, grad.Shape()[0]})
					gradOther = vUnsqueeze.Matmul(gradUnsqueeze)
				case vNdim == 2 && otherNdim == 1:
					// v [M,K] @ other [K] → result [M]
					gradUnsqueeze := grad.Reshape([]int64{grad.Shape()[0], 1})
					otherUnsqueeze := otherData.Reshape([]int64{1, otherData.Shape()[0]})
					gradV = gradUnsqueeze.Matmul(otherUnsqueeze)
					vT := vData.Transpose(0, 1)
					gradOther = vT.Matmul(grad)
				default:
					// 1D @ 1D → scalar (dot product)
					gradV = otherData.MulScalar(1).Mul(grad)
					gradOther = vData.MulScalar(1).Mul(grad)
				}
				return []*tensor.Tensor{gradV, gradOther}
			},
		}
	}
	return newVar(result, fn)
}

// --- Unary operations ---

// ReLU applies max(0, x).
// Backward: grad_input = gradOutput * (input > 0).
func (v *Variable) ReLU() *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.ReLU()
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		inputData := v.data
		fn = &gradFn{
			name:   "ReLUBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				mask := inputData.GTScalar(0)
				return []*tensor.Tensor{grad.Mul(mask)}
			},
		}
	}
	return newVar(result, fn)
}

// Sigmoid applies 1 / (1 + exp(-x)).
// Backward: grad_input = gradOutput * output * (1 - output).
func (v *Variable) Sigmoid() *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Sigmoid()
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		output := result
		fn = &gradFn{
			name:   "SigmoidBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				// σ'(x) = σ(x) * (1 - σ(x))
				oneMinusOut := output.OnesLike().Sub(output)
				return []*tensor.Tensor{grad.Mul(output).Mul(oneMinusOut)}
			},
		}
	}
	return newVar(result, fn)
}

// Tanh applies the hyperbolic tangent.
// Backward: grad_input = gradOutput * (1 - output²).
func (v *Variable) Tanh() *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Tanh()
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		output := result
		fn = &gradFn{
			name:   "TanhBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				// tanh'(x) = 1 - tanh²(x)
				outSquared := output.Mul(output)
				oneMinusSquared := output.OnesLike().Sub(outSquared)
				return []*tensor.Tensor{grad.Mul(oneMinusSquared)}
			},
		}
	}
	return newVar(result, fn)
}

// Sum reduces all elements to a scalar.
// Backward: grad_input = gradOutput expanded to input shape (all ones * grad).
func (v *Variable) Sum() *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Sum()
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		inputData := v.data
		fn = &gradFn{
			name:   "SumBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				// Sum backward: gradient flows to every element equally.
				// grad is scalar; expand to input shape.
				ones := inputData.OnesLike()
				return []*tensor.Tensor{ones.Mul(grad)}
			},
		}
	}
	return newVar(result, fn)
}

// MulScalar multiplies every element by a scalar.
// Backward: grad_input = gradOutput * scalar.
func (v *Variable) MulScalar(scalar float64) *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.MulScalar(scalar)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		fn = &gradFn{
			name:   "MulScalarBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				return []*tensor.Tensor{grad.MulScalar(scalar)}
			},
		}
	}
	return newVar(result, fn)
}

// Exp returns element-wise exponential.
// Backward: grad_input = gradOutput * exp(input).
func (v *Variable) Exp() *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Exp()
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		output := result
		fn = &gradFn{
			name:   "ExpBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				return []*tensor.Tensor{grad.Mul(output)}
			},
		}
	}
	return newVar(result, fn)
}

// Log returns element-wise natural logarithm.
// Backward: grad_input = gradOutput / input.
func (v *Variable) Log() *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Log()
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		fn = &gradFn{
			name:   "LogBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				// d/dx ln(x) = 1/x
				// Compute 1/input as exp(-log(input)) = exp(-output)
				invInput := result.Neg().Exp()
				return []*tensor.Tensor{grad.Mul(invInput)}
			},
		}
	}
	return newVar(result, fn)
}

// Neg returns element-wise negation.
// Backward: grad_input = -gradOutput.
func (v *Variable) Neg() *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Neg()
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		fn = &gradFn{
			name:   "NegBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				return []*tensor.Tensor{grad.Neg()}
			},
		}
	}
	return newVar(result, fn)
}

// AddScalar adds a constant to every element.
// Backward: grad_input = gradOutput (scalar doesn't affect gradient).
func (v *Variable) AddScalar(scalar float64) *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.AddScalar(scalar)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		fn = &gradFn{
			name:   "AddScalarBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				return []*tensor.Tensor{grad}
			},
		}
	}
	return newVar(result, fn)
}

// SumDim reduces along a single dimension.
// Backward: gradient is expanded back to the input shape.
func (v *Variable) SumDim(dim int, keepdim bool) *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.SumDim(dim, keepdim)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		inputData := v.data
		fn = &gradFn{
			name:   "SumDimBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				g := grad
				if !keepdim {
					// Restore the reduced dimension for broadcasting.
					gradShape := g.Shape()
					newShape := make([]int64, len(gradShape)+1)
					copy(newShape[:dim], gradShape[:dim])
					newShape[dim] = 1
					copy(newShape[dim+1:], gradShape[dim:])
					g = g.Reshape(newShape)
				}
				ones := inputData.OnesLike()
				return []*tensor.Tensor{ones.Mul(g)}
			},
		}
	}
	return newVar(result, fn)
}

// Transpose swaps two dimensions.
// Backward: grad_input = transpose(gradOutput, dim0, dim1).
func (v *Variable) Transpose(dim0, dim1 int) *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Transpose(dim0, dim1)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		fn = &gradFn{
			name:   "TransposeBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				// Transpose is its own inverse with the same dims.
				return []*tensor.Tensor{grad.Transpose(dim0, dim1)}
			},
		}
	}
	return newVar(result, fn)
}

// Reshape returns a variable with the given shape.
// Backward: grad_input = reshape(gradOutput, original_shape).
func (v *Variable) Reshape(shape []int64) *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Reshape(shape)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		originalShape := v.data.Shape()
		fn = &gradFn{
			name:   "ReshapeBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				return []*tensor.Tensor{grad.Reshape(originalShape)}
			},
		}
	}
	return newVar(result, fn)
}

// Softmax applies softmax along a dimension.
// Backward: grad_input = output * (grad - sum(grad * output, dim, keepdim=true)).
func (v *Variable) Softmax(dim int) *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Softmax(dim)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		output := result
		fn = &gradFn{
			name:   "SoftmaxBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				gy := grad.Mul(output)
				sumGy := gy.SumDim(dim, true)
				return []*tensor.Tensor{output.Mul(grad.Sub(sumGy))}
			},
		}
	}
	return newVar(result, fn)
}

// Select picks a single index along a dimension, removing that dimension.
// Backward: gradient is embedded back into the original shape at the selected index.
func (v *Variable) Select(dim int, index int64) *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Select(dim, index)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		inputData := v.data
		fn = &gradFn{
			name:   "SelectBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				zeros := inputData.ZerosLike()
				return []*tensor.Tensor{zeros.SelectScatter(grad, dim, index)}
			},
		}
	}
	return newVar(result, fn)
}

// Narrow extracts a slice along dim: v[dim, start:start+length].
// Backward: gradient is placed back at the original position in a zeros tensor.
func (v *Variable) Narrow(dim int, start, length int64) *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Narrow(dim, start, length)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		inputData := v.data
		fn = &gradFn{
			name:   "NarrowBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				zeros := inputData.ZerosLike()
				return []*tensor.Tensor{zeros.NarrowScatter(grad, dim, start)}
			},
		}
	}
	return newVar(result, fn)
}

// Cat concatenates v and other along dim.
// Backward: gradient is split and routed to each input.
func (v *Variable) Cat(other *Variable, dim int) *Variable {
	if !v.valid() {
		return v
	}
	if !other.valid() {
		return other
	}
	result := v.data.Cat(other.data, dim)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if needsGrad(v, other) {
		vSize := v.data.Shape()[dim]
		otherSize := other.data.Shape()[dim]
		fn = &gradFn{
			name:   "CatBackward",
			inputs: []*Variable{v, other},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				gradV := grad.Narrow(dim, 0, vSize)
				gradOther := grad.Narrow(dim, vSize, otherSize)
				return []*tensor.Tensor{gradV, gradOther}
			},
		}
	}
	return newVar(result, fn)
}

// MeanDim computes the mean along a single dimension.
// Backward: grad_input = gradOutput / dim_size, expanded to input shape.
func (v *Variable) MeanDim(dim int, keepdim bool) *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.MeanDim(dim, keepdim)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		inputData := v.data
		fn = &gradFn{
			name:   "MeanDimBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				dimSize := float64(inputData.Shape()[dim])
				g := grad
				if !keepdim {
					// Restore the reduced dimension for broadcasting.
					gradShape := g.Shape()
					newShape := make([]int64, len(gradShape)+1)
					copy(newShape[:dim], gradShape[:dim])
					newShape[dim] = 1
					copy(newShape[dim+1:], gradShape[dim:])
					g = g.Reshape(newShape)
				}
				// Broadcast to input shape and divide by dim size.
				return []*tensor.Tensor{inputData.OnesLike().Mul(g).MulScalar(1.0 / dimSize)}
			},
		}
	}
	return newVar(result, fn)
}

// IndexSelect gathers slices along dim at the given indices.
// Only the source tensor gets gradients (indices are not differentiable).
// Backward: grad_input = zeros.index_add(dim, index, grad_output).
func (v *Variable) IndexSelect(dim int, index *tensor.Tensor) *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.IndexSelect(dim, index)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		inputData := v.data
		fn = &gradFn{
			name:   "IndexSelectBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				// Scatter gradients back to the positions they came from.
				zeros := inputData.ZerosLike()
				return []*tensor.Tensor{zeros.IndexAdd(dim, index, grad)}
			},
		}
	}
	return newVar(result, fn)
}

// Sqrt returns element-wise square root.
// Backward: grad_input = gradOutput / (2 * sqrt(input)).
func (v *Variable) Sqrt() *Variable {
	if !v.valid() {
		return v
	}
	result := v.data.Sqrt()
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if v.requiresGrad {
		output := result
		fn = &gradFn{
			name:   "SqrtBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				// d/dx sqrt(x) = 1 / (2*sqrt(x)) = 0.5 / sqrt(x)
				return []*tensor.Tensor{grad.Div(output.MulScalar(2))}
			},
		}
	}
	return newVar(result, fn)
}

// Div returns element-wise division v / other.
// Backward: grad_v = gradOutput / other, grad_other = -gradOutput * v / other².
func (v *Variable) Div(other *Variable) *Variable {
	if !v.valid() {
		return v
	}
	if !other.valid() {
		return other
	}
	result := v.data.Div(other.data)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}
	var fn *gradFn
	if needsGrad(v, other) {
		vShape := v.data.Shape()
		otherShape := other.data.Shape()
		vData := v.data
		otherData := other.data
		fn = &gradFn{
			name:   "DivBackward",
			inputs: []*Variable{v, other},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				// grad_v = grad / other
				gradV := unbroadcast(grad.Div(otherData), vShape)
				// grad_other = -grad * v / other²
				gradOther := unbroadcast(grad.Neg().Mul(vData).Div(otherData.Mul(otherData)), otherShape)
				return []*tensor.Tensor{gradV, gradOther}
			},
		}
	}
	return newVar(result, fn)
}

// --- Broadcast gradient reduction ---

// unbroadcast reduces a gradient tensor to match the original input shape.
// When broadcasting expands dimensions during forward, the backward pass
// must sum along those expanded dimensions to get the correct gradient shape.
func unbroadcast(grad *tensor.Tensor, targetShape []int64) *tensor.Tensor {
	if grad.Err() != nil {
		return grad
	}

	gradShape := grad.Shape()

	// Fast path: shapes already match.
	if shapesEqual(gradShape, targetShape) {
		return grad
	}

	result := grad

	// If gradient has more dimensions than target, sum leading dims.
	for len(gradShape) > len(targetShape) {
		result = result.SumDim(0, false)
		gradShape = result.Shape()
	}

	// Sum along dimensions where target has size 1 (broadcast dims).
	for i := 0; i < len(targetShape); i++ {
		if targetShape[i] == 1 && gradShape[i] != 1 {
			result = result.SumDim(i, true)
		}
	}

	return result
}

// --- Convolution ---

// Conv2d applies a 2D convolution. bias may be nil.
// Input shape: [N, C_in, H, W]. Weight shape: [C_out, C_in/groups, kH, kW].
func (v *Variable) Conv2d(weight, bias *Variable, stride, padding, dilation []int64, groups int64) *Variable {
	if !v.valid() {
		return v
	}
	if !weight.valid() {
		return weight
	}
	if bias != nil && !bias.valid() {
		return bias
	}

	// Forward.
	var biasT *tensor.Tensor
	if bias != nil {
		biasT = bias.data
	}
	result := v.data.Conv2d(weight.data, biasT, stride, padding, dilation, groups)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}

	// Backward.
	inputs := []*Variable{v, weight}
	hasBias := bias != nil
	if hasBias {
		inputs = append(inputs, bias)
	}

	var fn *gradFn
	if needsGrad(inputs...) {
		savedInput := v.data
		savedWeight := weight.data
		fn = &gradFn{
			name:   "Conv2dBackward",
			inputs: inputs,
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				gi, gw, gb := tensor.Conv2dBackward(
					grad, savedInput, savedWeight,
					stride, padding, dilation, groups, hasBias,
				)
				grads := []*tensor.Tensor{gi, gw}
				if hasBias {
					grads = append(grads, gb)
				}
				return grads
			},
		}
	}
	return newVar(result, fn)
}

// --- Expand ---

// Expand broadcasts a variable to a larger shape. Dimensions of size 1 are
// expanded to the requested size. The backward pass sums gradients along
// the expanded dimensions to restore the original shape.
func (v *Variable) Expand(shape []int64) *Variable {
	if !v.valid() {
		return v
	}

	result := v.data.Expand(shape)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}

	var fn *gradFn
	if needsGrad(v) {
		origShape := v.data.Shape()
		fn = &gradFn{
			name:   "ExpandBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				// Sum along dimensions that were expanded (size 1 -> size N).
				g := grad
				for i := range origShape {
					if origShape[i] == 1 && shape[i] != 1 {
						g = g.SumDim(i, true)
					}
				}
				return []*tensor.Tensor{g}
			},
		}
	}
	return newVar(result, fn)
}

// --- Transposed convolution ---

// ConvTranspose2d applies a 2D transposed convolution. bias may be nil.
func (v *Variable) ConvTranspose2d(weight, bias *Variable, stride, padding, outputPadding, dilation []int64, groups int64) *Variable {
	if !v.valid() {
		return v
	}
	if !weight.valid() {
		return weight
	}
	if bias != nil && !bias.valid() {
		return bias
	}

	var biasT *tensor.Tensor
	if bias != nil {
		biasT = bias.data
	}
	result := v.data.ConvTranspose2d(weight.data, biasT, stride, padding, outputPadding, dilation, groups)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}

	inputs := []*Variable{v, weight}
	hasBias := bias != nil
	if hasBias {
		inputs = append(inputs, bias)
	}

	var fn *gradFn
	if needsGrad(inputs...) {
		savedInput := v.data
		savedWeight := weight.data
		fn = &gradFn{
			name:   "ConvTranspose2dBackward",
			inputs: inputs,
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				gi, gw, gb := tensor.ConvTranspose2dBackward(
					grad, savedInput, savedWeight,
					stride, padding, outputPadding, dilation, groups, hasBias,
				)
				grads := []*tensor.Tensor{gi, gw}
				if hasBias {
					grads = append(grads, gb)
				}
				return grads
			},
		}
	}
	return newVar(result, fn)
}

// --- Adaptive average pooling ---

// AdaptiveAvgPool2d pools to the given output size.
func (v *Variable) AdaptiveAvgPool2d(outputSize []int64) *Variable {
	if !v.valid() {
		return v
	}

	result := v.data.AdaptiveAvgPool2d(outputSize)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}

	var fn *gradFn
	if needsGrad(v) {
		savedInput := v.data
		fn = &gradFn{
			name:   "AdaptiveAvgPool2dBackward",
			inputs: []*Variable{v},
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				return []*tensor.Tensor{tensor.AdaptiveAvgPool2dBackward(grad, savedInput)}
			},
		}
	}
	return newVar(result, fn)
}

// --- Grid sampling ---

// GridSample performs differentiable 2D grid sampling.
// Input shape: [N, C, H, W]. Grid shape: [N, H_out, W_out, 2].
// mode: 0=bilinear, 1=nearest, 2=bicubic.
// paddingMode: 0=zeros, 1=border, 2=reflection.
func (v *Variable) GridSample(grid *Variable, mode, paddingMode int, alignCorners bool) *Variable {
	if !v.valid() {
		return v
	}
	if !grid.valid() {
		return grid
	}

	result := v.data.GridSample(grid.data, mode, paddingMode, alignCorners)
	if err := result.Err(); err != nil {
		return errVariable(err)
	}

	inputs := []*Variable{v, grid}
	var fn *gradFn
	if needsGrad(inputs...) {
		savedInput := v.data
		savedGrid := grid.data
		fn = &gradFn{
			name:   "GridSampleBackward",
			inputs: inputs,
			apply: func(grad *tensor.Tensor) []*tensor.Tensor {
				gi, gg := tensor.GridSampleBackward(
					grad, savedInput, savedGrid,
					mode, paddingMode, alignCorners,
				)
				return []*tensor.Tensor{gi, gg}
			},
		}
	}
	return newVar(result, fn)
}

func shapesEqual(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
