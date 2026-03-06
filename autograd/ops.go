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
