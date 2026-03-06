// Package nn provides neural network layers, loss functions, and optimizers.
//
// All layers implement the Module interface. Modules compose naturally —
// a model is a Module that contains other Modules.
//
//	linear := nn.NewLinear(784, 128)
//	output := linear.Forward(input)
//	loss := nn.MSELoss(output, target)
//	loss.Backward()
package nn

import (
	"math"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// Module is anything with learnable parameters that transforms variables.
type Module interface {
	// Forward computes the module's output.
	Forward(inputs ...*autograd.Variable) *autograd.Variable

	// Parameters returns all learnable parameters.
	Parameters() []*Parameter
}

// Parameter is a variable that requires gradients — a learnable weight.
type Parameter struct {
	*autograd.Variable
	Name string
}

// NewParameter creates a parameter from a tensor with gradient tracking.
func NewParameter(data *tensor.Tensor, name string) *Parameter {
	return &Parameter{
		Variable: autograd.NewVariable(data, true),
		Name:     name,
	}
}

// --- Weight initialization ---

// KaimingUniform initializes a tensor with Kaiming (He) uniform distribution.
// Suitable for layers followed by ReLU. fan_in is the number of input features.
func KaimingUniform(shape []int64, fanIn int64, opts ...tensor.Option) (*tensor.Tensor, error) {
	// Kaiming uniform: U(-bound, bound) where bound = sqrt(6 / fan_in)
	bound := math.Sqrt(6.0 / float64(fanIn))
	// Generate uniform [0, 1) then scale to [-bound, bound)
	t, err := tensor.Rand(shape, opts...)
	if err != nil {
		return nil, err
	}
	// t * 2*bound - bound
	result := t.MulScalar(2 * bound).AddScalar(-bound)
	if err := result.Err(); err != nil {
		t.Release()
		return nil, err
	}
	t.Release()
	return result, nil
}
