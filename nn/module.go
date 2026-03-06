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

// TrainToggler is implemented by modules that behave differently during
// training vs inference (e.g., Dropout, BatchNorm). Modules that don't
// care about training mode simply don't implement this interface.
type TrainToggler interface {
	SetTraining(training bool)
}

// SetTraining sets training mode on a module if it supports it.
// For composite modules like Graph, this propagates to all children.
func SetTraining(m Module, training bool) {
	if t, ok := m.(TrainToggler); ok {
		t.SetTraining(training)
	}
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
// Draws from U(-bound, bound) where bound = sqrt(6 / fan_in).
func KaimingUniform(shape []int64, fanIn int64, opts ...tensor.Option) (*tensor.Tensor, error) {
	bound := math.Sqrt(6.0 / float64(fanIn))
	return uniformInit(shape, bound, opts...)
}

// KaimingNormal initializes a tensor with Kaiming (He) normal distribution.
// Suitable for layers followed by ReLU. fan_in is the number of input features.
// Draws from N(0, std) where std = sqrt(2 / fan_in).
func KaimingNormal(shape []int64, fanIn int64, opts ...tensor.Option) (*tensor.Tensor, error) {
	std := math.Sqrt(2.0 / float64(fanIn))
	return normalInit(shape, std, opts...)
}

// XavierUniform initializes a tensor with Xavier (Glorot) uniform distribution.
// Suitable for layers followed by sigmoid or tanh.
// Draws from U(-bound, bound) where bound = sqrt(6 / (fan_in + fan_out)).
func XavierUniform(shape []int64, fanIn, fanOut int64, opts ...tensor.Option) (*tensor.Tensor, error) {
	bound := math.Sqrt(6.0 / float64(fanIn+fanOut))
	return uniformInit(shape, bound, opts...)
}

// XavierNormal initializes a tensor with Xavier (Glorot) normal distribution.
// Suitable for layers followed by sigmoid or tanh.
// Draws from N(0, std) where std = sqrt(2 / (fan_in + fan_out)).
func XavierNormal(shape []int64, fanIn, fanOut int64, opts ...tensor.Option) (*tensor.Tensor, error) {
	std := math.Sqrt(2.0 / float64(fanIn+fanOut))
	return normalInit(shape, std, opts...)
}

// uniformInit generates a tensor with values from U(-bound, bound).
func uniformInit(shape []int64, bound float64, opts ...tensor.Option) (*tensor.Tensor, error) {
	t, err := tensor.Rand(shape, opts...)
	if err != nil {
		return nil, err
	}
	result := t.MulScalar(2 * bound).AddScalar(-bound)
	if err := result.Err(); err != nil {
		t.Release()
		return nil, err
	}
	t.Release()
	return result, nil
}

// normalInit generates a tensor with values from N(0, std).
func normalInit(shape []int64, std float64, opts ...tensor.Option) (*tensor.Tensor, error) {
	t, err := tensor.RandN(shape, opts...)
	if err != nil {
		return nil, err
	}
	result := t.MulScalar(std)
	if err := result.Err(); err != nil {
		t.Release()
		return nil, err
	}
	t.Release()
	return result, nil
}
