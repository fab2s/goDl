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

// NamedInputModule is an optional interface for modules that receive
// Using references in a graph. When a module implements this interface,
// the graph runtime calls ForwardNamed instead of Forward, passing
// Using refs as a named map instead of positional arguments.
//
// On the first forward pass with forward refs (Using before Tag),
// refs that are not yet available are omitted from the map. Use the
// standard Go map lookup to check:
//
//	if state, ok := refs["memory"]; ok {
//	    // state is available
//	}
//
// The module must also implement Module (for Parameters and as a
// fallback when the module is used without Using refs).
//
//	type myRouter struct{}
//
//	func (r *myRouter) ForwardNamed(stream *autograd.Variable, refs map[string]*autograd.Variable) *autograd.Variable {
//	    ctx := refs["context"]
//	    // use stream and ctx to make a decision
//	}
type NamedInputModule interface {
	ForwardNamed(stream *autograd.Variable, refs map[string]*autograd.Variable) *autograd.Variable
}

// RefValidator is an optional interface that modules can implement to
// declare which Using refs they expect. The graph validates at Build
// time that exactly these refs are wired — catching typos, missing
// Using calls, and unexpected refs before any forward pass runs.
//
// RefValidator works independently of NamedInputModule: any module
// that receives Using refs (whether via Forward or ForwardNamed) can
// implement RefValidator to get build-time validation.
//
//	func (r *myRouter) RefNames() []string { return []string{"context"} }
type RefValidator interface {
	RefNames() []string
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
