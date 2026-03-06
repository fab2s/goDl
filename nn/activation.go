package nn

import (
	"math"

	"github.com/fab2s/goDl/autograd"
)

// ReLU is a module that applies max(0, x) element-wise.
type ReLU struct{}

// NewReLU creates a ReLU activation module.
func NewReLU() *ReLU { return &ReLU{} }

// Forward applies ReLU to the first input.
func (r *ReLU) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return inputs[0].ReLU()
}

// Parameters returns nil — ReLU has no learnable parameters.
func (r *ReLU) Parameters() []*Parameter { return nil }

// Sigmoid is a module that applies the logistic sigmoid element-wise.
type Sigmoid struct{}

// NewSigmoid creates a Sigmoid activation module.
func NewSigmoid() *Sigmoid { return &Sigmoid{} }

// Forward applies Sigmoid to the first input.
func (s *Sigmoid) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return inputs[0].Sigmoid()
}

// Parameters returns nil — Sigmoid has no learnable parameters.
func (s *Sigmoid) Parameters() []*Parameter { return nil }

// Tanh is a module that applies hyperbolic tangent element-wise.
type Tanh struct{}

// NewTanh creates a Tanh activation module.
func NewTanh() *Tanh { return &Tanh{} }

// Forward applies Tanh to the first input.
func (t *Tanh) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return inputs[0].Tanh()
}

// Parameters returns nil — Tanh has no learnable parameters.
func (t *Tanh) Parameters() []*Parameter { return nil }

// GELU applies the Gaussian Error Linear Unit activation.
// Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Backward is computed automatically through the composed autograd ops.
type GELU struct{}

// NewGELU creates a GELU activation module.
func NewGELU() *GELU { return &GELU{} }

// Forward applies GELU to the first input.
func (g *GELU) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	x := inputs[0]
	// sqrt(2/pi) ≈ 0.7978845608
	coeff := math.Sqrt(2.0 / math.Pi)
	// inner = sqrt(2/pi) * (x + 0.044715 * x^3)
	x3 := x.Mul(x).Mul(x)
	inner := x.Add(x3.MulScalar(0.044715)).MulScalar(coeff)
	// 0.5 * x * (1 + tanh(inner))
	return x.MulScalar(0.5).Mul(inner.Tanh().AddScalar(1))
}

// Parameters returns nil — GELU has no learnable parameters.
func (g *GELU) Parameters() []*Parameter { return nil }

// SiLU applies the Sigmoid Linear Unit (Swish) activation: x * sigmoid(x).
// Backward is computed automatically through the composed autograd ops.
type SiLU struct{}

// NewSiLU creates a SiLU activation module.
func NewSiLU() *SiLU { return &SiLU{} }

// Forward applies SiLU to the first input.
func (s *SiLU) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	x := inputs[0]
	return x.Mul(x.Sigmoid())
}

// Parameters returns nil — SiLU has no learnable parameters.
func (s *SiLU) Parameters() []*Parameter { return nil }

// SoftmaxModule applies softmax along a given dimension.
type SoftmaxModule struct {
	Dim int
}

// NewSoftmax creates a Softmax activation module for the given dimension.
func NewSoftmax(dim int) *SoftmaxModule { return &SoftmaxModule{Dim: dim} }

// Forward applies Softmax to the first input.
func (s *SoftmaxModule) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return inputs[0].Softmax(s.Dim)
}

// Parameters returns nil — Softmax has no learnable parameters.
func (s *SoftmaxModule) Parameters() []*Parameter { return nil }
