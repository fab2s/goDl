package nn

import "github.com/fab2s/goDl/autograd"

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
