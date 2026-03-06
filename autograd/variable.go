// Package autograd provides reverse-mode automatic differentiation.
//
// Variables wrap tensors with gradient tracking. When requiresGrad is
// true, operations build a computation graph. Calling Backward() walks
// the graph in reverse, accumulating gradients at each leaf variable.
//
// Operations are chainable and carry errors, mirroring the tensor API:
//
//	x := autograd.NewVariable(tensorX, true)
//	w := autograd.NewVariable(tensorW, true)
//	loss := x.Matmul(w).Sum()
//	loss.Backward()
//	fmt.Println(w.Grad()) // gradient of loss w.r.t. w
package autograd

import (
	"fmt"

	"github.com/fab2s/goDl/tensor"
)

// Variable is a tensor with gradient tracking.
//
// Leaf variables (created by the user, not by an operation) accumulate
// gradients during backward. Non-leaf variables are intermediate
// computation results whose gradients are transient unless RetainGrad
// is called.
type Variable struct {
	data         *tensor.Tensor
	grad         *tensor.Tensor // accumulated gradient, nil until backward
	requiresGrad bool
	gradFn       *gradFn // nil for leaf variables
	isLeaf       bool
	retainGrad   bool
	err          error
}

// gradFn records how a variable was created, linking it to its inputs.
// The backward pass walks these links in reverse topological order.
type gradFn struct {
	name   string      // e.g. "AddBackward", for debugging
	inputs []*Variable // back-edges to input variables
	apply  func(gradOutput *tensor.Tensor) []*tensor.Tensor
}

// NewVariable creates a leaf variable from an existing tensor.
// If requiresGrad is true, operations on this variable will be tracked
// and gradients will be accumulated during backward.
func NewVariable(data *tensor.Tensor, requiresGrad bool) *Variable {
	if err := data.Err(); err != nil {
		return &Variable{err: err}
	}
	return &Variable{
		data:         data,
		requiresGrad: requiresGrad,
		isLeaf:       true,
	}
}

// newVar creates a non-leaf variable as the result of an operation.
// If none of the inputs require gradients, tracking is skipped.
func newVar(data *tensor.Tensor, fn *gradFn) *Variable {
	return &Variable{
		data:         data,
		requiresGrad: fn != nil,
		gradFn:       fn,
		isLeaf:       false,
	}
}

// errVariable creates a variable that carries an error.
func errVariable(err error) *Variable {
	return &Variable{err: err}
}

// Data returns the underlying tensor.
func (v *Variable) Data() *tensor.Tensor {
	return v.data
}

// Grad returns the accumulated gradient tensor, or nil if no gradient
// has been computed yet.
func (v *Variable) Grad() *tensor.Tensor {
	return v.grad
}

// RequiresGrad returns whether this variable tracks gradients.
func (v *Variable) RequiresGrad() bool {
	return v.requiresGrad
}

// IsLeaf returns true for user-created variables (not operation results).
func (v *Variable) IsLeaf() bool {
	return v.isLeaf
}

// RetainGrad marks a non-leaf variable to keep its gradient after backward.
// By default, only leaf variables retain gradients.
func (v *Variable) RetainGrad() *Variable {
	v.retainGrad = true
	return v
}

// Err returns the error carried by this variable, or nil if valid.
func (v *Variable) Err() error {
	if v.err != nil {
		return v.err
	}
	if v.data == nil {
		return fmt.Errorf("autograd: nil variable")
	}
	return v.data.Err()
}

// valid returns true if the variable can be used in operations.
func (v *Variable) valid() bool {
	return v.Err() == nil
}

// needsGrad returns true if any of the given variables requires gradients
// AND gradient tracking is currently enabled (not inside a NoGrad block).
func needsGrad(vars ...*Variable) bool {
	if !IsGradEnabled() {
		return false
	}
	for _, v := range vars {
		if v.requiresGrad {
			return true
		}
	}
	return false
}

// ZeroGrad resets the accumulated gradient to nil.
func (v *Variable) ZeroGrad() {
	if v.grad != nil {
		v.grad.Release()
		v.grad = nil
	}
}

// Detach returns a new leaf variable sharing the same tensor data
// but with no gradient tracking. Useful for stopping gradient flow.
func (v *Variable) Detach() *Variable {
	if !v.valid() {
		return v
	}
	return NewVariable(v.data, false)
}
