package nn

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// Dropout randomly zeroes elements during training with probability p.
// Uses inverted dropout: surviving elements are scaled by 1/(1-p) so that
// the expected value is preserved and no scaling is needed at inference.
//
//	drop := nn.NewDropout(0.1)
//	output := drop.Forward(input)
type Dropout struct {
	p        float64
	training bool
}

// NewDropout creates a Dropout module with the given drop probability.
func NewDropout(p float64) *Dropout {
	return &Dropout{p: p, training: true}
}

// SetTraining enables or disables dropout.
func (d *Dropout) SetTraining(training bool) {
	d.training = training
}

// Forward applies dropout to the input.
// During training: randomly zeroes elements and scales survivors.
// During inference: identity function.
func (d *Dropout) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	x := inputs[0]
	if !d.training || d.p == 0 {
		return x
	}
	if d.p >= 1 {
		// Drop everything
		zt, err := tensor.Zeros(x.Data().Shape())
		if err != nil {
			return autograd.ErrVariable(err)
		}
		return autograd.NewVariable(zt, false)
	}

	// Generate random mask: 1 where rand > p, 0 elsewhere
	maskT, err := tensor.Rand(x.Data().Shape())
	if err != nil {
		return autograd.ErrVariable(err)
	}
	mask := maskT.GTScalar(d.p)
	maskVar := autograd.NewVariable(mask, false)

	// Inverted dropout: scale by 1/(1-p)
	scale := 1.0 / (1.0 - d.p)
	return x.Mul(maskVar).MulScalar(scale)
}

// Parameters returns nil — Dropout has no learnable parameters.
func (d *Dropout) Parameters() []*Parameter { return nil }
