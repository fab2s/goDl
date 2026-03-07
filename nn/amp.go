package nn

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// GradScaler implements dynamic loss scaling for mixed precision training.
// It scales the loss before backward to prevent float16 gradient underflow,
// then unscales gradients before the optimizer step.
//
//	scaler := nn.NewGradScaler()
//	for loader.Next() {
//	    pred := model.Forward(autograd.NewVariable(input.Half(), true))
//	    loss := nn.MSELoss(pred, autograd.NewVariable(target.Half(), false))
//
//	    optimizer.ZeroGrad()
//	    scaler.Scale(loss).Backward()
//	    scaler.Step(optimizer)
//	    scaler.Update()
//	}
type GradScaler struct {
	scale       float64
	growth      float64 // factor to grow scale (default 2.0)
	backoff     float64 // factor to shrink scale (default 0.5)
	interval    int     // steps between growth attempts (default 2000)
	stepsSinceG int     // steps since last growth
	foundInf    bool    // set by Step if any grad has inf/nan
}

// NewGradScaler creates a gradient scaler with default settings.
// Initial scale is 2^16 = 65536, a common starting point.
func NewGradScaler() *GradScaler {
	return &GradScaler{
		scale:    65536.0,
		growth:   2.0,
		backoff:  0.5,
		interval: 2000,
	}
}

// Scale multiplies the loss by the current scale factor and returns
// the result for calling Backward on.
func (s *GradScaler) Scale(loss *autograd.Variable) *autograd.Variable {
	return loss.MulScalar(s.scale)
}

// Step unscales gradients on the optimizer's parameters, checks for
// inf/nan, and either performs the optimizer step or skips it.
// Returns true if the step was performed (gradients were finite).
func (s *GradScaler) Step(opt LRAdjustable) bool {
	params := optParams(opt)
	invScale := 1.0 / s.scale

	// Unscale and check for infs/nans.
	s.foundInf = false
	for _, p := range params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		unscaled := grad.MulScalar(invScale)
		if !unscaled.AllFinite() {
			s.foundInf = true
			break
		}
	}

	if s.foundInf {
		// Skip optimizer step, discard gradients.
		return false
	}

	// Apply unscaled gradients and step.
	for _, p := range params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		p.SetGrad(grad.MulScalar(invScale))
	}
	opt.Step()
	return true
}

// Update adjusts the scale factor based on whether inf/nan was found.
// Call this after every Step.
func (s *GradScaler) Update() {
	if s.foundInf {
		s.scale *= s.backoff
		s.stepsSinceG = 0
		return
	}
	s.stepsSinceG++
	if s.stepsSinceG >= s.interval {
		s.scale *= s.growth
		s.stepsSinceG = 0
	}
}

// ScaleFactor returns the current scale factor.
func (s *GradScaler) ScaleFactor() float64 {
	return s.scale
}

// optParams extracts parameters from an optimizer.
func optParams(opt LRAdjustable) []*Parameter {
	switch o := opt.(type) {
	case *SGD:
		return o.params
	case *Adam:
		return o.params
	case *AdamW:
		return o.adam.params
	default:
		return nil
	}
}

// CastParameters casts all parameter data tensors to the given dtype.
// Useful for converting a model to float16 for mixed precision training.
//
//	nn.CastParameters(model.Parameters(), tensor.Float16)
func CastParameters(params []*Parameter, dtype tensor.DType) {
	for _, p := range params {
		if p.Data().DType() != dtype {
			p.SetData(p.Data().ToDType(dtype))
		}
	}
}
