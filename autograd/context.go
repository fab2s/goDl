package autograd

import "sync/atomic"

// noGradDepth tracks nested NoGrad contexts. When > 0, all new
// variables are created without gradient tracking regardless of
// their requiresGrad setting.
var noGradDepth atomic.Int64

// IsGradEnabled returns true if gradient tracking is currently active.
func IsGradEnabled() bool {
	return noGradDepth.Load() == 0
}

// NoGrad executes fn with gradient tracking disabled. Variables
// created inside fn will not build a computation graph, even if
// their inputs require gradients. This is used for inference.
//
//	autograd.NoGrad(func() {
//	    output := model.Forward(input)
//	    // No graph is built, no memory overhead from tracking.
//	})
func NoGrad(fn func()) {
	noGradDepth.Add(1)
	defer noGradDepth.Add(-1)
	fn()
}
