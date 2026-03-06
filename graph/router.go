package graph

import (
	"fmt"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// SoftmaxRouter returns a Gate router that produces softmax-normalized
// weights over numExperts experts.
//
// When the router receives multiple inputs (via Gate.Using), they are
// summed element-wise before projection — this lets the router see
// both the stream and any tagged references without changing dimensions.
//
//	graph.From(embed).
//	    Tag("ctx").
//	    Through(layer).
//	    Gate(graph.SoftmaxRouter(hidden, 3), expertA, expertB, expertC).Using("ctx").
//	    Build()
func SoftmaxRouter(inputDim int64, numExperts int) nn.Module {
	return &softmaxRouter{
		proj: nn.MustLinear(inputDim, int64(numExperts)),
	}
}

type softmaxRouter struct {
	proj *nn.Linear
}

func (r *softmaxRouter) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	combined := inputs[0]
	for i := 1; i < len(inputs); i++ {
		if inputs[i] != nil {
			combined = combined.Add(inputs[i])
		}
	}
	out := r.proj.Forward(combined)
	dim := out.Data().Ndim() - 1
	return out.Softmax(dim)
}

func (r *softmaxRouter) Parameters() []*nn.Parameter {
	return r.proj.Parameters()
}

// SigmoidRouter returns a Gate router that produces independent sigmoid
// weights over numExperts experts. Unlike SoftmaxRouter, weights do not
// sum to 1 — each expert is gated independently between 0 and 1.
//
//	Gate(graph.SigmoidRouter(hidden, 2), expertA, expertB)
func SigmoidRouter(inputDim int64, numExperts int) nn.Module {
	return &sigmoidRouter{
		proj: nn.MustLinear(inputDim, int64(numExperts)),
	}
}

type sigmoidRouter struct {
	proj *nn.Linear
}

func (r *sigmoidRouter) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	combined := inputs[0]
	for i := 1; i < len(inputs); i++ {
		if inputs[i] != nil {
			combined = combined.Add(inputs[i])
		}
	}
	return r.proj.Forward(combined).Sigmoid()
}

func (r *sigmoidRouter) Parameters() []*nn.Parameter {
	return r.proj.Parameters()
}

// FixedSelector returns a Switch router that always selects the same
// branch. Useful for testing, ablation studies, or static configurations.
//
//	Switch(graph.FixedSelector(0), branchA, branchB)
func FixedSelector(index int) nn.Module {
	return &fixedSel{index: float32(index)}
}

type fixedSel struct {
	index float32
}

func (r *fixedSel) Forward(_ ...*autograd.Variable) *autograd.Variable {
	t, _ := tensor.FromFloat32([]float32{r.index}, []int64{1})
	return autograd.NewVariable(t, false)
}

func (r *fixedSel) Parameters() []*nn.Parameter { return nil }

// ArgmaxSelector returns a Switch router with a learnable linear
// projection. It picks the branch whose logit is highest (argmax).
//
// Selection is non-differentiable — gradients flow through the selected
// branch only. The projection parameters are included in the graph's
// Parameters() for training with policy-gradient methods if desired.
//
//	Switch(graph.ArgmaxSelector(hidden, 3), branchA, branchB, branchC)
func ArgmaxSelector(inputDim int64, numBranches int) nn.Module {
	return &argmaxSel{
		proj: nn.MustLinear(inputDim, int64(numBranches)),
	}
}

type argmaxSel struct {
	proj *nn.Linear
}

func (r *argmaxSel) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	combined := inputs[0]
	for i := 1; i < len(inputs); i++ {
		if inputs[i] != nil {
			combined = combined.Add(inputs[i])
		}
	}
	logits := r.proj.Forward(combined)
	data, err := logits.Data().Float32Data()
	if err != nil {
		return autograd.ErrVariable(fmt.Errorf("graph: ArgmaxSelector: %w", err))
	}
	best := 0
	for i := 1; i < len(data); i++ {
		if data[i] > data[best] {
			best = i
		}
	}
	t, _ := tensor.FromFloat32([]float32{float32(best)}, []int64{1})
	return autograd.NewVariable(t, false)
}

func (r *argmaxSel) Parameters() []*nn.Parameter {
	return r.proj.Parameters()
}
