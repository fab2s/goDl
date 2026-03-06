package graph

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// ThresholdHalt returns a halt condition for Loop.Until that signals
// stop when the maximum element of the state exceeds the threshold.
//
//	Loop(body).Until(graph.ThresholdHalt(50), 20)
func ThresholdHalt(threshold float32) nn.Module {
	return &threshHalt{threshold: threshold}
}

type threshHalt struct {
	threshold float32
}

func (h *threshHalt) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	data, _ := inputs[0].Data().Float32Data()
	maxVal := data[0]
	for _, v := range data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	val := maxVal - h.threshold // positive when exceeded → halt
	t, _ := tensor.FromFloat32([]float32{val}, []int64{1})
	return autograd.NewVariable(t, false)
}

func (h *threshHalt) Parameters() []*nn.Parameter { return nil }

// LearnedHalt returns a halt condition for Loop.Until with a learnable
// linear probe. The probe projects the state to a scalar — iteration
// stops when the output is positive.
//
// This is the Adaptive Computation Time (ACT) pattern: the network
// learns when to stop iterating.
//
//	Loop(body).Until(graph.LearnedHalt(hidden), 20)
func LearnedHalt(inputDim int64) nn.Module {
	return &learnedHaltModule{
		proj: nn.MustLinear(inputDim, 1),
	}
}

type learnedHaltModule struct {
	proj *nn.Linear
}

func (h *learnedHaltModule) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return h.proj.Forward(inputs[0])
}

func (h *learnedHaltModule) Parameters() []*nn.Parameter {
	return h.proj.Parameters()
}
