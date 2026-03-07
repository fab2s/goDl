package graph

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// StateAdd returns an additive state cell for use with forward
// references (Using before Tag).
//
// On the first Forward call, the state is auto-zeroed by the graph,
// so stream + zeros = stream (pass-through). On subsequent calls,
// the accumulated state is added to the current stream.
//
//	graph.From(embed).
//	    Through(graph.StateAdd()).Using("memory").
//	    Tag("memory").
//	    Build()
func StateAdd() nn.Module {
	return &stateAdd{}
}

type stateAdd struct{}

func (a *stateAdd) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	var result *autograd.Variable
	for _, inp := range inputs {
		if inp == nil {
			continue
		}
		if result == nil {
			result = inp
		} else {
			result = result.Add(inp)
		}
	}
	return result
}

func (a *stateAdd) Parameters() []*nn.Parameter { return nil }
