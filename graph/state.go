package graph

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// StateAdd returns a nil-safe additive state cell for use with forward
// references (Using before Tag).
//
// On the first Forward call, the state input is nil — only the stream
// passes through. On subsequent calls, the previous state is added to
// the current stream, creating an accumulator.
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
