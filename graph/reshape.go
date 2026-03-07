package graph

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// Reshape returns a zero-parameter module that reshapes its input
// to the given shape. This is a graph-level primitive useful for
// adapting tensor dimensions between modules.
//
//	graph.From(encoder).
//	    Through(graph.Reshape(4, 2)).     // [1, 8] → [4, 2]
//	    Map(readHead(2)).Each().
//	    Through(graph.Reshape(1, 8)).     // [4, 2] → [1, 8]
//	    Build()
func Reshape(shape ...int64) nn.Module {
	s := make([]int64, len(shape))
	copy(s, shape)
	return &reshapeModule{shape: s}
}

type reshapeModule struct {
	shape []int64
}

func (r *reshapeModule) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return inputs[0].Reshape(r.shape)
}

func (r *reshapeModule) Parameters() []*nn.Parameter { return nil }
