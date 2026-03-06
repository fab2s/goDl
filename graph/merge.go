package graph

import (
	"fmt"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// Add returns a merge module that element-wise adds all inputs.
// Gradients flow to every input equally (each gets the full upstream gradient).
//
// Used for residual connections and skip connections:
//
//	graph.From(encoder).
//	    Split(branchA, branchB).
//	    Merge(graph.Add()).
//	    Build()
func Add() nn.Module {
	return &addMerge{}
}

type addMerge struct{}

func (a *addMerge) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	if len(inputs) == 0 {
		return autograd.ErrVariable(fmt.Errorf("graph: Add merge received no inputs"))
	}
	result := inputs[0]
	for i := 1; i < len(inputs); i++ {
		result = result.Add(inputs[i])
	}
	return result
}

func (a *addMerge) Parameters() []*nn.Parameter { return nil }

// Mean returns a merge module that averages all inputs element-wise.
//
//	graph.From(encoder).
//	    Split(branchA, branchB).
//	    Merge(graph.Mean()).
//	    Build()
func Mean() nn.Module {
	return &meanMerge{}
}

type meanMerge struct{}

func (m *meanMerge) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	if len(inputs) == 0 {
		return autograd.ErrVariable(fmt.Errorf("graph: Mean merge received no inputs"))
	}
	result := inputs[0]
	for i := 1; i < len(inputs); i++ {
		result = result.Add(inputs[i])
	}
	return result.MulScalar(1.0 / float64(len(inputs)))
}

func (m *meanMerge) Parameters() []*nn.Parameter { return nil }

// Cat returns a merge module that concatenates all inputs along the
// given dimension.
//
//	graph.From(encoder).
//	    Split(branchA, branchB).
//	    Merge(graph.Cat(1)).
//	    Through(nn.MustLinear(combined, hidden)).
//	    Build()
func Cat(dim int) nn.Module {
	return &catMerge{dim: dim}
}

type catMerge struct {
	dim int
}

func (c *catMerge) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	if len(inputs) == 0 {
		return autograd.ErrVariable(fmt.Errorf("graph: Cat merge received no inputs"))
	}
	result := inputs[0]
	for i := 1; i < len(inputs); i++ {
		result = result.Cat(inputs[i], c.dim)
	}
	return result
}

func (c *catMerge) Parameters() []*nn.Parameter { return nil }
