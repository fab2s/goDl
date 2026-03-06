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
