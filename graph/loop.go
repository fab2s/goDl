package graph

import (
	"fmt"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// LoopBuilder configures a loop construct in the graph flow.
// A loop repeats a body module N times, carrying state between iterations.
//
// The body receives the current state as input and returns the new state.
// After all iterations, the final state continues downstream.
//
// Backward passes unroll automatically via autograd — each iteration
// builds its own computation graph, and the backward walk reverses
// through all of them (backpropagation through time).
type LoopBuilder struct {
	fb   *FlowBuilder
	body nn.Module
}

// For sets the loop iteration count and wires the loop into the graph.
// Returns the FlowBuilder for continued chaining.
//
//	graph.From(encoder).
//	    Loop(refinementStep).For(5).
//	    Through(decoder).
//	    Build()
func (lb *LoopBuilder) For(n int) *FlowBuilder {
	fb := lb.fb
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Loop requires single stream (got %d)", len(fb.current))
		return fb
	}
	if n < 1 {
		fb.err = fmt.Errorf("graph: Loop requires at least 1 iteration (got %d)", n)
		return fb
	}

	cur := fb.current[0]

	name := fmt.Sprintf("loop_%d", fb.counter)
	fb.counter++

	run, params := makeLoopFunc(lb.body, n)
	fb.nodes[name] = &Node{
		id:          name,
		inputPorts:  []string{DefaultInput},
		outputPorts: []string{DefaultOutput},
		run:         run,
		params:      params,
	}

	fb.edges = append(fb.edges, &Edge{
		fromNode: cur.id,
		fromPort: cur.outputPort,
		toNode:   name,
		toPort:   DefaultInput,
	})

	fb.current = []*nodeRef{{id: name, outputPort: DefaultOutput}}
	return fb
}

// makeLoopFunc creates a nodeFunc that executes a body module N times,
// feeding each iteration's output as the next iteration's input.
//
// Autograd tracks each iteration's operations, so backward naturally
// unrolls N steps in reverse (BPTT) without any special handling.
func makeLoopFunc(body nn.Module, count int) (nodeFunc, func() []*nn.Parameter) {
	run := func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		state := inputs[0]
		for i := range count {
			state = body.Forward(state)
			if err := state.Err(); err != nil {
				return nil, fmt.Errorf("loop iteration %d: %w", i, err)
			}
		}
		return []*autograd.Variable{state}, nil
	}
	return run, body.Parameters
}
