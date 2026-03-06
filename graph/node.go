package graph

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// nodeFunc is the internal execution contract for a graph node.
// It receives ordered inputs (one per input port) and returns
// ordered outputs (one per output port).
type nodeFunc func(inputs []*autograd.Variable) ([]*autograd.Variable, error)

// Node is a computation unit in a graph with named input/output ports.
type Node struct {
	id          string
	inputPorts  []string
	outputPorts []string
	run         nodeFunc
	params      func() []*nn.Parameter
	module      nn.Module // nil for internal nodes (add, gated_merge, state_read)
}

// wrapModule adapts an nn.Module to the graph engine's internal contract.
// The module's variadic inputs map to the node's ordered input ports,
// and its single output maps to the sole output port.
func wrapModule(m nn.Module) (nodeFunc, func() []*nn.Parameter) {
	run := func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		result := m.Forward(inputs...)
		if err := result.Err(); err != nil {
			return nil, err
		}
		return []*autograd.Variable{result}, nil
	}
	return run, m.Parameters
}
