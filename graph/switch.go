package graph

import (
	"fmt"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// Switch creates a hard-routing conditional construct. A router module
// selects which branch to execute based on its output.
//
// The router must return a scalar (1-element tensor) containing the
// 0-based branch index. The router owns the selection logic — argmax,
// sampling, round-robin, or any other strategy. Only the selected
// branch executes; unselected branches are skipped entirely.
//
// The selection is non-differentiable — gradients flow through the
// selected branch only. The router does not receive gradient through
// the selection. For differentiable routing, use Gate instead.
//
// Use Using after Switch to wire additional tagged references to the router:
//
//	graph.From(encoder).
//	    Through(layer1).Tag("features").
//	    Through(layer2).
//	    Switch(router, branchA, branchB, branchC).Using("features").
//	    Through(decoder).
//	    Build()
func (fb *FlowBuilder) Switch(router nn.Module, branches ...nn.Module) *FlowBuilder {
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Switch requires single stream (got %d)", len(fb.current))
		return fb
	}
	if len(branches) < 2 {
		fb.err = fmt.Errorf("graph: Switch requires at least 2 branches (got %d)", len(branches))
		return fb
	}

	cur := fb.current[0]
	composite := &switchComposite{router: router, branches: branches}

	name := fmt.Sprintf("switch_%d", fb.counter)
	fb.counter++

	fb.nodes[name] = &Node{
		id:          name,
		inputPorts:  []string{DefaultInput},
		outputPorts: []string{DefaultOutput},
		run:         makeSwitchFunc(router, branches),
		params:      composite.Parameters,
		module:      composite,
	}

	fb.edges = append(fb.edges, &Edge{
		fromNode: cur.id,
		fromPort: cur.outputPort,
		toNode:   name,
		toPort:   DefaultInput,
	})

	ref := &nodeRef{id: name, outputPort: DefaultOutput}
	fb.current = []*nodeRef{ref}
	fb.onTarget = ref
	return fb
}

// makeSwitchFunc creates a nodeFunc that runs the router to select a
// branch index, then executes only that branch.
func makeSwitchFunc(router nn.Module, branches []nn.Module) nodeFunc {
	return func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		// Run router with all inputs (stream + Using refs).
		logits := router.Forward(inputs...)
		if err := logits.Err(); err != nil {
			return nil, fmt.Errorf("switch router: %w", err)
		}

		// Read branch index from first element.
		data, err := logits.Data().Float32Data()
		if err != nil {
			return nil, fmt.Errorf("switch router data: %w", err)
		}
		if len(data) == 0 {
			return nil, fmt.Errorf("switch: router returned empty tensor")
		}

		selected := int(data[0])
		if selected < 0 || selected >= len(branches) {
			return nil, fmt.Errorf("switch: router selected index %d, valid range [0, %d)", selected, len(branches))
		}

		// Execute only the selected branch (stream input only).
		result := branches[selected].Forward(inputs[0])
		if err := result.Err(); err != nil {
			return nil, fmt.Errorf("switch branch %d: %w", selected, err)
		}

		return []*autograd.Variable{result}, nil
	}
}

// switchComposite bundles router + branches for SetTraining/Parameters.
type switchComposite struct {
	router   nn.Module
	branches []nn.Module
}

func (sc *switchComposite) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return sc.router.Forward(inputs...)
}

func (sc *switchComposite) Parameters() []*nn.Parameter {
	all := sc.router.Parameters()
	for _, b := range sc.branches {
		all = append(all, b.Parameters()...)
	}
	return all
}

func (sc *switchComposite) SetTraining(training bool) {
	nn.SetTraining(sc.router, training)
	for _, b := range sc.branches {
		nn.SetTraining(b, training)
	}
}
