package graph

import (
	"fmt"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// Gate creates a gated routing construct. A router module produces
// weights over a set of expert modules. All experts execute on the current
// stream, and their outputs are combined using the router's weights.
//
// The router is responsible for its own normalization strategy:
//   - Softmax for standard MoE (weights sum to 1)
//   - Sigmoid for independent gating (each expert 0-1)
//   - Top-k + softmax for sparse routing
//   - Or any other scheme — the merge just applies the weights as-is
//
// The router must output a tensor of shape [..., numExperts].
// Each expert receives the current stream and produces an output of the same shape.
//
// Use Using after Gate to wire additional tagged references to the router:
//
//	graph.From(encoder).
//	    Through(layer1).Tag("features").
//	    Through(layer2).
//	    Gate(router, expertA, expertB, expertC).Using("features").
//	    Through(decoder).
//	    Build()
func (fb *FlowBuilder) Gate(router nn.Module, experts ...nn.Module) *FlowBuilder {
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Gate requires single stream (got %d)", len(fb.current))
		return fb
	}
	if len(experts) < 2 {
		fb.err = fmt.Errorf("graph: Gate requires at least 2 experts (got %d)", len(experts))
		return fb
	}

	cur := fb.current[0]

	// Add router node.
	routerRef := fb.addModule(router)
	routerNode := fb.nodes[routerRef.id]
	fb.edges = append(fb.edges, &Edge{
		fromNode: cur.id,
		fromPort: cur.outputPort,
		toNode:   routerRef.id,
		toPort:   routerNode.inputPorts[0],
	})

	// Add expert nodes.
	expertRefs := make([]*nodeRef, len(experts))
	for i, expert := range experts {
		ref := fb.addModule(expert)
		node := fb.nodes[ref.id]
		fb.edges = append(fb.edges, &Edge{
			fromNode: cur.id,
			fromPort: cur.outputPort,
			toNode:   ref.id,
			toPort:   node.inputPorts[0],
		})
		expertRefs[i] = ref
	}

	// Add gated merge node.
	mergeRef := fb.addGatedMergeNode(len(experts))
	mergeNode := fb.nodes[mergeRef.id]

	// Wire router → merge (weights input).
	fb.edges = append(fb.edges, &Edge{
		fromNode: routerRef.id,
		fromPort: routerRef.outputPort,
		toNode:   mergeRef.id,
		toPort:   mergeNode.inputPorts[0],
	})

	// Wire experts → merge.
	for i, ref := range expertRefs {
		fb.edges = append(fb.edges, &Edge{
			fromNode: ref.id,
			fromPort: ref.outputPort,
			toNode:   mergeRef.id,
			toPort:   mergeNode.inputPorts[i+1],
		})
	}

	fb.onTarget = routerRef // Using() wires additional inputs to the router
	fb.current = []*nodeRef{mergeRef}
	return fb
}

// addGatedMergeNode creates an internal node that combines expert outputs
// using weights from a router. The weights are used as-is — the router
// is responsible for any normalization (softmax, sigmoid, top-k, etc.).
//
// Inputs: [router_weights, expert_0, expert_1, ..., expert_N-1]
// Output: sum_i(weights[..., i] * expert_i)
func (fb *FlowBuilder) addGatedMergeNode(numExperts int) *nodeRef {
	name := fmt.Sprintf("gated_merge_%d", fb.counter)
	fb.counter++

	inputPorts := make([]string, numExperts+1)
	inputPorts[0] = "weights"
	for i := range numExperts {
		inputPorts[i+1] = fmt.Sprintf("expert_%d", i)
	}

	run := func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		// Router weights used directly — normalization is the router's job.
		weights := inputs[0]
		dim := weights.Data().Ndim() - 1

		// Weighted sum of expert outputs.
		experts := inputs[1:]
		var result *autograd.Variable
		for i, expert := range experts {
			w := weights.Select(dim, int64(i))
			if err := w.Err(); err != nil {
				return nil, fmt.Errorf("gated merge select weight %d: %w", i, err)
			}

			// Unsqueeze for broadcasting: [batch] → [batch, 1]
			if w.Data().Ndim() > 0 {
				wShape := append(w.Data().Shape(), 1)
				w = w.Reshape(wShape)
			}

			weighted := w.Mul(expert)
			if result == nil {
				result = weighted
			} else {
				result = result.Add(weighted)
			}
			if err := result.Err(); err != nil {
				return nil, fmt.Errorf("gated merge expert %d: %w", i, err)
			}
		}

		return []*autograd.Variable{result}, nil
	}

	fb.nodes[name] = &Node{
		id:          name,
		inputPorts:  inputPorts,
		outputPorts: []string{DefaultOutput},
		run:         run,
		params:      func() []*nn.Parameter { return nil },
	}
	return &nodeRef{id: name, outputPort: DefaultOutput}
}
