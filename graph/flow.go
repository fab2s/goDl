package graph

import (
	"fmt"
	"reflect"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

const (
	// DefaultInput is the port name for single-input modules.
	DefaultInput = "input"
	// DefaultOutput is the port name for single-output modules.
	DefaultOutput = "output"
)

// nodeRef tracks the current position in the flow being built.
type nodeRef struct {
	id         string
	outputPort string
}

// FlowBuilder builds a graph using a fluent API that reads as data flow.
//
//	g, err := graph.From(encoder).
//	    Through(attention).
//	    Through(decoder).
//	    Build()
type FlowBuilder struct {
	nodes   map[string]*Node
	edges   []*Edge
	inputs  []exposedPort
	outputs []exposedPort
	current []*nodeRef // current stream position(s) — >1 after Split
	counter int
	err     error
}

// From starts a new graph flow at the given module.
// The module's input ports become the graph's inputs.
func From(m nn.Module) *FlowBuilder {
	fb := &FlowBuilder{
		nodes: make(map[string]*Node),
	}

	ref := fb.addModule(m)
	fb.current = []*nodeRef{ref}

	// Expose the entry node's input ports as graph-level inputs.
	node := fb.nodes[ref.id]
	for _, port := range node.inputPorts {
		fb.inputs = append(fb.inputs, exposedPort{
			name:   port,
			nodeID: ref.id,
			port:   port,
		})
	}

	return fb
}

// Through passes the flow through a module. Requires a single stream
// (call Merge first if the flow is split).
func (fb *FlowBuilder) Through(m nn.Module) *FlowBuilder {
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Through requires single stream (got %d); use Merge first", len(fb.current))
		return fb
	}

	ref := fb.addModule(m)
	cur := fb.current[0]
	node := fb.nodes[ref.id]

	fb.edges = append(fb.edges, &Edge{
		fromNode: cur.id,
		fromPort: cur.outputPort,
		toNode:   ref.id,
		toPort:   node.inputPorts[0],
	})

	fb.current = []*nodeRef{ref}
	return fb
}

// Also creates a residual connection: the input passes through the module,
// and the result is added element-wise back to the original.
// output = input + module(input)
func (fb *FlowBuilder) Also(m nn.Module) *FlowBuilder {
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Also requires single stream (got %d)", len(fb.current))
		return fb
	}

	cur := fb.current[0]

	// Add the transform module.
	transform := fb.addModule(m)
	transformNode := fb.nodes[transform.id]
	fb.edges = append(fb.edges, &Edge{
		fromNode: cur.id,
		fromPort: cur.outputPort,
		toNode:   transform.id,
		toPort:   transformNode.inputPorts[0],
	})

	// Add an element-wise add node to merge original + transformed.
	merge := fb.addAddNode(2)
	mergeNode := fb.nodes[merge.id]
	fb.edges = append(fb.edges, &Edge{
		fromNode: cur.id,
		fromPort: cur.outputPort,
		toNode:   merge.id,
		toPort:   mergeNode.inputPorts[0],
	})
	fb.edges = append(fb.edges, &Edge{
		fromNode: transform.id,
		fromPort: transform.outputPort,
		toNode:   merge.id,
		toPort:   mergeNode.inputPorts[1],
	})

	fb.current = []*nodeRef{merge}
	return fb
}

// Split forks the flow into parallel branches, one per module.
// Each branch receives the current stream's output as input.
// Call Merge to recombine the branches.
func (fb *FlowBuilder) Split(modules ...nn.Module) *FlowBuilder {
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Split requires single stream (got %d)", len(fb.current))
		return fb
	}
	if len(modules) < 2 {
		fb.err = fmt.Errorf("graph: Split requires at least 2 branches")
		return fb
	}

	cur := fb.current[0]
	refs := make([]*nodeRef, len(modules))

	for i, m := range modules {
		ref := fb.addModule(m)
		node := fb.nodes[ref.id]
		fb.edges = append(fb.edges, &Edge{
			fromNode: cur.id,
			fromPort: cur.outputPort,
			toNode:   ref.id,
			toPort:   node.inputPorts[0],
		})
		refs[i] = ref
	}

	fb.current = refs
	return fb
}

// Merge combines parallel streams using the given module.
// The module receives all branch outputs as its variadic inputs.
func (fb *FlowBuilder) Merge(m nn.Module) *FlowBuilder {
	if fb.err != nil {
		return fb
	}
	if len(fb.current) < 2 {
		fb.err = fmt.Errorf("graph: Merge requires multiple streams (got %d)", len(fb.current))
		return fb
	}

	ref := fb.addMergeModule(m, len(fb.current))
	node := fb.nodes[ref.id]

	for i, cur := range fb.current {
		fb.edges = append(fb.edges, &Edge{
			fromNode: cur.id,
			fromPort: cur.outputPort,
			toNode:   ref.id,
			toPort:   node.inputPorts[i],
		})
	}

	fb.current = []*nodeRef{ref}
	return fb
}

// Build finalizes the graph. The current stream's output becomes the
// graph's output. Returns an error if the flow has open branches or
// structural problems.
func (fb *FlowBuilder) Build() (*Graph, error) {
	if fb.err != nil {
		return nil, fb.err
	}
	if len(fb.current) != 1 {
		return nil, fmt.Errorf("graph: cannot build with %d open streams; use Merge to combine", len(fb.current))
	}

	// Expose the final node's output as the graph output.
	cur := fb.current[0]
	fb.outputs = append(fb.outputs, exposedPort{
		name:   DefaultOutput,
		nodeID: cur.id,
		port:   cur.outputPort,
	})

	return buildGraph(fb.nodes, fb.edges, fb.inputs, fb.outputs)
}

// --- internal helpers ---

func (fb *FlowBuilder) addModule(m nn.Module) *nodeRef {
	name := fb.autoName(m)
	run, params := wrapModule(m)
	fb.nodes[name] = &Node{
		id:          name,
		inputPorts:  []string{DefaultInput},
		outputPorts: []string{DefaultOutput},
		run:         run,
		params:      params,
	}
	return &nodeRef{id: name, outputPort: DefaultOutput}
}

// addAddNode creates an internal node that element-wise adds n inputs.
// Used by Also() for residual connections.
func (fb *FlowBuilder) addAddNode(n int) *nodeRef {
	name := fmt.Sprintf("add_%d", fb.counter)
	fb.counter++

	inputPorts := make([]string, n)
	for i := range n {
		inputPorts[i] = fmt.Sprintf("input_%d", i)
	}

	run := func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		result := inputs[0]
		for i := 1; i < len(inputs); i++ {
			result = result.Add(inputs[i])
		}
		if err := result.Err(); err != nil {
			return nil, err
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

// addMergeModule wraps a user-provided merge module with n input ports.
func (fb *FlowBuilder) addMergeModule(m nn.Module, n int) *nodeRef {
	name := fb.autoName(m)

	inputPorts := make([]string, n)
	for i := range n {
		inputPorts[i] = fmt.Sprintf("input_%d", i)
	}

	run, params := wrapModule(m)
	fb.nodes[name] = &Node{
		id:          name,
		inputPorts:  inputPorts,
		outputPorts: []string{DefaultOutput},
		run:         run,
		params:      params,
	}
	return &nodeRef{id: name, outputPort: DefaultOutput}
}

func (fb *FlowBuilder) autoName(v any) string {
	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	base := t.Name()
	if base == "" {
		base = "node"
	}
	name := fmt.Sprintf("%s_%d", base, fb.counter)
	fb.counter++
	return name
}
