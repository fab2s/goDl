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

// pendingUsing records a forward reference — Using called before the matching Tag.
type pendingUsing struct {
	readerID string // state_read node that was created
	targetID string // node that consumes this state
}

// forwardRef is a resolved forward reference (Using before Tag, now matched).
type forwardRef struct {
	name       string
	readerID   string // state_read node ID
	writerID   string // node that Tag points to
	writerPort string // output port on the writer
}

// FlowBuilder builds a graph using a fluent API that reads as data flow.
//
//	g, err := graph.From(encoder).
//	    Through(attention).
//	    Through(decoder).
//	    Build()
type FlowBuilder struct {
	nodes       map[string]*Node
	edges       []*Edge
	inputs      []exposedPort
	outputs     []exposedPort
	current     []*nodeRef // current stream position(s) — >1 after Split
	taps        map[string]*nodeRef
	onTarget    *nodeRef                  // node that Using() wires to
	pending     map[string][]pendingUsing // forward refs awaiting Tag
	forwardRefs []forwardRef              // resolved forward refs
	counter     int
	err         error
	openBuilder string // non-empty when a Loop/Map builder hasn't been finalized
}

// From starts a new graph flow at the given module.
// The module's input ports become the graph's inputs.
func From(m nn.Module) *FlowBuilder {
	fb := &FlowBuilder{
		nodes: make(map[string]*Node),
	}

	ref := fb.addModule(m)
	fb.current = []*nodeRef{ref}
	fb.onTarget = ref

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
	fb.onTarget = ref
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
	fb.onTarget = transform
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
	fb.onTarget = nil
	return fb
}

// Map starts a map construct that applies a body module independently
// to each element along dim 0 of a tensor. Results are concatenated
// back along dim 0.
//
// Call .Over(tag) to iterate over a tagged tensor, or .Each() to
// iterate over the current stream. Additional Using refs are broadcast
// to every invocation.
//
//	graph.From(positionDecoder).
//	    Map(readHead).Each().Using("image").
//	    Through(decoder).
//	    Build()
func (fb *FlowBuilder) Map(body nn.Module) *MapBuilder {
	fb.openBuilder = "Map (call .Each() or .Over(tag) to finalize)"
	return &MapBuilder{fb: fb, body: body}
}

// Loop starts a loop construct that repeats a body module, carrying
// state between iterations. Call .For(n) to set the iteration count.
//
//	graph.From(encoder).
//	    Loop(refinementStep).For(5).
//	    Through(decoder).
//	    Build()
func (fb *FlowBuilder) Loop(body nn.Module) *LoopBuilder {
	fb.openBuilder = "Loop (call .For(n), .While(cond, max), or .Until(cond, max) to finalize)"
	return &LoopBuilder{fb: fb, body: body}
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
	fb.onTarget = ref
	return fb
}

// Tag names the current position in the flow so it can be referenced
// later with Using. The name must be unique within the graph.
//
//	graph.From(encoder).
//	    Through(layer1).Tag("hidden").
//	    Through(layer2).
//	    Through(crossAttn).Using("hidden").
//	    Build()
func (fb *FlowBuilder) Tag(name string) *FlowBuilder {
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Tag requires single stream (got %d)", len(fb.current))
		return fb
	}
	if fb.taps == nil {
		fb.taps = make(map[string]*nodeRef)
	}
	if _, exists := fb.taps[name]; exists {
		fb.err = fmt.Errorf("graph: duplicate tag name %q", name)
		return fb
	}

	cur := fb.current[0]
	fb.taps[name] = cur

	// Resolve any pending forward references to this tag.
	if pending, ok := fb.pending[name]; ok {
		for _, p := range pending {
			fb.forwardRefs = append(fb.forwardRefs, forwardRef{
				name:       name,
				readerID:   p.readerID,
				writerID:   cur.id,
				writerPort: cur.outputPort,
			})
		}
		delete(fb.pending, name)
	}

	return fb
}

// Using wires additional inputs from previously tagged points to the
// preceding node(s). After Through, Gate, Merge, or Also, it targets
// that single node. After Split, it targets all branch modules —
// each branch receives the tagged references as extra Forward arguments.
//
//	// Single target:
//	graph.From(encoder).Tag("memory").
//	    Through(crossAttention).Using("memory").
//	    Build()
//
//	// All branches:
//	graph.From(encoder).Tag("memory").
//	    Split(headA, headB, headC).Using("memory").
//	    Merge(concat).
//	    Build()
func (fb *FlowBuilder) Using(refs ...string) *FlowBuilder {
	if fb.err != nil {
		return fb
	}
	if len(refs) == 0 {
		return fb
	}

	// Determine targets: single node or all split branches.
	var targets []*nodeRef
	switch {
	case fb.onTarget != nil:
		targets = []*nodeRef{fb.onTarget}
	case len(fb.current) > 1:
		targets = fb.current
	default:
		fb.err = fmt.Errorf("graph: Using requires a preceding Through, Gate, Split, or Merge")
		return fb
	}

	for _, target := range targets {
		if err := fb.wireUsing(target, refs); err != nil {
			fb.err = err
			return fb
		}
	}
	return fb
}

// wireUsing adds tagged references as extra input ports to a single node.
// If a tag hasn't been set yet (Using before Tag), it creates a forward
// reference — state carried between Forward() calls.
func (fb *FlowBuilder) wireUsing(target *nodeRef, refs []string) error {
	node := fb.nodes[target.id]
	for _, ref := range refs {
		portName := fmt.Sprintf("ref_%s", ref)

		tap, ok := fb.taps[ref]
		if ok {
			// Backward reference: tag already exists, wire directly.
			node.inputPorts = append(node.inputPorts, portName)
			fb.edges = append(fb.edges, &Edge{
				fromNode: tap.id,
				fromPort: tap.outputPort,
				toNode:   target.id,
				toPort:   portName,
			})
			continue
		}

		// Forward reference: tag not set yet. Create a state reader node.
		readerRef := fb.addStateReadNode(ref)
		node.inputPorts = append(node.inputPorts, portName)
		fb.edges = append(fb.edges, &Edge{
			fromNode: readerRef.id,
			fromPort: readerRef.outputPort,
			toNode:   target.id,
			toPort:   portName,
		})
		if fb.pending == nil {
			fb.pending = make(map[string][]pendingUsing)
		}
		fb.pending[ref] = append(fb.pending[ref], pendingUsing{
			readerID: readerRef.id,
			targetID: target.id,
		})
	}
	return nil
}

// Build finalizes the graph. The current stream's output becomes the
// graph's output. Returns an error if the flow has open branches or
// structural problems.
func (fb *FlowBuilder) Build() (*Graph, error) {
	if fb.err != nil {
		return nil, fb.err
	}
	if fb.openBuilder != "" {
		return nil, fmt.Errorf("graph: incomplete builder: %s", fb.openBuilder)
	}
	if len(fb.current) != 1 {
		return nil, fmt.Errorf("graph: cannot build with %d open streams; use Merge to combine", len(fb.current))
	}

	// Check for unresolved forward references.
	for name := range fb.pending {
		return nil, fmt.Errorf("graph: unresolved forward reference %q (Using without matching Tag)", name)
	}

	// Expose the final node's output as the graph output.
	cur := fb.current[0]
	fb.outputs = append(fb.outputs, exposedPort{
		name:   DefaultOutput,
		nodeID: cur.id,
		port:   cur.outputPort,
	})

	// Convert taps to tag name → node ID map for visualization.
	tags := make(map[string]string, len(fb.taps))
	for name, ref := range fb.taps {
		tags[name] = ref.id
	}

	return buildGraph(fb.nodes, fb.edges, fb.inputs, fb.outputs, fb.forwardRefs, tags)
}

// --- internal helpers ---

func (fb *FlowBuilder) addModule(m nn.Module) *nodeRef {
	name := fb.autoName(m)
	node := &Node{
		id:          name,
		inputPorts:  []string{DefaultInput},
		outputPorts: []string{DefaultOutput},
		params:      m.Parameters,
		module:      m,
	}
	node.run = wrapModule(m, node)
	fb.nodes[name] = node
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

	node := &Node{
		id:          name,
		inputPorts:  inputPorts,
		outputPorts: []string{DefaultOutput},
		params:      m.Parameters,
		module:      m,
	}
	node.run = wrapModule(m, node)
	fb.nodes[name] = node
	return &nodeRef{id: name, outputPort: DefaultOutput}
}

// addStateReadNode creates a root node that reads from a state buffer.
// Its run function is set during buildGraph once the state entry exists.
func (fb *FlowBuilder) addStateReadNode(name string) *nodeRef {
	nodeID := fmt.Sprintf("state_read_%s_%d", name, fb.counter)
	fb.counter++
	fb.nodes[nodeID] = &Node{
		id:          nodeID,
		inputPorts:  []string{},
		outputPorts: []string{DefaultOutput},
		run:         nil, // set by buildGraph
		params:      func() []*nn.Parameter { return nil },
	}
	return &nodeRef{id: nodeID, outputPort: DefaultOutput}
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
