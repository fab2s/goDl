package graph

import (
	"fmt"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// MapBuilder configures a map construct in the graph flow.
// A map slices a tensor along dim 0 and runs a body module on each
// element independently. Results are concatenated back along dim 0.
//
// Two iteration sources:
//   - Over(tag): iterate over a tagged tensor (backward ref)
//   - Each(): iterate over the current stream
//
// Additional Using refs are broadcast — every body invocation receives
// the same value. If the body implements nn.NamedInputModule, broadcast
// refs are passed by tag name via ForwardNamed.
//
//	graph.From(positionDecoder).Tag("positions").
//	    Map(readHead).Over("positions").Using("image").
//	    Through(decoder).
//	    Build()
type MapBuilder struct {
	fb      *FlowBuilder
	body    nn.Module
	batched bool
}

// Batched enables the batched fast path: instead of iterating element
// by element (Narrow+Cat), the entire source tensor is passed to the
// body module in one call. This is significantly faster for stateless
// bodies (Linear, activations, etc.) that handle batch dimensions
// natively.
//
// Use only when the body module processes each batch element independently.
// Modules that normalize across the batch (e.g. BatchNorm) will produce
// different results in batched mode.
//
//	graph.From(encoder).
//	    Map(nn.MustLinear(4, 4)).Batched().Each().
//	    Build()
func (mb *MapBuilder) Batched() *MapBuilder {
	mb.batched = true
	return mb
}

// Over sets the iteration source to a tagged tensor. The tag must
// reference a point already defined (backward ref). Each element
// along dim 0 is passed to the body as its stream input.
func (mb *MapBuilder) Over(tag string) *FlowBuilder {
	fb := mb.fb
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Map requires single stream (got %d); use Merge first", len(fb.current))
		return fb
	}

	tap, ok := fb.taps[tag]
	if !ok {
		fb.err = fmt.Errorf("graph: Map.Over(%q) references unknown tag; Map requires a backward reference", tag)
		return fb
	}

	return mb.wire(tap)
}

// Each iterates over elements of the current stream (dim 0).
//
//	graph.From(positionDecoder).
//	    Map(readHead).Each().Using("image").
//	    Through(decoder).
//	    Build()
func (mb *MapBuilder) Each() *FlowBuilder {
	fb := mb.fb
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Map requires single stream (got %d); use Merge first", len(fb.current))
		return fb
	}

	return mb.wire(fb.current[0])
}

// Slices decomposes the last dimension of the current stream into n
// equal slices, applies the body to each, and recomposes the results.
// This handles dynamic batch sizes at runtime.
//
// For input [B, D] with Slices(n): reshape [B, D] → [B*n, D/n],
// map body over B*n elements, reshape back to [B, D'].
// D must be divisible by n. D' depends on the body's output dimension.
//
//	graph.From(encoder).
//	    Map(readHead(2)).Slices(4).   // [B, 8] → 4 positions × [B, 2] → [B, 8]
//	    Through(decoder).
//	    Build()
func (mb *MapBuilder) Slices(n int) *FlowBuilder {
	fb := mb.fb
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Map requires single stream (got %d); use Merge first", len(fb.current))
		return fb
	}
	if n < 1 {
		fb.err = fmt.Errorf("graph: Map.Slices requires n >= 1 (got %d)", n)
		return fb
	}

	fb.openBuilder = ""
	cur := fb.current[0]
	body := mb.body
	composite := &mapComposite{body: body}

	name := fmt.Sprintf("map_%d", fb.counter)
	fb.counter++

	node := &Node{
		id:          name,
		inputPorts:  []string{DefaultInput},
		outputPorts: []string{DefaultOutput},
		params:      composite.Parameters,
		module:      composite,
	}
	if mb.batched {
		node.run = makeBatchedSlicesMapFunc(body, n)
	} else {
		node.run = makeSlicesMapFunc(body, n, node, fb.execCtx)
	}
	fb.nodes[name] = node

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

// wire creates the map node and wires it into the graph.
func (mb *MapBuilder) wire(source *nodeRef) *FlowBuilder {
	fb := mb.fb
	fb.openBuilder = ""
	body := mb.body
	composite := &mapComposite{body: body}

	name := fmt.Sprintf("map_%d", fb.counter)
	fb.counter++

	node := &Node{
		id:          name,
		inputPorts:  []string{DefaultInput},
		outputPorts: []string{DefaultOutput},
		params:      composite.Parameters,
		module:      composite,
	}
	if mb.batched {
		node.run = makeBatchedMapFunc(body, node)
	} else {
		node.run = makeMapFunc(body, node, fb.execCtx)
	}
	fb.nodes[name] = node

	fb.edges = append(fb.edges, &Edge{
		fromNode: source.id,
		fromPort: source.outputPort,
		toNode:   name,
		toPort:   DefaultInput,
	})

	ref := &nodeRef{id: name, outputPort: DefaultOutput}
	fb.current = []*nodeRef{ref}
	fb.onTarget = ref
	return fb
}

// makeMapFunc creates a nodeFunc that slices inputs[0] along dim 0
// and runs the body module on each element. Additional inputs (from
// Using) are broadcast to every invocation.
// The execCtx is checked between elements for cancellation.
func makeMapFunc(body nn.Module, node *Node, ec *execCtx) nodeFunc {
	return func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		source := inputs[0]
		shape := source.Data().Shape()
		n := shape[0]
		if n == 0 {
			return nil, fmt.Errorf("map: source tensor has 0 elements along dim 0")
		}

		// Broadcast refs (inputs[1:]) — from Using calls.
		broadcastInputs := inputs[1:]
		named, hasNamed := body.(nn.NamedInputModule)

		results := make([]*autograd.Variable, n)
		for i := int64(0); i < n; i++ {
			if err := ec.ctx.Err(); err != nil {
				return nil, err
			}
			elem := source.Narrow(0, i, 1) // [1, ...rest]
			results[i] = mapBodyForward(body, named, hasNamed, elem, broadcastInputs, node.inputPorts[1:])
			if err := results[i].Err(); err != nil {
				return nil, fmt.Errorf("map element %d: %w", i, err)
			}
		}

		// Concatenate results along dim 0.
		stacked := results[0]
		for i := 1; i < len(results); i++ {
			stacked = stacked.Cat(results[i], 0)
		}
		if err := stacked.Err(); err != nil {
			return nil, fmt.Errorf("map concat: %w", err)
		}

		return []*autograd.Variable{stacked}, nil
	}
}

// makeSlicesMapFunc creates a nodeFunc that decomposes the last dimension
// into n slices, maps the body over each, and recomposes.
// The execCtx is checked between elements for cancellation.
func makeSlicesMapFunc(body nn.Module, n int, node *Node, ec *execCtx) nodeFunc {
	return func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		source := inputs[0]
		shape := source.Data().Shape()
		if len(shape) < 2 {
			return nil, fmt.Errorf("map slices: input must be at least 2D (got %dD)", len(shape))
		}

		lastDim := shape[len(shape)-1]
		if lastDim%int64(n) != 0 {
			return nil, fmt.Errorf("map slices: last dim %d not divisible by %d", lastDim, n)
		}
		sliceDim := lastDim / int64(n)
		origDim0 := shape[0]

		// Decompose: [B, ..., D] → [B*n, ..., D/n]
		decomposed := source.Reshape([]int64{origDim0 * int64(n), sliceDim})

		// Broadcast refs (inputs[1:]) — from Using calls.
		broadcastInputs := inputs[1:]
		named, hasNamed := body.(nn.NamedInputModule)

		numRows := origDim0 * int64(n)
		results := make([]*autograd.Variable, numRows)
		for i := int64(0); i < numRows; i++ {
			if err := ec.ctx.Err(); err != nil {
				return nil, err
			}
			elem := decomposed.Narrow(0, i, 1)
			results[i] = mapBodyForward(body, named, hasNamed, elem, broadcastInputs, node.inputPorts[1:])
			if err := results[i].Err(); err != nil {
				return nil, fmt.Errorf("map slices element %d: %w", i, err)
			}
		}

		// Cat results: [B*n, outD]
		stacked := results[0]
		for i := 1; i < len(results); i++ {
			stacked = stacked.Cat(results[i], 0)
		}

		// Recompose: [B*n, outD] → [B, outD*n]
		stackedShape := stacked.Data().Shape()
		outFeatures := stackedShape[1] * int64(n)
		recomposed := stacked.Reshape([]int64{origDim0, outFeatures})

		if err := recomposed.Err(); err != nil {
			return nil, fmt.Errorf("map slices recompose: %w", err)
		}
		return []*autograd.Variable{recomposed}, nil
	}
}

// mapComposite bundles the body module for SetTraining/Parameters.
type mapComposite struct {
	body nn.Module
}

func (mc *mapComposite) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return mc.body.Forward(inputs...)
}

func (mc *mapComposite) Parameters() []*nn.Parameter {
	return mc.body.Parameters()
}

func (mc *mapComposite) SetTraining(training bool) {
	nn.SetTraining(mc.body, training)
}

// mapBodyForward dispatches a single map element to the body, handling
// NamedInputModule and broadcast refs.
func mapBodyForward(body nn.Module, named nn.NamedInputModule, hasNamed bool, elem *autograd.Variable, broadcastInputs []*autograd.Variable, refPorts []string) *autograd.Variable {
	switch {
	case hasNamed && len(broadcastInputs) > 0:
		if refs := extractRefs(refPorts, broadcastInputs); refs != nil {
			return named.ForwardNamed(elem, refs)
		}
		return body.Forward(append([]*autograd.Variable{elem}, broadcastInputs...)...)
	case len(broadcastInputs) > 0:
		return body.Forward(append([]*autograd.Variable{elem}, broadcastInputs...)...)
	default:
		return body.Forward(elem)
	}
}

// makeBatchedMapFunc creates a nodeFunc that passes the entire source
// tensor to the body in one call, bypassing element-by-element iteration.
// The body must handle batch dimensions natively (e.g. Linear, activations).
func makeBatchedMapFunc(body nn.Module, _ *Node) nodeFunc {
	return func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		result := body.Forward(inputs...)
		if err := result.Err(); err != nil {
			return nil, fmt.Errorf("map batched: %w", err)
		}
		return []*autograd.Variable{result}, nil
	}
}

// makeBatchedSlicesMapFunc creates a nodeFunc that decomposes the last
// dimension into n slices, runs the body on the full decomposed batch
// in one call, and recomposes. Much faster than element-by-element.
func makeBatchedSlicesMapFunc(body nn.Module, n int) nodeFunc {
	return func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		source := inputs[0]
		shape := source.Data().Shape()
		if len(shape) < 2 {
			return nil, fmt.Errorf("map slices batched: input must be at least 2D (got %dD)", len(shape))
		}

		lastDim := shape[len(shape)-1]
		if lastDim%int64(n) != 0 {
			return nil, fmt.Errorf("map slices batched: last dim %d not divisible by %d", lastDim, n)
		}
		sliceDim := lastDim / int64(n)
		origDim0 := shape[0]

		// Decompose: [B, D] → [B*n, D/n]
		decomposed := source.Reshape([]int64{origDim0 * int64(n), sliceDim})

		// Run body on entire decomposed batch at once.
		result := body.Forward(decomposed)
		if err := result.Err(); err != nil {
			return nil, fmt.Errorf("map slices batched: %w", err)
		}

		// Recompose: [B*n, outD] → [B, outD*n]
		resultShape := result.Data().Shape()
		outFeatures := resultShape[1] * int64(n)
		recomposed := result.Reshape([]int64{origDim0, outFeatures})

		if err := recomposed.Err(); err != nil {
			return nil, fmt.Errorf("map slices batched recompose: %w", err)
		}
		return []*autograd.Variable{recomposed}, nil
	}
}
