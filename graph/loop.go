package graph

import (
	"fmt"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// LoopBuilder configures a loop construct in the graph flow.
// A loop repeats a body module, carrying state between iterations.
//
// The body receives the current state as input and returns the new state.
// After all iterations, the final state continues downstream.
//
// Three termination modes:
//   - For(n): fixed iteration count, always runs exactly n times
//   - While(cond, maxIter): condition checked before body (0..maxIter iterations)
//   - Until(cond, maxIter): condition checked after body (1..maxIter iterations)
//
// While and Until use the same halt convention: the condition module
// receives the current state and returns a scalar. Positive (> 0) means
// halt. They differ only in timing — While can skip the body entirely,
// Until always runs it at least once.
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

	return lb.wire(makeForLoopFunc(lb.body, n), lb.body)
}

// While repeats the body while the condition module says "continue",
// up to maxIter iterations. The condition is checked before each
// iteration — if it signals halt immediately, the body never runs
// and the input passes through unchanged.
//
// The condition module receives the current state and returns a scalar.
// Positive (> 0) means halt — same convention as Until.
//
//	graph.From(encoder).
//	    Loop(refine).While(graph.ThresholdHalt(100), 20).
//	    Through(decoder).
//	    Build()
func (lb *LoopBuilder) While(cond nn.Module, maxIter int) *FlowBuilder {
	fb := lb.fb
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Loop requires single stream (got %d)", len(fb.current))
		return fb
	}
	if maxIter < 1 {
		fb.err = fmt.Errorf("graph: Loop.While requires maxIter >= 1 (got %d)", maxIter)
		return fb
	}

	composite := &loopComposite{body: lb.body, cond: cond}
	return lb.wire(makeWhileLoopFunc(lb.body, cond, maxIter), composite)
}

// Until repeats the body until the condition module signals halt, up to
// maxIter iterations. The body always executes at least once.
//
// After each body execution, the condition module receives the state and
// returns a scalar. Iteration stops when the scalar is positive (> 0),
// indicating the halt condition is satisfied. The stop decision is
// non-differentiable — gradients flow through the body iterations.
//
// The condition module's parameters are included in Parameters(),
// and SetTraining propagates to it.
//
//	graph.From(encoder).
//	    Loop(refinement).Until(haltProbe, 20).
//	    Through(decoder).
//	    Build()
func (lb *LoopBuilder) Until(cond nn.Module, maxIter int) *FlowBuilder {
	fb := lb.fb
	if fb.err != nil {
		return fb
	}
	if len(fb.current) != 1 {
		fb.err = fmt.Errorf("graph: Loop requires single stream (got %d)", len(fb.current))
		return fb
	}
	if maxIter < 1 {
		fb.err = fmt.Errorf("graph: Loop.Until requires maxIter >= 1 (got %d)", maxIter)
		return fb
	}

	composite := &loopComposite{body: lb.body, cond: cond}
	return lb.wire(makeUntilLoopFunc(lb.body, cond, maxIter), composite)
}

// wire is a shared helper that wires a loop node into the graph.
func (lb *LoopBuilder) wire(run nodeFunc, mod nn.Module) *FlowBuilder {
	fb := lb.fb
	cur := fb.current[0]

	name := fmt.Sprintf("loop_%d", fb.counter)
	fb.counter++

	fb.nodes[name] = &Node{
		id:          name,
		inputPorts:  []string{DefaultInput},
		outputPorts: []string{DefaultOutput},
		run:         run,
		params:      mod.Parameters,
		module:      mod,
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

// --- loop executors ---

// makeForLoopFunc creates a nodeFunc that executes a body module N times,
// feeding each iteration's output as the next iteration's input.
func makeForLoopFunc(body nn.Module, count int) nodeFunc {
	return func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		state := inputs[0]
		for i := range count {
			state = body.Forward(state)
			if err := state.Err(); err != nil {
				return nil, fmt.Errorf("loop iteration %d: %w", i, err)
			}
		}
		return []*autograd.Variable{state}, nil
	}
}

// makeWhileLoopFunc creates a nodeFunc that checks a condition module
// before each body execution. Positive output = halt.
func makeWhileLoopFunc(body, cond nn.Module, maxIter int) nodeFunc {
	return func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		state := inputs[0]
		for i := range maxIter {
			halt := cond.Forward(state)
			if err := halt.Err(); err != nil {
				return nil, fmt.Errorf("loop condition at iteration %d: %w", i, err)
			}
			data, err := halt.Data().Float32Data()
			if err != nil {
				return nil, fmt.Errorf("loop condition data at iteration %d: %w", i, err)
			}
			if len(data) > 0 && data[0] > 0 {
				break
			}
			state = body.Forward(state)
			if err := state.Err(); err != nil {
				return nil, fmt.Errorf("loop iteration %d: %w", i, err)
			}
		}
		return []*autograd.Variable{state}, nil
	}
}

// makeUntilLoopFunc creates a nodeFunc that executes body until cond
// returns a positive scalar. The body always runs at least once.
func makeUntilLoopFunc(body, cond nn.Module, maxIter int) nodeFunc {
	return func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		state := inputs[0]
		for i := range maxIter {
			state = body.Forward(state)
			if err := state.Err(); err != nil {
				return nil, fmt.Errorf("loop iteration %d: %w", i, err)
			}
			// Skip condition check on last iteration (we stop regardless).
			if i < maxIter-1 {
				halt := cond.Forward(state)
				if err := halt.Err(); err != nil {
					return nil, fmt.Errorf("loop condition at iteration %d: %w", i, err)
				}
				data, err := halt.Data().Float32Data()
				if err != nil {
					return nil, fmt.Errorf("loop condition data at iteration %d: %w", i, err)
				}
				if len(data) > 0 && data[0] > 0 {
					break
				}
			}
		}
		return []*autograd.Variable{state}, nil
	}
}

// --- composite module for While and Until ---

// loopComposite bundles body + condition for SetTraining propagation
// and parameter collection. Used by both While and Until.
type loopComposite struct {
	body nn.Module
	cond nn.Module
}

func (lc *loopComposite) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return lc.body.Forward(inputs...)
}

func (lc *loopComposite) Parameters() []*nn.Parameter {
	all := lc.body.Parameters()
	all = append(all, lc.cond.Parameters()...)
	return all
}

func (lc *loopComposite) SetTraining(training bool) {
	nn.SetTraining(lc.body, training)
	nn.SetTraining(lc.cond, training)
}
