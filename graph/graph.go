// Package graph provides a composable execution graph where nodes are Modules.
//
// A Graph is itself a Module — the key composition primitive. Any trained
// graph can be placed as a node inside a larger graph, enabling hierarchical
// composition of models (Graph-as-Module).
//
// Build graphs using the fluent API:
//
//	g, err := graph.From(encoder).
//	    Through(relu).Tag("hidden").
//	    Through(decoder).Tag("loss").
//	    Build()
//
//	result := g.Forward(input)  // Graph implements nn.Module
//
// Nodes in the same topological level (no dependencies on each other)
// execute concurrently via goroutines. This naturally parallelizes
// Split branches without any special configuration.
//
// # Observation
//
// Tagged node outputs are captured during Forward and can be logged,
// collected into batch buffers, flushed to epoch-level history, and
// queried as trends. See [Graph.Log], [Graph.Collect], [Graph.Flush],
// and [Graph.Trend] in observe.go for details.
package graph

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// execCtx holds the context.Context for the current Forward call.
// Created at graph build time and shared with loop/map closures via
// pointer. ForwardCtx sets the value; loops and maps read it between
// iterations for cancellation and timeout support.
type execCtx struct {
	ctx context.Context
}

// Edge connects an output port of one node to an input port of another.
type Edge struct {
	fromNode string
	fromPort string
	toNode   string
	toPort   string
}

// exposedPort maps a graph-level port name to a specific node's port.
type exposedPort struct {
	name   string // graph-level name (e.g. "input", "output")
	nodeID string // which node
	port   string // which port on that node
}

// stateEntry holds a forward-reference state buffer. When Using appears
// before Tag in the flow, the state carries between Forward() calls —
// enabling recurrent connections inside loops and persistent state
// across training steps.
//
// On the first Forward() call, value is nil. The graph execution
// auto-fills nil state inputs with ZerosLike(stream), so modules
// never receive nil — they get zero tensors matching the stream shape.
type stateEntry struct {
	name       string
	readerID   string             // state_read node that outputs this value
	writerID   string             // node whose output populates the buffer
	writerPort string             // which output port to capture
	value      *autograd.Variable // current state (nil until first write)
}

// Graph is a composition of connected Nodes. It implements nn.Module,
// enabling Graph-as-Module composition — a graph can be a node in
// a parent graph.
//
// Nodes that have no dependencies on each other execute in parallel
// via goroutines. This is determined at Build time from the graph
// topology — no runtime scheduling overhead.
type Graph struct {
	nodes     map[string]*Node
	edges     []*Edge
	inputs    []exposedPort       // graph-level inputs → node input ports
	outputs   []exposedPort       // graph-level outputs → node output ports
	order     []*Node             // flat topological order (for Parameters)
	levels    [][]*Node           // grouped by execution level (for parallel Forward)
	edgesFrom map[string][]*Edge  // edges grouped by source node for fast routing
	state     []*stateEntry       // forward-reference state buffers
	tags      map[string]string   // tag name → node ID (for visualization)
	tagGroups map[string][]string // group name → suffixed tag names (from TagGroup)
	frozen    map[string]bool     // frozen tag names (for parameter freezing)

	// Context — see ForwardCtx.
	execCtx *execCtx // shared with loop/map closures for cancellation

	// Profiling — see profile.go.
	// When profiling is false (the default), no time.Now() calls are made.
	profiling     bool
	lastProfile   *Profile
	profileFunc   func(*Profile)       // OnProfile hook
	timingBuffer  map[string][]float64 // CollectTimings batch buffer (seconds)
	timingHistory map[string][]float64 // FlushTimings epoch history (seconds)

	// Observation — see observe.go.
	// tagsByNode and taggedOutputs are initialized at Build time and
	// populated automatically during Forward (zero runtime cost when
	// no tags exist). The remaining fields are lazily initialized by
	// the first Collect/OnLog/OnFlush call.
	tagsByNode    map[string]string                   // node ID → tag name (reverse of tags)
	taggedOutputs map[string]*autograd.Variable       // last Forward's tagged outputs
	batchBuffer   map[string][]float64                // Collect accumulation (within epoch)
	epochHistory  map[string][]float64                // epoch means (populated by Flush)
	logFunc       func(map[string]*autograd.Variable) // Log hook
	flushFunc     func(map[string]float64)            // Flush hook
}

// Forward executes the graph, routing variables along edges between nodes.
// Nodes in the same topological level run concurrently via goroutines.
// Implements nn.Module.
//
// For cancellation and timeout support, use [Graph.ForwardCtx] instead.
func (g *Graph) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return g.ForwardCtx(context.Background(), inputs...)
}

// ForwardCtx executes the graph with a context for cancellation and timeouts.
// Loops (For, While, Until) and Maps check ctx between iterations; if the
// context is cancelled or its deadline exceeded, execution returns the
// context error. The context is also checked between topological levels.
//
//	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
//	defer cancel()
//	result := g.ForwardCtx(ctx, input)
func (g *Graph) ForwardCtx(ctx context.Context, inputs ...*autograd.Variable) *autograd.Variable {
	g.execCtx.ctx = ctx
	defer func() { g.execCtx.ctx = context.Background() }()

	var forwardStart time.Time
	if g.profiling {
		forwardStart = time.Now()
		g.lastProfile = &Profile{
			Levels: make([]LevelTiming, 0, len(g.levels)),
			Nodes:  make([]NodeTiming, 0, len(g.order)),
		}
	}

	if len(inputs) != len(g.inputs) {
		return autograd.ErrVariable(fmt.Errorf(
			"graph: expected %d inputs, got %d", len(g.inputs), len(inputs)))
	}

	// Allocate input slots for each node.
	slots := make(map[string][]*autograd.Variable, len(g.nodes))
	for _, node := range g.order {
		slots[node.id] = make([]*autograd.Variable, len(node.inputPorts))
	}

	// Route graph-level inputs to node input ports.
	for i, ep := range g.inputs {
		node := g.nodes[ep.nodeID]
		idx := portIndex(node.inputPorts, ep.port)
		slots[ep.nodeID][idx] = inputs[i]
	}

	// Clear tagged outputs from previous Forward.
	for k := range g.taggedOutputs {
		delete(g.taggedOutputs, k)
	}

	// Execute level by level. Within a level, nodes are independent.
	nodeOutputs := make(map[string][]*autograd.Variable, len(g.nodes))
	for i, level := range g.levels {
		if err := ctx.Err(); err != nil {
			return autograd.ErrVariable(err)
		}
		if err := g.executeLevel(i, level, slots, nodeOutputs); err != nil {
			return autograd.ErrVariable(err)
		}
	}

	if g.profiling {
		g.lastProfile.Total = time.Since(forwardStart)
		if g.profileFunc != nil {
			g.profileFunc(g.lastProfile)
		}
	}

	// Return graph-level output.
	ref := g.outputs[0]
	outNode := g.nodes[ref.nodeID]
	idx := portIndex(outNode.outputPorts, ref.port)
	return nodeOutputs[ref.nodeID][idx]
}

// executeLevel runs all nodes in a level and routes their outputs.
// Single-node levels execute directly (no goroutine overhead).
// Multi-node levels execute concurrently via goroutines.
func (g *Graph) executeLevel(
	levelIdx int,
	level []*Node,
	slots map[string][]*autograd.Variable,
	nodeOutputs map[string][]*autograd.Variable,
) error {
	var levelStart time.Time
	if g.profiling {
		levelStart = time.Now()
	}

	if len(level) == 1 {
		err := g.executeAndRoute(level[0], levelIdx, slots, nodeOutputs)
		if g.profiling {
			elapsed := time.Since(levelStart)
			g.lastProfile.Levels = append(g.lastProfile.Levels, LevelTiming{
				Index:     levelIdx,
				WallClock: elapsed,
				SumNodes:  elapsed,
				NumNodes:  1,
			})
		}
		return err
	}

	// Parallel execution: each goroutine writes to its own index.
	results := make([][]*autograd.Variable, len(level))
	errs := make([]error, len(level))
	var nodeTimes []time.Duration
	if g.profiling {
		nodeTimes = make([]time.Duration, len(level))
	}

	var wg sync.WaitGroup
	for i, node := range level {
		wg.Add(1)
		go func(idx int, n *Node) {
			defer wg.Done()
			var start time.Time
			if g.profiling {
				start = time.Now()
			}
			zeroFillNilInputs(slots[n.id])
			outs, err := n.run(slots[n.id])
			if g.profiling {
				nodeTimes[idx] = time.Since(start)
			}
			if err != nil {
				errs[idx] = fmt.Errorf("graph: node %q: %w", n.id, err)
				return
			}
			results[idx] = outs
		}(i, node)
	}
	wg.Wait()

	// Check errors, route outputs, and record per-node timings.
	var sumNodes time.Duration
	for i, node := range level {
		if errs[i] != nil {
			return errs[i]
		}
		nodeOutputs[node.id] = results[i]
		g.routeOutputs(node, results[i], slots)
		g.captureState(node, results[i])
		g.captureTagged(node, results[i])
		if g.profiling {
			sumNodes += nodeTimes[i]
			g.lastProfile.Nodes = append(g.lastProfile.Nodes, NodeTiming{
				ID:       node.id,
				Tag:      g.tagsByNode[node.id],
				Duration: nodeTimes[i],
				Level:    levelIdx,
			})
		}
	}

	if g.profiling {
		g.lastProfile.Levels = append(g.lastProfile.Levels, LevelTiming{
			Index:     levelIdx,
			WallClock: time.Since(levelStart),
			SumNodes:  sumNodes,
			NumNodes:  len(level),
		})
	}

	return nil
}

// zeroFillNilInputs replaces nil inputs (from uninitialized forward refs)
// with zero-valued variables matching the stream input's shape.
// This makes nil handling transparent — modules never receive nil.
func zeroFillNilInputs(inputs []*autograd.Variable) {
	if len(inputs) < 2 || inputs[0] == nil {
		return
	}
	for i := 1; i < len(inputs); i++ {
		if inputs[i] == nil {
			inputs[i] = autograd.NewVariable(inputs[0].Data().ZerosLike(), false)
		}
	}
}

// executeAndRoute runs a single node and routes its outputs downstream.
func (g *Graph) executeAndRoute(
	node *Node,
	levelIdx int,
	slots map[string][]*autograd.Variable,
	nodeOutputs map[string][]*autograd.Variable,
) error {
	var start time.Time
	if g.profiling {
		start = time.Now()
	}

	zeroFillNilInputs(slots[node.id])
	outs, err := node.run(slots[node.id])

	if g.profiling {
		g.lastProfile.Nodes = append(g.lastProfile.Nodes, NodeTiming{
			ID:       node.id,
			Tag:      g.tagsByNode[node.id],
			Duration: time.Since(start),
			Level:    levelIdx,
		})
	}

	if err != nil {
		return fmt.Errorf("graph: node %q: %w", node.id, err)
	}
	nodeOutputs[node.id] = outs
	g.routeOutputs(node, outs, slots)
	g.captureState(node, outs)
	g.captureTagged(node, outs)
	return nil
}

// routeOutputs copies a node's outputs to downstream nodes' input slots.
func (g *Graph) routeOutputs(
	node *Node,
	outs []*autograd.Variable,
	slots map[string][]*autograd.Variable,
) {
	for _, edge := range g.edgesFrom[node.id] {
		fromIdx := portIndex(node.outputPorts, edge.fromPort)
		toNode := g.nodes[edge.toNode]
		toIdx := portIndex(toNode.inputPorts, edge.toPort)
		slots[edge.toNode][toIdx] = outs[fromIdx]
	}
}

// Parameters collects all learnable parameters from all nodes in the graph.
// Deduplicates by pointer identity so shared parameters (weight tying)
// are not counted twice.
func (g *Graph) Parameters() []*nn.Parameter {
	seen := make(map[*nn.Parameter]bool)
	var params []*nn.Parameter
	for _, node := range g.order {
		if node.params == nil {
			continue
		}
		for _, p := range node.params() {
			if !seen[p] {
				seen[p] = true
				params = append(params, p)
			}
		}
	}
	return params
}

// captureState stores a node's output into any state buffer it writes to.
func (g *Graph) captureState(node *Node, outs []*autograd.Variable) {
	for _, s := range g.state {
		if s.writerID == node.id {
			idx := portIndex(node.outputPorts, s.writerPort)
			s.value = outs[idx]
		}
	}
}

// SetTraining propagates training mode to all modules in the graph.
// Modules that implement nn.TrainToggler (e.g., Dropout, BatchNorm)
// will switch behavior. Nested graphs propagate recursively.
func (g *Graph) SetTraining(training bool) {
	for _, node := range g.order {
		if node.module == nil {
			continue
		}
		nn.SetTraining(node.module, training)
	}
}

// ParametersByTag returns parameters belonging to the module at the
// tagged node. If the module is a sub-graph, its parameters are included
// recursively. Returns nil for unknown tags or nodes without parameters.
//
// Use this for selective parameter freezing or per-group learning rates:
//
//	// Freeze encoder parameters after backward.
//	for _, p := range g.ParametersByTag("encoder") {
//	    p.ZeroGrad()
//	}
//	optimizer.Step()
func (g *Graph) ParametersByTag(tag string) []*nn.Parameter {
	nodeID, ok := g.tags[tag]
	if !ok {
		return nil
	}
	node, ok := g.nodes[nodeID]
	if !ok || node.params == nil {
		return nil
	}
	return node.params()
}

// Freeze prevents gradient updates for parameters at the tagged node(s).
// After calling Freeze, subsequent Backward() calls will still compute
// gradients through frozen nodes (for upstream flow), but their parameter
// gradients are zeroed before optimizer.Step().
//
// Call after Build():
//
//	g.Freeze("encoder")       // freeze one tag
//	g.Freeze("embed", "norm") // freeze multiple tags
//
// Use Unfreeze to re-enable gradient updates.
func (g *Graph) Freeze(tags ...string) {
	if g.frozen == nil {
		g.frozen = make(map[string]bool)
	}
	for _, tag := range tags {
		g.frozen[tag] = true
	}
}

// Unfreeze re-enables gradient updates for previously frozen tags.
func (g *Graph) Unfreeze(tags ...string) {
	for _, tag := range tags {
		delete(g.frozen, tag)
	}
}

// ZeroFrozenGrads zeroes out gradients for all frozen parameters.
// Call between Backward() and optimizer.Step():
//
//	loss.Backward()
//	g.ZeroFrozenGrads()
//	optimizer.Step()
func (g *Graph) ZeroFrozenGrads() {
	for tag := range g.frozen {
		for _, p := range g.ParametersByTag(tag) {
			p.ZeroGrad()
		}
	}
}

// expandGroups expands tag group names into their member tags.
// Non-group tags pass through unchanged.
func (g *Graph) expandGroups(tags []string) []string {
	if g.tagGroups == nil {
		return tags
	}
	var expanded []string
	for _, tag := range tags {
		if members, ok := g.tagGroups[tag]; ok {
			expanded = append(expanded, members...)
		} else {
			expanded = append(expanded, tag)
		}
	}
	return expanded
}

// TagGroup returns the member tags of a tag group, or nil if the
// group name is not registered.
func (g *Graph) TagGroup(name string) []string {
	if g.tagGroups == nil {
		return nil
	}
	return g.tagGroups[name]
}

// ResetState clears all forward-reference state buffers to nil.
// Call this when starting inference on a new sequence.
func (g *Graph) ResetState() {
	for _, s := range g.state {
		s.value = nil
	}
}

// DetachState breaks the gradient chain on all state buffers.
// Call this between training steps to prevent unbounded graph growth.
func (g *Graph) DetachState() {
	for _, s := range g.state {
		if s.value != nil {
			s.value = autograd.NewVariable(s.value.Data(), false)
		}
	}
}

// topologicalLevels groups nodes by execution level using Kahn's algorithm.
// Nodes in the same level have no dependencies on each other and can
// execute in parallel. Returns an error if the graph contains a cycle.
func topologicalLevels(nodes map[string]*Node, edges []*Edge) ([][]*Node, error) {
	// Build dependency sets: for each node, which nodes must run before it.
	deps := make(map[string]map[string]bool, len(nodes))
	for id := range nodes {
		deps[id] = make(map[string]bool)
	}
	for _, edge := range edges {
		deps[edge.toNode][edge.fromNode] = true
	}

	// Initialize in-degree from dependency sets.
	inDegree := make(map[string]int, len(nodes))
	for id, d := range deps {
		inDegree[id] = len(d)
	}

	// Seed with nodes that have no dependencies.
	var queue []string
	for id, deg := range inDegree {
		if deg == 0 {
			queue = append(queue, id)
		}
	}

	var levels [][]*Node
	for len(queue) > 0 {
		// All nodes in the current queue can execute in parallel.
		level := make([]*Node, len(queue))
		for i, id := range queue {
			level[i] = nodes[id]
		}
		levels = append(levels, level)

		// Find the next level: nodes whose dependencies are all resolved.
		var next []string
		for _, id := range queue {
			for toID, d := range deps {
				if d[id] {
					delete(d, id)
					inDegree[toID]--
					if inDegree[toID] == 0 {
						next = append(next, toID)
					}
				}
			}
		}
		queue = next
	}

	// Check for cycles.
	total := 0
	for _, level := range levels {
		total += len(level)
	}
	if total != len(nodes) {
		return nil, fmt.Errorf("graph: cycle detected (%d nodes, %d reachable)", len(nodes), total)
	}

	return levels, nil
}

// buildGraph validates and finalizes a graph from its components.
func buildGraph(nodes map[string]*Node, edges []*Edge, inputs, outputs []exposedPort, fwdRefs []forwardRef, tags map[string]string, tagGroups map[string][]string, ec *execCtx) (*Graph, error) {
	// Validate edges reference valid nodes and ports.
	for _, edge := range edges {
		fromNode, ok := nodes[edge.fromNode]
		if !ok {
			return nil, fmt.Errorf("graph: edge references unknown source node %q", edge.fromNode)
		}
		if portIndex(fromNode.outputPorts, edge.fromPort) < 0 {
			return nil, fmt.Errorf("graph: node %q has no output port %q", edge.fromNode, edge.fromPort)
		}
		toNode, ok := nodes[edge.toNode]
		if !ok {
			return nil, fmt.Errorf("graph: edge references unknown target node %q", edge.toNode)
		}
		if portIndex(toNode.inputPorts, edge.toPort) < 0 {
			return nil, fmt.Errorf("graph: node %q has no input port %q", edge.toNode, edge.toPort)
		}
	}

	// Validate no duplicate edges to the same input port.
	portWired := make(map[string]bool)
	for _, edge := range edges {
		key := edge.toNode + ":" + edge.toPort
		if portWired[key] {
			return nil, fmt.Errorf("graph: duplicate edge to port %q on node %q", edge.toPort, edge.toNode)
		}
		portWired[key] = true
	}

	// Validate exposed ports.
	for _, ep := range inputs {
		node, ok := nodes[ep.nodeID]
		if !ok {
			return nil, fmt.Errorf("graph: input %q references unknown node %q", ep.name, ep.nodeID)
		}
		if portIndex(node.inputPorts, ep.port) < 0 {
			return nil, fmt.Errorf("graph: input %q references unknown port %q on node %q", ep.name, ep.port, ep.nodeID)
		}
	}
	for _, ep := range outputs {
		node, ok := nodes[ep.nodeID]
		if !ok {
			return nil, fmt.Errorf("graph: output %q references unknown node %q", ep.name, ep.nodeID)
		}
		if portIndex(node.outputPorts, ep.port) < 0 {
			return nil, fmt.Errorf("graph: output %q references unknown port %q on node %q", ep.name, ep.port, ep.nodeID)
		}
	}

	// Validate RefValidator contracts (build-time, zero runtime cost).
	if err := validateRefContracts(nodes); err != nil {
		return nil, err
	}

	// Set up forward-reference state buffers and wire state read nodes.
	state := make([]*stateEntry, 0, len(fwdRefs))
	for _, fr := range fwdRefs {
		entry := &stateEntry{
			name:       fr.name,
			readerID:   fr.readerID,
			writerID:   fr.writerID,
			writerPort: fr.writerPort,
		}
		state = append(state, entry)

		// Wire the state read node to return the buffer value.
		nodes[fr.readerID].run = func(_ []*autograd.Variable) ([]*autograd.Variable, error) {
			return []*autograd.Variable{entry.value}, nil
		}
	}

	// Compute execution levels (for parallel execution) and flat order.
	levels, err := topologicalLevels(nodes, edges)
	if err != nil {
		return nil, err
	}

	var order []*Node
	for _, level := range levels {
		order = append(order, level...)
	}

	// Build edge lookup by source node.
	edgesFrom := make(map[string][]*Edge, len(nodes))
	for _, edge := range edges {
		edgesFrom[edge.fromNode] = append(edgesFrom[edge.fromNode], edge)
	}

	// Build reverse tag lookup for tagged output capture during Forward.
	tagsByNode := make(map[string]string, len(tags))
	for name, nodeID := range tags {
		tagsByNode[nodeID] = name
	}

	return &Graph{
		nodes:         nodes,
		edges:         edges,
		inputs:        inputs,
		outputs:       outputs,
		order:         order,
		levels:        levels,
		edgesFrom:     edgesFrom,
		state:         state,
		tags:          tags,
		tagGroups:     tagGroups,
		execCtx:       ec,
		tagsByNode:    tagsByNode,
		taggedOutputs: make(map[string]*autograd.Variable, len(tags)),
	}, nil
}

// portIndex returns the index of a port name in a port list, or -1.
func portIndex(ports []string, name string) int {
	for i, p := range ports {
		if p == name {
			return i
		}
	}
	return -1
}
