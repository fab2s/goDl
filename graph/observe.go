// Observation layer for training metrics.
//
// Tags already name meaningful nodes in the graph. The observation layer
// builds on this: every Forward call captures tagged node outputs, making
// them available for logging, collection, and trend analysis.
//
// # Logging
//
// Log prints current tagged values. The default writes to stderr;
// replace it with OnLog for custom handling (structured logging, etc).
//
//	g.Forward(input)
//	g.Log("loss")                // loss: 0.2341
//	g.Log()                      // all tags
//
// # Collecting and Flushing
//
// Collect snapshots scalar values into a batch buffer (within an epoch).
// Flush promotes the batch mean to epoch-level history, then clears
// the buffer. This two-level structure gives you both fine-grained
// batch data and coarse-grained epoch trends from the same mechanism.
//
//	for epoch := range epochs {
//	    for _, batch := range loader {
//	        g.Forward(batch.Input)
//	        g.Collect("loss")        // one value per batch
//	    }
//	    g.Flush()                    // batch mean → epoch history
//	}
//
// # Trends
//
// Trend returns the epoch-level time series for a tag, with built-in
// statistical queries that drive training decisions:
//
//	if g.Trend("loss").Stalled(5, 1e-4) {
//	    scheduler.Decay()            // reduce LR on plateau
//	}
//	if g.Trend("loss").Improving(3) {
//	    g.Unfreeze("decoder")        // start fine-tuning
//	}
//
// # Sub-graph observation
//
// Since a Graph is a Module, sub-graphs run automatically as part of
// the parent's Forward — no separate Forward call needed. Use Sub to
// reach into a sub-graph and observe its internal tags:
//
//	// Build a sub-graph with its own tags.
//	refiner, _ := graph.From(step).Tag("residual").Build()
//
//	// Compose into root graph.
//	g, _ := graph.From(encoder).
//	    Through(refiner).Tag("refine").
//	    Through(lossModule).Tag("loss").
//	    Build()
//
//	// Training loop — only Forward on root.
//	for epoch := range epochs {
//	    for _, batch := range loader {
//	        g.Forward(batch.Input)
//	        g.Collect("loss")                     // root tag
//	        g.Sub("refine").Collect("residual")   // sub-graph tag
//	    }
//	    g.Flush()
//	    g.Sub("refine").Flush()
//
//	    if g.Trend("loss").Stalled(5, 1e-4) {
//	        scheduler.Decay()
//	    }
//	}
package graph

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/fab2s/goDl/autograd"
)

// Tagged returns the output of a tagged node from the last Forward call.
// Returns nil if the tag is unknown or Forward hasn't been called.
//
// Tagged values are captured automatically during Forward for every
// node that has a Tag. No explicit setup is needed.
func (g *Graph) Tagged(tag string) *autograd.Variable {
	if g.taggedOutputs == nil {
		return nil
	}
	return g.taggedOutputs[tag]
}

// Traces returns the per-iteration side outputs collected from a
// [nn.Traced] loop body at the tagged node. Returns nil if the tag
// is unknown, the node isn't a loop, or the body doesn't implement Traced.
//
// The slice contains one entry per iteration plus the initial state
// (captured after Reset, before the first iteration). For a loop with
// N iterations, Traces returns N+1 entries.
//
//	g.Forward(input)
//	locations := g.Traces("attention") // [initial, step1, step2, ...]
func (g *Graph) Traces(tag string) []*autograd.Variable {
	nodeID, ok := g.tags[tag]
	if !ok {
		return nil
	}
	node, ok := g.nodes[nodeID]
	if !ok {
		return nil
	}
	return node.traces
}

// Sub returns the sub-graph at a tagged node, or nil if the tag
// doesn't exist or the node isn't a Graph.
//
// The sub-graph's Forward runs automatically as part of the parent's
// Forward (Graph-as-Module), so its tagged outputs are already
// populated. Use Sub to observe the sub-graph's internal metrics
// without a separate Forward call:
//
//	g.Forward(input)                          // runs everything
//	g.Sub("encoder").Collect("attention")     // inner tag
func (g *Graph) Sub(tag string) *Graph {
	nodeID, ok := g.tags[tag]
	if !ok {
		return nil
	}
	node, ok := g.nodes[nodeID]
	if !ok {
		return nil
	}
	sub, _ := node.module.(*Graph)
	return sub
}

// Log prints the current values of the specified tagged nodes.
// If no tags are specified, all tagged values are printed.
// If OnLog has been set, calls the hook instead of printing.
//
//	g.Forward(input)
//	g.Log("loss", "accuracy")  // loss: 0.2341 | accuracy: 0.891
func (g *Graph) Log(tags ...string) {
	values := g.taggedValues(tags)
	if len(values) == 0 {
		return
	}
	if g.logFunc != nil {
		g.logFunc(values)
		return
	}
	defaultLog(values)
}

// OnLog sets a custom handler for Log calls. Pass nil to restore
// the default (print to stderr).
//
//	g.OnLog(func(values map[string]*autograd.Variable) {
//	    fmt.Printf("loss=%.4f\n", scalarValue(values["loss"]))
//	})
func (g *Graph) OnLog(fn func(values map[string]*autograd.Variable)) {
	g.logFunc = fn
}

// Collect snapshots the current scalar value of the specified tagged
// nodes into the batch buffer. Call Flush to promote the batch mean
// to epoch-level history and clear the buffer.
//
//	for _, batch := range loader {
//	    g.Forward(batch.Input)
//	    g.Collect("loss")
//	}
//	g.Flush("loss")
func (g *Graph) Collect(tags ...string) {
	if g.taggedOutputs == nil {
		return
	}
	if g.batchBuffer == nil {
		g.batchBuffer = make(map[string][]float64)
	}
	for _, tag := range tags {
		v, ok := g.taggedOutputs[tag]
		if !ok || v == nil {
			continue
		}
		g.batchBuffer[tag] = append(g.batchBuffer[tag], scalarValue(v))
	}
}

// Collected returns the raw batch-level buffer for a tag — all values
// since the last Flush. Returns nil if nothing has been collected.
func (g *Graph) Collected(tag string) []float64 {
	if g.batchBuffer == nil {
		return nil
	}
	return g.batchBuffer[tag]
}

// Flush promotes the mean of each tag's batch buffer to the epoch
// history, then clears the batch buffer. If no tags are specified,
// all buffered tags are flushed.
//
// If OnFlush has been set, calls the hook with the flushed values
// (tag → epoch mean).
//
//	for epoch := range epochs {
//	    for _, batch := range loader {
//	        g.Forward(batch.Input)
//	        g.Collect("loss")
//	    }
//	    g.Flush()  // promotes batch mean → epoch history
//	}
func (g *Graph) Flush(tags ...string) {
	if g.batchBuffer == nil {
		return
	}
	if len(tags) == 0 {
		tags = g.bufferedTags()
	}
	if g.epochHistory == nil {
		g.epochHistory = make(map[string][]float64)
	}

	flushed := make(map[string]float64, len(tags))
	for _, tag := range tags {
		vals, ok := g.batchBuffer[tag]
		if !ok || len(vals) == 0 {
			continue
		}
		mean := 0.0
		for _, v := range vals {
			mean += v
		}
		mean /= float64(len(vals))
		g.epochHistory[tag] = append(g.epochHistory[tag], mean)
		flushed[tag] = mean
		delete(g.batchBuffer, tag)
	}

	if g.flushFunc != nil && len(flushed) > 0 {
		g.flushFunc(flushed)
	}
}

// OnFlush sets a custom handler called after each Flush with the
// epoch values (tag → mean). Pass nil to remove the hook.
//
//	g.OnFlush(func(flushed map[string]float64) {
//	    fmt.Printf("epoch loss: %.4f\n", flushed["loss"])
//	})
func (g *Graph) OnFlush(fn func(flushed map[string]float64)) {
	g.flushFunc = fn
}

// Trend returns the epoch-level trend for a tag — one data point
// per Flush call. Returns an empty Trend if no data has been flushed.
//
//	trend := g.Trend("loss")
//	if trend.Stalled(5, 1e-4) {
//	    scheduler.Decay()
//	}
func (g *Graph) Trend(tag string) *Trend {
	if g.epochHistory == nil {
		return NewTrend(nil)
	}
	return NewTrend(g.epochHistory[tag])
}

// Trends returns a [TrendGroup] for the given tags, expanding any
// tag group names registered with [FlowBuilder.TagGroup].
//
//	// With TagGroup("head") → ["head_0", "head_1", "head_2"]:
//	if g.Trends("head").AllImproving(5) {
//	    fmt.Println("all heads improving")
//	}
func (g *Graph) Trends(tags ...string) TrendGroup {
	expanded := g.expandGroups(tags)
	tg := make(TrendGroup, len(expanded))
	for i, tag := range expanded {
		tg[i] = g.Trend(tag)
	}
	return tg
}

// ResetTrend clears the epoch history for the specified tags.
// If no tags are specified, all epoch history is cleared.
func (g *Graph) ResetTrend(tags ...string) {
	if g.epochHistory == nil {
		return
	}
	if len(tags) == 0 {
		g.epochHistory = nil
		return
	}
	for _, tag := range tags {
		delete(g.epochHistory, tag)
	}
}

// captureTagged stores the output of tagged nodes during Forward.
func (g *Graph) captureTagged(node *Node, outs []*autograd.Variable) {
	if tag, ok := g.tagsByNode[node.id]; ok {
		g.taggedOutputs[tag] = outs[0]
	}
}

// taggedValues returns current tagged outputs for the given tags.
// If tags is empty, returns all tagged outputs.
func (g *Graph) taggedValues(tags []string) map[string]*autograd.Variable {
	if g.taggedOutputs == nil {
		return nil
	}
	if len(tags) == 0 {
		result := make(map[string]*autograd.Variable, len(g.taggedOutputs))
		for k, v := range g.taggedOutputs {
			if v != nil {
				result[k] = v
			}
		}
		if len(result) == 0 {
			return nil
		}
		return result
	}
	result := make(map[string]*autograd.Variable, len(tags))
	for _, tag := range tags {
		if v, ok := g.taggedOutputs[tag]; ok && v != nil {
			result[tag] = v
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

// bufferedTags returns all tag names that have data in the batch buffer.
func (g *Graph) bufferedTags() []string {
	tags := make([]string, 0, len(g.batchBuffer))
	for tag := range g.batchBuffer {
		tags = append(tags, tag)
	}
	return tags
}

// scalarValue extracts a float64 scalar from a Variable.
// Returns 0 if the extraction fails.
func scalarValue(v *autograd.Variable) float64 {
	data, err := v.Data().Float64Data()
	if err != nil || len(data) == 0 {
		return 0
	}
	return data[0]
}

// defaultLog prints tagged values to stderr in a clean format.
func defaultLog(values map[string]*autograd.Variable) {
	keys := make([]string, 0, len(values))
	for k := range values {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s: %.6g", k, scalarValue(values[k])))
	}
	fmt.Fprintln(os.Stderr, strings.Join(parts, " | "))
}
