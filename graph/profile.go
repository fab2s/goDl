// Profiling captures per-node and per-level execution timings during Forward.
//
// Profiling is opt-in: call [Graph.EnableProfiling] to start recording.
// When disabled (the default), profiling adds zero overhead — the bool
// gate is never true and no timing calls are made.
//
// After each Forward call with profiling enabled, [Graph.Profile] returns
// a [Profile] with per-node durations, per-level wall-clock times, and
// parallelism efficiency metrics for multi-node levels.
//
// # Timing Trends
//
// Profiling integrates with the observation layer for epoch-level analysis.
// [Graph.CollectTimings] snapshots tagged node durations into a batch buffer,
// [Graph.FlushTimings] promotes the batch mean to epoch history, and
// [Graph.TimingTrend] returns a [Trend] over the timing history:
//
//	g.EnableProfiling()
//	for epoch := range epochs {
//	    for loader.Next() {
//	        g.Forward(input)
//	        g.CollectTimings("encoder", "decoder")
//	    }
//	    g.FlushTimings()
//	    fmt.Printf("encoder: %v (slope: %.4f)\n",
//	        g.Timing("encoder"),
//	        g.TimingTrend("encoder").Slope(5))
//	}
package graph

import (
	"fmt"
	"strings"
	"time"
)

// NodeTiming records the execution time of a single node in a Forward pass.
type NodeTiming struct {
	ID       string        // internal node ID (e.g. "Linear_0")
	Tag      string        // tag name, empty if untagged
	Duration time.Duration // wall-clock time for node.run()
	Level    int           // topological level index
}

// LevelTiming records the execution time of a topological level.
// Multi-node levels execute in parallel via goroutines.
type LevelTiming struct {
	Index     int           // topological level index
	WallClock time.Duration // wall-clock time for the entire level
	SumNodes  time.Duration // sum of all node durations in this level
	NumNodes  int           // number of nodes in this level
}

// Parallelism returns the ratio of sequential node time to wall-clock
// time. Values above 1.0 indicate effective parallelism — a value of
// 2.5 means the level ran 2.5x faster than sequential execution.
// Returns 1.0 for single-node levels or when wall-clock is zero.
func (lt *LevelTiming) Parallelism() float64 {
	if lt.WallClock <= 0 || lt.NumNodes <= 1 {
		return 1.0
	}
	return float64(lt.SumNodes) / float64(lt.WallClock)
}

// Profile holds timing data from a single Forward pass.
// Obtain it via [Graph.Profile] after a forward call with profiling enabled.
type Profile struct {
	Total  time.Duration // total forward pass wall-clock time
	Levels []LevelTiming // per-level timing (in execution order)
	Nodes  []NodeTiming  // per-node timing (in execution order)
}

// Timing returns the duration of a tagged node, or zero if the tag
// is not found in this profile.
func (p *Profile) Timing(tag string) time.Duration {
	if p == nil {
		return 0
	}
	for i := range p.Nodes {
		if p.Nodes[i].Tag == tag {
			return p.Nodes[i].Duration
		}
	}
	return 0
}

// String returns a human-readable profile summary grouped by level.
//
//	Forward: 12.345ms (5 levels, 12 nodes)
//
//	  Level 0  2.100ms
//	    Linear_0 "encoder"                   2.100ms
//
//	  Level 1  5.200ms  3 nodes  ×2.4
//	    Linear_2 "headA"                     4.100ms
//	    Linear_3 "headB"                     3.800ms
//	    Linear_4 "headC"                     4.600ms
func (p *Profile) String() string {
	if p == nil {
		return "<no profile>"
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Forward: %v (%d levels, %d nodes)\n",
		p.Total.Round(time.Microsecond), len(p.Levels), len(p.Nodes))

	nodeIdx := 0
	for _, level := range p.Levels {
		fmt.Fprintf(&b, "\n  Level %d  %v", level.Index, level.WallClock.Round(time.Microsecond))
		if level.NumNodes > 1 {
			fmt.Fprintf(&b, "  %d nodes  ×%.1f", level.NumNodes, level.Parallelism())
		}
		fmt.Fprintln(&b)

		for nodeIdx < len(p.Nodes) && p.Nodes[nodeIdx].Level == level.Index {
			n := p.Nodes[nodeIdx]
			label := n.ID
			if n.Tag != "" {
				label += " " + fmt.Sprintf("%q", n.Tag)
			}
			fmt.Fprintf(&b, "    %-40s %v\n", label, n.Duration.Round(time.Microsecond))
			nodeIdx++
		}
	}

	return b.String()
}

// --- Graph profiling methods ---

// EnableProfiling turns on per-node and per-level timing for subsequent
// Forward calls. Overhead is ~20-50ns per node (two time.Now() calls).
// Call [Graph.Profile] after Forward to retrieve the results.
func (g *Graph) EnableProfiling() {
	g.profiling = true
}

// DisableProfiling turns off timing. Subsequent Forward calls have zero
// profiling overhead.
func (g *Graph) DisableProfiling() {
	g.profiling = false
	g.lastProfile = nil
}

// Profiling reports whether profiling is currently enabled.
func (g *Graph) Profiling() bool {
	return g.profiling
}

// Profile returns the timing data from the most recent Forward call,
// or nil if profiling is disabled or no Forward has been called.
func (g *Graph) Profile() *Profile {
	return g.lastProfile
}

// Timing returns the execution duration of a tagged node from the most
// recent Forward call. Returns zero if profiling is disabled or the tag
// is not found.
func (g *Graph) Timing(tag string) time.Duration {
	return g.lastProfile.Timing(tag)
}

// OnProfile sets a callback invoked after each Forward call when profiling
// is enabled. The callback receives the completed Profile. Set to nil to
// remove the hook.
//
//	g.OnProfile(func(p *graph.Profile) {
//	    fmt.Print(p)  // print the profile summary
//	})
func (g *Graph) OnProfile(fn func(*Profile)) {
	g.profileFunc = fn
}

// CollectTimings snapshots the execution duration of tagged nodes from
// the most recent Forward into a timing batch buffer. If no tags are
// specified, all tagged nodes with timing data are collected.
//
// Call [Graph.FlushTimings] at epoch boundaries to promote the batch
// mean to epoch history.
func (g *Graph) CollectTimings(tags ...string) {
	if g.lastProfile == nil {
		return
	}
	if g.timingBuffer == nil {
		g.timingBuffer = make(map[string][]float64)
	}

	if len(tags) == 0 {
		for i := range g.lastProfile.Nodes {
			n := &g.lastProfile.Nodes[i]
			if n.Tag != "" {
				g.timingBuffer[n.Tag] = append(g.timingBuffer[n.Tag], n.Duration.Seconds())
			}
		}
		return
	}

	for _, tag := range tags {
		d := g.lastProfile.Timing(tag)
		if d > 0 {
			g.timingBuffer[tag] = append(g.timingBuffer[tag], d.Seconds())
		}
	}
}

// FlushTimings computes the mean of each tag's timing batch buffer,
// appends it to the timing epoch history, and clears the buffer.
// If specific tags are given, only those are flushed; otherwise all
// buffered tags are flushed.
func (g *Graph) FlushTimings(tags ...string) {
	if g.timingBuffer == nil {
		return
	}
	if g.timingHistory == nil {
		g.timingHistory = make(map[string][]float64)
	}

	flush := func(tag string) {
		buf := g.timingBuffer[tag]
		if len(buf) == 0 {
			return
		}
		sum := 0.0
		for _, v := range buf {
			sum += v
		}
		g.timingHistory[tag] = append(g.timingHistory[tag], sum/float64(len(buf)))
		delete(g.timingBuffer, tag)
	}

	if len(tags) == 0 {
		for tag := range g.timingBuffer {
			flush(tag)
		}
		return
	}
	for _, tag := range tags {
		flush(tag)
	}
}

// TimingTrend returns an epoch-level trend over the timing history of a
// tagged node. The trend values are mean execution times in seconds,
// one per flushed epoch. Supports the same queries as value trends:
// Slope, Stalled, Improving, Converged.
//
//	if g.TimingTrend("encoder").Slope(5) > 0.001 {
//	    log.Println("encoder getting slower — possible memory issue")
//	}
func (g *Graph) TimingTrend(tag string) *Trend {
	if g.timingHistory == nil {
		return &Trend{}
	}
	return &Trend{values: g.timingHistory[tag]}
}

// TimingTrends returns a [TrendGroup] for timing trends of the given tags,
// expanding any tag group names registered with [FlowBuilder.TagGroup].
//
//	if g.TimingTrends("head").MeanSlope(5) > 0.001 {
//	    fmt.Println("heads getting slower")
//	}
func (g *Graph) TimingTrends(tags ...string) TrendGroup {
	expanded := g.expandGroups(tags)
	tg := make(TrendGroup, len(expanded))
	for i, tag := range expanded {
		tg[i] = g.TimingTrend(tag)
	}
	return tg
}

// ResetTimingTrend clears the timing epoch history. If specific tags
// are given, only those are cleared; otherwise all timing history is
// cleared.
func (g *Graph) ResetTimingTrend(tags ...string) {
	if g.timingHistory == nil {
		return
	}
	if len(tags) == 0 {
		g.timingHistory = make(map[string][]float64)
		return
	}
	for _, tag := range tags {
		delete(g.timingHistory, tag)
	}
}
