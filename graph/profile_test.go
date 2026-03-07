package graph

import (
	"strings"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

func makeProfileInput() *autograd.Variable {
	t, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{1, 4})
	return autograd.NewVariable(t, false)
}

// TestProfilingDisabledByDefault verifies no overhead when profiling is off.
func TestProfilingDisabledByDefault(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Build()
	if err != nil {
		t.Fatal(err)
	}

	g.Forward(makeProfileInput())

	if g.Profiling() {
		t.Fatal("profiling should be disabled by default")
	}
	if g.Profile() != nil {
		t.Fatal("Profile() should be nil when profiling is disabled")
	}
	if g.Timing("anything") != 0 {
		t.Fatal("Timing() should return 0 when profiling is disabled")
	}
}

// TestProfilingBasic verifies per-node and per-level timing is recorded.
func TestProfilingBasic(t *testing.T) {
	l1, _ := nn.NewLinear(4, 4)
	l2, _ := nn.NewLinear(4, 2)

	g, err := From(l1).Tag("encoder").
		Through(nn.NewGELU()).
		Through(l2).Tag("decoder").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makeProfileInput())

	p := g.Profile()
	if p == nil {
		t.Fatal("Profile() should not be nil after Forward with profiling")
	}

	// Total time should be positive.
	if p.Total <= 0 {
		t.Fatalf("Total should be positive, got %v", p.Total)
	}

	// Should have nodes and levels.
	if len(p.Nodes) == 0 {
		t.Fatal("expected at least one node timing")
	}
	if len(p.Levels) == 0 {
		t.Fatal("expected at least one level timing")
	}

	// Tagged nodes should be findable.
	if g.Timing("encoder") <= 0 {
		t.Fatal("encoder timing should be positive")
	}
	if g.Timing("decoder") <= 0 {
		t.Fatal("decoder timing should be positive")
	}
	if g.Timing("nonexistent") != 0 {
		t.Fatal("unknown tag should return 0")
	}

	// Every node should have a positive duration.
	for _, n := range p.Nodes {
		if n.Duration <= 0 {
			t.Fatalf("node %q duration should be positive, got %v", n.ID, n.Duration)
		}
	}

	// Every level should have a positive wall-clock.
	for _, l := range p.Levels {
		if l.WallClock <= 0 {
			t.Fatalf("level %d WallClock should be positive, got %v", l.Index, l.WallClock)
		}
		if l.NumNodes <= 0 {
			t.Fatalf("level %d NumNodes should be positive", l.Index)
		}
	}
}

// TestProfilingParallelism verifies that parallel levels report
// meaningful parallelism metrics.
func TestProfilingParallelism(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4), nn.MustLinear(4, 4)).
		Merge(Mean()).
		Through(nn.MustLinear(4, 2)).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makeProfileInput())

	p := g.Profile()
	if p == nil {
		t.Fatal("expected profile")
	}

	// Find the parallel level (3 nodes).
	var parallelLevel *LevelTiming
	for i := range p.Levels {
		if p.Levels[i].NumNodes == 3 {
			parallelLevel = &p.Levels[i]
			break
		}
	}
	if parallelLevel == nil {
		t.Fatal("expected a level with 3 nodes")
	}

	// SumNodes should be >= WallClock (parallelism means overlap).
	if parallelLevel.SumNodes < parallelLevel.WallClock {
		// This isn't guaranteed (scheduling jitter), but for 3 concurrent
		// nodes with non-trivial work, sum should typically exceed wall-clock.
		t.Logf("SumNodes (%v) < WallClock (%v) — might indicate no parallelism",
			parallelLevel.SumNodes, parallelLevel.WallClock)
	}

	par := parallelLevel.Parallelism()
	if par < 0.5 {
		t.Fatalf("parallelism %.2f seems too low for 3 concurrent nodes", par)
	}
}

// TestProfilingSingleNodeParallelism verifies single-node levels
// report parallelism as 1.0.
func TestProfilingSingleNodeParallelism(t *testing.T) {
	lt := &LevelTiming{NumNodes: 1, WallClock: 100, SumNodes: 100}
	if lt.Parallelism() != 1.0 {
		t.Fatalf("expected 1.0 for single node, got %f", lt.Parallelism())
	}
}

// TestProfilingDisableClears verifies that DisableProfiling clears state.
func TestProfilingDisableClears(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makeProfileInput())
	if g.Profile() == nil {
		t.Fatal("expected profile")
	}

	g.DisableProfiling()
	if g.Profile() != nil {
		t.Fatal("Profile() should be nil after DisableProfiling")
	}
}

// TestProfilingUpdatesEachForward verifies that Profile is replaced
// on each Forward call.
func TestProfilingUpdatesEachForward(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makeProfileInput())
	p1 := g.Profile()

	g.Forward(makeProfileInput())
	p2 := g.Profile()

	if p1 == p2 {
		t.Fatal("Profile should be a new object on each Forward")
	}
}

// TestProfilingHook verifies OnProfile is called after each Forward.
func TestProfilingHook(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Build()
	if err != nil {
		t.Fatal(err)
	}

	var hookCalled bool
	var hookProfile *Profile
	g.OnProfile(func(p *Profile) {
		hookCalled = true
		hookProfile = p
	})

	// Hook should not fire without profiling.
	g.Forward(makeProfileInput())
	if hookCalled {
		t.Fatal("hook should not fire when profiling is disabled")
	}

	// Hook should fire with profiling.
	g.EnableProfiling()
	g.Forward(makeProfileInput())
	if !hookCalled {
		t.Fatal("hook should fire when profiling is enabled")
	}
	if hookProfile == nil || hookProfile.Total <= 0 {
		t.Fatal("hook should receive a valid profile")
	}
}

// TestProfileString verifies the human-readable output.
func TestProfileString(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).Tag("encoder").
		Through(nn.MustLinear(4, 2)).Tag("decoder").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makeProfileInput())

	s := g.Profile().String()
	if !strings.Contains(s, "Forward:") {
		t.Fatal("String should contain 'Forward:'")
	}
	if !strings.Contains(s, "Level 0") {
		t.Fatal("String should contain 'Level 0'")
	}
	if !strings.Contains(s, `"encoder"`) {
		t.Fatal("String should contain encoder tag")
	}
	if !strings.Contains(s, `"decoder"`) {
		t.Fatal("String should contain decoder tag")
	}
}

// TestProfileStringNil verifies nil profile doesn't panic.
func TestProfileStringNil(t *testing.T) {
	var p *Profile
	if p.String() != "<no profile>" {
		t.Fatal("nil profile should return '<no profile>'")
	}
}

// TestTimingCollectFlushTrend verifies the full timing trend pipeline.
func TestTimingCollectFlushTrend(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Tag("layer").Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()

	// Simulate 3 epochs, 2 batches each.
	for epoch := 0; epoch < 3; epoch++ {
		for batch := 0; batch < 2; batch++ {
			g.Forward(makeProfileInput())
			g.CollectTimings("layer")
		}
		g.FlushTimings()
	}

	trend := g.TimingTrend("layer")
	if trend.Len() != 3 {
		t.Fatalf("expected 3 epoch entries, got %d", trend.Len())
	}

	// All timing values should be positive.
	for _, v := range trend.Values() {
		if v <= 0 {
			t.Fatalf("timing trend value should be positive, got %v", v)
		}
	}
}

// TestTimingCollectAll verifies CollectTimings with no args collects all tags.
func TestTimingCollectAll(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).Tag("a").
		Through(nn.MustLinear(4, 2)).Tag("b").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makeProfileInput())
	g.CollectTimings() // all tags
	g.FlushTimings()

	if g.TimingTrend("a").Len() != 1 {
		t.Fatal("expected 1 entry for tag 'a'")
	}
	if g.TimingTrend("b").Len() != 1 {
		t.Fatal("expected 1 entry for tag 'b'")
	}
}

// TestResetTimingTrend verifies clearing timing history.
func TestResetTimingTrend(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Tag("x").Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makeProfileInput())
	g.CollectTimings("x")
	g.FlushTimings()

	if g.TimingTrend("x").Len() != 1 {
		t.Fatal("expected 1 entry")
	}

	g.ResetTimingTrend("x")
	if g.TimingTrend("x").Len() != 0 {
		t.Fatal("expected 0 entries after reset")
	}
}

// TestResetTimingTrendAll verifies clearing all timing history.
func TestResetTimingTrendAll(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).Tag("a").
		Through(nn.MustLinear(4, 2)).Tag("b").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makeProfileInput())
	g.CollectTimings()
	g.FlushTimings()

	g.ResetTimingTrend() // clear all
	if g.TimingTrend("a").Len() != 0 || g.TimingTrend("b").Len() != 0 {
		t.Fatal("expected 0 entries after reset all")
	}
}

// TestProfilingWithLoop verifies that loop nodes are timed as a whole.
func TestProfilingWithLoop(t *testing.T) {
	body := &ctxCounter{}
	g, err := From(body).
		Loop(body).For(10).Tag("loop").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	input, _ := tensor.FromFloat32([]float32{1}, []int64{1, 1})
	g.Forward(autograd.NewVariable(input, false))

	loopTime := g.Timing("loop")
	if loopTime <= 0 {
		t.Fatal("loop timing should be positive")
	}
}
