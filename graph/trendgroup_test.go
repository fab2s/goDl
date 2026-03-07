package graph

import (
	"math"
	"testing"

	"github.com/fab2s/goDl/nn"
)

// TestTrendGroupAllImproving verifies AllImproving on a group.
func TestTrendGroupAllImproving(t *testing.T) {
	tg := TrendGroup{
		NewTrend([]float64{5, 4, 3, 2, 1}), // improving
		NewTrend([]float64{5, 4, 3, 2, 1}), // improving
	}
	if !tg.AllImproving(5) {
		t.Fatal("expected AllImproving to be true")
	}

	tg = append(tg, NewTrend([]float64{1, 2, 3, 4, 5})) // not improving
	if tg.AllImproving(5) {
		t.Fatal("expected AllImproving to be false with one non-improving")
	}
}

// TestTrendGroupAnyImproving verifies AnyImproving on a group.
func TestTrendGroupAnyImproving(t *testing.T) {
	tg := TrendGroup{
		NewTrend([]float64{1, 2, 3}), // not improving
		NewTrend([]float64{3, 2, 1}), // improving
	}
	if !tg.AnyImproving(3) {
		t.Fatal("expected AnyImproving to be true")
	}

	tg = TrendGroup{
		NewTrend([]float64{1, 2, 3}),
		NewTrend([]float64{1, 2, 3}),
	}
	if tg.AnyImproving(3) {
		t.Fatal("expected AnyImproving to be false when none improving")
	}
}

// TestTrendGroupAllStalled verifies AllStalled on a group.
func TestTrendGroupAllStalled(t *testing.T) {
	tg := TrendGroup{
		NewTrend([]float64{1.0, 1.0, 1.0}),
		NewTrend([]float64{2.0, 2.0, 2.0}),
	}
	if !tg.AllStalled(3, 0.01) {
		t.Fatal("expected AllStalled to be true for flat series")
	}
}

// TestTrendGroupAnyStalled verifies AnyStalled on a group.
func TestTrendGroupAnyStalled(t *testing.T) {
	tg := TrendGroup{
		NewTrend([]float64{5, 4, 3}),       // not stalled
		NewTrend([]float64{1.0, 1.0, 1.0}), // stalled
	}
	if !tg.AnyStalled(3, 0.01) {
		t.Fatal("expected AnyStalled to be true")
	}
}

// TestTrendGroupAllConverged verifies AllConverged on a group.
func TestTrendGroupAllConverged(t *testing.T) {
	tg := TrendGroup{
		NewTrend([]float64{1.0, 1.0, 1.0}),
		NewTrend([]float64{2.0, 2.0, 2.0}),
	}
	if !tg.AllConverged(3, 1e-8) {
		t.Fatal("expected AllConverged for constant series")
	}
}

// TestTrendGroupAnyConverged verifies AnyConverged on a group.
func TestTrendGroupAnyConverged(t *testing.T) {
	tg := TrendGroup{
		NewTrend([]float64{1, 5, 10}),      // not converged
		NewTrend([]float64{2.0, 2.0, 2.0}), // converged
	}
	if !tg.AnyConverged(3, 1e-8) {
		t.Fatal("expected AnyConverged to be true")
	}
}

// TestTrendGroupMeanSlope verifies MeanSlope across a group.
func TestTrendGroupMeanSlope(t *testing.T) {
	tg := TrendGroup{
		NewTrend([]float64{0, 2, 4}), // slope = 2
		NewTrend([]float64{0, 1, 2}), // slope = 1
	}
	mean := tg.MeanSlope(0)
	if math.Abs(mean-1.5) > 1e-10 {
		t.Fatalf("expected MeanSlope 1.5, got %f", mean)
	}
}

// TestTrendGroupSlopes verifies Slopes returns per-trend slopes.
func TestTrendGroupSlopes(t *testing.T) {
	tg := TrendGroup{
		NewTrend([]float64{0, 2, 4}), // slope = 2
		NewTrend([]float64{0, 1, 2}), // slope = 1
	}
	slopes := tg.Slopes(0)
	if len(slopes) != 2 {
		t.Fatalf("expected 2 slopes, got %d", len(slopes))
	}
	if math.Abs(slopes[0]-2.0) > 1e-10 {
		t.Fatalf("slope[0] expected 2.0, got %f", slopes[0])
	}
	if math.Abs(slopes[1]-1.0) > 1e-10 {
		t.Fatalf("slope[1] expected 1.0, got %f", slopes[1])
	}
}

// TestTrendGroupEmpty verifies safe behavior for empty groups.
func TestTrendGroupEmpty(t *testing.T) {
	var tg TrendGroup
	if tg.AllImproving(5) {
		t.Fatal("empty group should not be AllImproving")
	}
	if tg.AnyImproving(5) {
		t.Fatal("empty group should not be AnyImproving")
	}
	if tg.AllStalled(5, 0.01) {
		t.Fatal("empty group should not be AllStalled")
	}
	if tg.AnyStalled(5, 0.01) {
		t.Fatal("empty group should not be AnyStalled")
	}
	if tg.AllConverged(5, 0.01) {
		t.Fatal("empty group should not be AllConverged")
	}
	if tg.AnyConverged(5, 0.01) {
		t.Fatal("empty group should not be AnyConverged")
	}
	if tg.MeanSlope(5) != 0 {
		t.Fatal("empty group MeanSlope should be 0")
	}
	if len(tg.Slopes(5)) != 0 {
		t.Fatal("empty group Slopes should be empty")
	}
}

// TestTrendsExpandsGroup verifies g.Trends expands tag groups.
func TestTrendsExpandsGroup(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 1), nn.MustLinear(4, 1)).TagGroup("head").
		Merge(Mean()).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	// Collect data for 5 epochs with decreasing values.
	for epoch := 0; epoch < 5; epoch++ {
		g.Forward(makeTagGroupInput())
		g.Collect("head_0", "head_1")
		g.Flush()
	}

	trends := g.Trends("head")
	if len(trends) != 2 {
		t.Fatalf("expected 2 trends, got %d", len(trends))
	}
	for i, tr := range trends {
		if tr.Len() != 5 {
			t.Fatalf("trend %d: expected 5 epochs, got %d", i, tr.Len())
		}
	}
}

// TestTrendsMixedGroupAndTag verifies mixed group + individual tag expansion.
func TestTrendsMixedGroupAndTag(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).Tag("encoder").
		Split(nn.MustLinear(4, 1), nn.MustLinear(4, 1)).TagGroup("head").
		Merge(Mean()).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	for epoch := 0; epoch < 3; epoch++ {
		g.Forward(makeTagGroupInput())
		g.Collect("encoder", "head_0", "head_1")
		g.Flush()
	}

	// "head" expands to head_0, head_1; "encoder" stays as-is.
	trends := g.Trends("head", "encoder")
	if len(trends) != 3 {
		t.Fatalf("expected 3 trends (2 heads + encoder), got %d", len(trends))
	}
}

// TestTimingTrendsExpandsGroup verifies g.TimingTrends expands tag groups.
func TestTimingTrendsExpandsGroup(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4)).TagGroup("head").
		Merge(Mean()).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	for epoch := 0; epoch < 3; epoch++ {
		g.Forward(makeTagGroupInput())
		g.CollectTimings("head_0", "head_1")
		g.FlushTimings()
	}

	trends := g.TimingTrends("head")
	if len(trends) != 2 {
		t.Fatalf("expected 2 timing trends, got %d", len(trends))
	}
	for i, tr := range trends {
		if tr.Len() != 3 {
			t.Fatalf("timing trend %d: expected 3 epochs, got %d", i, tr.Len())
		}
	}
}
