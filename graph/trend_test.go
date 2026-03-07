package graph

import (
	"math"
	"testing"
)

func TestTrendEmpty(t *testing.T) {
	tr := NewTrend(nil)
	if tr.Len() != 0 {
		t.Errorf("Len: got %d, want 0", tr.Len())
	}
	if tr.Mean() != 0 {
		t.Error("Mean should be 0 for empty")
	}
	if tr.Min() != 0 {
		t.Error("Min should be 0 for empty")
	}
	if tr.Max() != 0 {
		t.Error("Max should be 0 for empty")
	}
	if tr.Slope(0) != 0 {
		t.Error("Slope should be 0 for empty")
	}
	if tr.Stalled(3, 0.1) {
		t.Error("Stalled should be false for empty")
	}
	if tr.Improving(3) {
		t.Error("Improving should be false for empty")
	}
	if tr.Converged(3, 0.1) {
		t.Error("Converged should be false for empty")
	}
}

func TestTrendSingleValue(t *testing.T) {
	tr := NewTrend([]float64{5.0})
	if tr.Len() != 1 {
		t.Errorf("Len: got %d, want 1", tr.Len())
	}
	if tr.Mean() != 5.0 {
		t.Errorf("Mean: got %f, want 5", tr.Mean())
	}
	if tr.Slope(0) != 0 {
		t.Error("Slope should be 0 for single value")
	}
}

func TestTrendMean(t *testing.T) {
	tr := NewTrend([]float64{2, 4, 6, 8})
	if tr.Mean() != 5.0 {
		t.Errorf("Mean: got %f, want 5", tr.Mean())
	}
}

func TestTrendMinMax(t *testing.T) {
	tr := NewTrend([]float64{3, 1, 4, 1, 5, 9})
	if tr.Min() != 1 {
		t.Errorf("Min: got %f, want 1", tr.Min())
	}
	if tr.Max() != 9 {
		t.Errorf("Max: got %f, want 9", tr.Max())
	}
}

func TestTrendLast(t *testing.T) {
	tr := NewTrend([]float64{1, 2, 3, 4, 5})

	last3 := tr.Last(3)
	if len(last3) != 3 || last3[0] != 3 || last3[1] != 4 || last3[2] != 5 {
		t.Errorf("Last(3): got %v, want [3 4 5]", last3)
	}

	// More than available → return all.
	last10 := tr.Last(10)
	if len(last10) != 5 {
		t.Errorf("Last(10): got %d values, want 5", len(last10))
	}
}

func TestTrendSlopeDecreasing(t *testing.T) {
	// Perfectly linear decrease: 5, 4, 3, 2, 1
	tr := NewTrend([]float64{5, 4, 3, 2, 1})
	slope := tr.Slope(0) // all values
	if math.Abs(slope-(-1.0)) > 1e-10 {
		t.Errorf("Slope: got %f, want -1.0", slope)
	}
}

func TestTrendSlopeIncreasing(t *testing.T) {
	// Perfectly linear increase: 1, 3, 5, 7
	tr := NewTrend([]float64{1, 3, 5, 7})
	slope := tr.Slope(0)
	if math.Abs(slope-2.0) > 1e-10 {
		t.Errorf("Slope: got %f, want 2.0", slope)
	}
}

func TestTrendSlopeWindow(t *testing.T) {
	// First half increases, last half decreases.
	tr := NewTrend([]float64{1, 2, 3, 4, 3, 2, 1})

	// Full slope ≈ 0 (symmetric).
	// Last 4: [4, 3, 2, 1] → slope = -1
	slope := tr.Slope(4)
	if math.Abs(slope-(-1.0)) > 1e-10 {
		t.Errorf("Slope(4): got %f, want -1.0", slope)
	}
}

func TestTrendStalled(t *testing.T) {
	// Flat: all the same value.
	tr := NewTrend([]float64{5, 5, 5, 5, 5})
	if !tr.Stalled(5, 0.01) {
		t.Error("should be stalled (flat)")
	}

	// Clearly moving.
	tr2 := NewTrend([]float64{10, 8, 6, 4, 2})
	if tr2.Stalled(5, 0.01) {
		t.Error("should not be stalled (decreasing)")
	}
}

func TestTrendImproving(t *testing.T) {
	// Loss going down → improving.
	tr := NewTrend([]float64{1.0, 0.8, 0.6, 0.4})
	if !tr.Improving(0) {
		t.Error("should be improving (decreasing loss)")
	}

	// Loss going up → not improving.
	tr2 := NewTrend([]float64{0.4, 0.6, 0.8, 1.0})
	if tr2.Improving(0) {
		t.Error("should not be improving (increasing)")
	}
}

func TestTrendConverged(t *testing.T) {
	// Very tight cluster → converged.
	tr := NewTrend([]float64{1.000, 1.001, 0.999, 1.000, 1.001})
	if !tr.Converged(5, 1e-4) {
		t.Error("should be converged (tight cluster)")
	}

	// Spread out → not converged.
	tr2 := NewTrend([]float64{1.0, 2.0, 1.5, 2.5, 1.0})
	if tr2.Converged(5, 1e-4) {
		t.Error("should not be converged (spread out)")
	}
}

func TestTrendImprovingWindow(t *testing.T) {
	// Overall increasing, but last 3 are decreasing.
	tr := NewTrend([]float64{0.1, 0.5, 1.0, 0.8, 0.6})
	if !tr.Improving(3) {
		t.Error("last 3 should be improving")
	}
	if tr.Improving(0) {
		t.Error("overall should not be improving")
	}
}
