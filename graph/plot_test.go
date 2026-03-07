package graph

import (
	"os"
	"strings"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

func makePlotInput() *autograd.Variable {
	t, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{1, 4})
	return autograd.NewVariable(t, false)
}

// --- Tier 1: DOTWithProfile tests ---

// TestDOTWithProfileContainsTimings verifies timing annotations appear.
func TestDOTWithProfileContainsTimings(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).Tag("encoder").
		Through(nn.MustLinear(4, 2)).Tag("decoder").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makePlotInput())

	dot := g.DOTWithProfile()

	// Should contain "Forward:" total timing as graph label.
	if !strings.Contains(dot, "Forward:") {
		t.Fatal("DOTWithProfile should contain Forward total timing")
	}
	// Should contain level timing.
	if !strings.Contains(dot, "level 0") {
		t.Fatal("DOTWithProfile should contain level labels")
	}
	// Node durations should appear (as µs or ms).
	if !strings.Contains(dot, "µs") && !strings.Contains(dot, "ms") {
		t.Fatal("DOTWithProfile should contain duration values")
	}
}

// TestDOTWithProfileNoProfile falls back to structural DOT.
func TestDOTWithProfileNoProfile(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Build()
	if err != nil {
		t.Fatal(err)
	}

	dot := g.DOTWithProfile()
	structural := g.DOT()

	// Without profile data, both should be identical.
	if dot != structural {
		t.Fatal("DOTWithProfile without profiling should equal DOT")
	}
}

// TestDOTWithProfileHeatColors verifies color annotations are hex.
func TestDOTWithProfileHeatColors(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4), nn.MustLinear(4, 4)).
		Merge(Mean()).
		Through(nn.MustLinear(4, 2)).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makePlotInput())

	dot := g.DOTWithProfile()
	// Heatmap colors are #rrggbb — should have colors beyond the default palette.
	// The slowest node gets red-ish, fastest gets green-ish.
	if !strings.Contains(dot, "fillcolor=\"#") {
		t.Fatal("DOTWithProfile should contain heatmap fillcolors")
	}
}

// TestDOTWithProfileParallelLevel verifies parallelism annotation.
func TestDOTWithProfileParallelLevel(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4)).
		Merge(Mean()).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makePlotInput())

	dot := g.DOTWithProfile()
	// Parallel level should show ×N.N parallelism.
	if !strings.Contains(dot, "×") {
		t.Fatal("DOTWithProfile should show parallelism on multi-node levels")
	}
}

// TestSVGWithProfile verifies SVG rendering with profile.
func TestSVGWithProfile(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makePlotInput())

	svg, err := g.SVGWithProfile()
	if err != nil {
		t.Fatal(err)
	}
	if len(svg) == 0 {
		t.Fatal("SVGWithProfile should return non-empty bytes")
	}
	if !strings.Contains(string(svg), "<svg") {
		t.Fatal("SVGWithProfile output should contain <svg")
	}
}

// TestHeatColor verifies the color gradient function.
func TestHeatColor(t *testing.T) {
	green := heatColor(0.0)
	yellow := heatColor(0.5)
	red := heatColor(1.0)

	if green == red {
		t.Fatal("green and red should be different")
	}
	if green == yellow {
		t.Fatal("green and yellow should be different")
	}
	if yellow == red {
		t.Fatal("yellow and red should be different")
	}

	// All should be valid hex colors.
	for _, c := range []string{green, yellow, red} {
		if len(c) != 7 || c[0] != '#' {
			t.Fatalf("invalid hex color: %s", c)
		}
	}

	// Clamping.
	if heatColor(-1) != heatColor(0) {
		t.Fatal("negative ratio should clamp to 0")
	}
	if heatColor(2) != heatColor(1) {
		t.Fatal("ratio >1 should clamp to 1")
	}
}

// --- Tier 2: PlotHTML tests ---

// TestPlotHTMLBasic verifies HTML generation with epoch data.
func TestPlotHTMLBasic(t *testing.T) {
	g, err := From(nn.MustLinear(4, 1)).Tag("loss").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Simulate 5 epochs.
	for range 5 {
		g.Forward(makePlotInput())
		g.Collect("loss")
		g.Flush()
	}

	path := t.TempDir() + "/training.html"
	if err := g.PlotHTML(path, "loss"); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	html := string(data)

	if !strings.Contains(html, "Training Curves") {
		t.Fatal("HTML should contain title")
	}
	if !strings.Contains(html, "loss") {
		t.Fatal("HTML should contain tag name")
	}
	if !strings.Contains(html, "canvas") {
		t.Fatal("HTML should contain canvas element")
	}
}

// TestPlotHTMLNoData verifies error when no epoch data exists.
func TestPlotHTMLNoData(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Build()
	if err != nil {
		t.Fatal(err)
	}

	path := t.TempDir() + "/empty.html"
	if err := g.PlotHTML(path); err == nil {
		t.Fatal("expected error when no epoch data exists")
	}
}

// TestPlotHTMLAllTags verifies plotting all tags when none specified.
func TestPlotHTMLAllTags(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).Tag("a").
		Through(nn.MustLinear(4, 1)).Tag("b").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	for range 3 {
		g.Forward(makePlotInput())
		g.Collect("a", "b")
		g.Flush()
	}

	path := t.TempDir() + "/all.html"
	if err := g.PlotHTML(path); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(path)
	html := string(data)
	if !strings.Contains(html, `"a"`) || !strings.Contains(html, `"b"`) {
		t.Fatal("HTML should contain both tag names")
	}
}

// TestPlotHTMLGroupExpansion verifies TagGroup expansion in PlotHTML.
func TestPlotHTMLGroupExpansion(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 1), nn.MustLinear(4, 1)).TagGroup("head").
		Merge(Mean()).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	for range 3 {
		g.Forward(makePlotInput())
		g.Collect("head_0", "head_1")
		g.Flush()
	}

	path := t.TempDir() + "/heads.html"
	if err := g.PlotHTML(path, "head"); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(path)
	html := string(data)
	if !strings.Contains(html, "head_0") || !strings.Contains(html, "head_1") {
		t.Fatal("HTML should contain expanded group tags")
	}
}

// TestPlotTimingsHTML verifies timing curves HTML generation.
func TestPlotTimingsHTML(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Tag("layer").Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	for range 3 {
		g.Forward(makePlotInput())
		g.CollectTimings("layer")
		g.FlushTimings()
	}

	path := t.TempDir() + "/timings.html"
	if err := g.PlotTimingsHTML(path, "layer"); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(path)
	html := string(data)
	if !strings.Contains(html, "Timing Trends") {
		t.Fatal("HTML should contain timing title")
	}
}

// --- Tier 2: ExportTrends tests ---

// TestExportTrendsCSV verifies CSV output format.
func TestExportTrendsCSV(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).Tag("a").
		Through(nn.MustLinear(4, 1)).Tag("b").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	for range 3 {
		g.Forward(makePlotInput())
		g.Collect("a", "b")
		g.Flush()
	}

	path := t.TempDir() + "/metrics.csv"
	if err := g.ExportTrends(path, "a", "b"); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	csv := string(data)

	// Header.
	lines := strings.Split(strings.TrimSpace(csv), "\n")
	if len(lines) != 4 { // header + 3 data rows
		t.Fatalf("expected 4 lines (header + 3 epochs), got %d", len(lines))
	}
	if !strings.HasPrefix(lines[0], "epoch,") {
		t.Fatal("CSV should start with epoch header")
	}
	if !strings.Contains(lines[0], "a") || !strings.Contains(lines[0], "b") {
		t.Fatal("CSV header should contain tag names")
	}
	// First data row should start with "1".
	if !strings.HasPrefix(lines[1], "1,") {
		t.Fatalf("first data row should start with '1,', got: %s", lines[1])
	}
}

// TestExportTrendsNoData verifies error when no data exists.
func TestExportTrendsNoData(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Build()
	if err != nil {
		t.Fatal(err)
	}

	path := t.TempDir() + "/empty.csv"
	if err := g.ExportTrends(path); err == nil {
		t.Fatal("expected error when no epoch data exists")
	}
}

// TestExportTimingTrends verifies timing CSV export.
func TestExportTimingTrends(t *testing.T) {
	g, err := From(nn.MustLinear(4, 2)).Tag("x").Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	for range 3 {
		g.Forward(makePlotInput())
		g.CollectTimings("x")
		g.FlushTimings()
	}

	path := t.TempDir() + "/timing.csv"
	if err := g.ExportTimingTrends(path, "x"); err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(path)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 4 {
		t.Fatalf("expected 4 lines, got %d", len(lines))
	}
}
