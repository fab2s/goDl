package graph

import (
	"math"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

func TestTagged(t *testing.T) {
	l1, _ := nn.NewLinear(2, 2)
	setLinearWeights(l1, []float32{1, 0, 0, 1}, []float32{0, 0}) // identity
	l2, _ := nn.NewLinear(2, 1)
	setLinearWeights(l2, []float32{1, 1}, []float32{0})

	g, err := From(l1).Tag("hidden").Through(l2).Tag("output").Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{3, 5}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// Tagged("hidden") should be identity([3,5]) = [3,5].
	hidden := g.Tagged("hidden")
	if hidden == nil {
		t.Fatal("Tagged(\"hidden\") returned nil")
	}
	hdata, _ := hidden.Data().Float32Data()
	if !approxEqual(hdata[0], 3, 1e-6) || !approxEqual(hdata[1], 5, 1e-6) {
		t.Errorf("hidden: got %v, want [3 5]", hdata)
	}

	// Tagged("output") should be sum → 8.
	output := g.Tagged("output")
	if output == nil {
		t.Fatal("Tagged(\"output\") returned nil")
	}
	if !approxEqual(f32(output), 8, 1e-6) {
		t.Errorf("output: got %v, want 8", f32(output))
	}

	// Unknown tag → nil.
	if g.Tagged("unknown") != nil {
		t.Error("unknown tag should return nil")
	}
}

func TestTaggedUpdatesOnForward(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	setLinearWeights(l, []float32{2}, []float32{0})

	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	// First forward: 3 * 2 = 6.
	x1, _ := tensor.FromFloat32([]float32{3}, []int64{1, 1})
	g.Forward(autograd.NewVariable(x1, false))
	v1 := scalarValue(g.Tagged("out"))

	// Second forward: 5 * 2 = 10.
	x2, _ := tensor.FromFloat32([]float32{5}, []int64{1, 1})
	g.Forward(autograd.NewVariable(x2, false))
	v2 := scalarValue(g.Tagged("out"))

	if math.Abs(v1-6) > 1e-4 {
		t.Errorf("first forward: got %f, want 6", v1)
	}
	if math.Abs(v2-10) > 1e-4 {
		t.Errorf("second forward: got %f, want 10", v2)
	}
}

func TestLog(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	setLinearWeights(l, []float32{1}, []float32{0})

	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{7}, []int64{1, 1})
	g.Forward(autograd.NewVariable(x, false))

	// Hook should receive the correct value.
	var logged map[string]*autograd.Variable
	g.OnLog(func(values map[string]*autograd.Variable) {
		logged = values
	})

	g.Log("out")
	if logged == nil {
		t.Fatal("OnLog hook was not called")
	}
	if _, ok := logged["out"]; !ok {
		t.Fatal("hook missing 'out' tag")
	}
	if math.Abs(scalarValue(logged["out"])-7) > 1e-4 {
		t.Errorf("logged value: got %f, want 7", scalarValue(logged["out"]))
	}
}

func TestLogAllTags(t *testing.T) {
	l1, _ := nn.NewLinear(2, 2)
	setLinearWeights(l1, []float32{1, 0, 0, 1}, []float32{0, 0})
	l2, _ := nn.NewLinear(2, 1)
	setLinearWeights(l2, []float32{1, 1}, []float32{0})

	g, err := From(l1).Tag("hidden").Through(l2).Tag("output").Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{3, 5}, []int64{1, 2})
	g.Forward(autograd.NewVariable(x, false))

	var logged map[string]*autograd.Variable
	g.OnLog(func(values map[string]*autograd.Variable) {
		logged = values
	})

	g.Log() // no tags → all tags
	if len(logged) != 2 {
		t.Errorf("expected 2 tags, got %d", len(logged))
	}
	if _, ok := logged["hidden"]; !ok {
		t.Error("missing 'hidden' tag")
	}
	if _, ok := logged["output"]; !ok {
		t.Error("missing 'output' tag")
	}
}

func TestCollectAndCollected(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	setLinearWeights(l, []float32{1}, []float32{0}) // identity

	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	values := []float32{3, 7, 11}
	for _, v := range values {
		x, _ := tensor.FromFloat32([]float32{v}, []int64{1, 1})
		g.Forward(autograd.NewVariable(x, false))
		g.Collect("out")
	}

	collected := g.Collected("out")
	if len(collected) != 3 {
		t.Fatalf("collected: got %d values, want 3", len(collected))
	}
	for i, want := range []float64{3, 7, 11} {
		if math.Abs(collected[i]-want) > 1e-4 {
			t.Errorf("collected[%d]: got %f, want %f", i, collected[i], want)
		}
	}

	// Unknown tag returns nil.
	if g.Collected("unknown") != nil {
		t.Error("unknown tag should return nil")
	}
}

func TestFlush(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	setLinearWeights(l, []float32{1}, []float32{0})

	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Simulate 2 epochs of 3 batches each.
	epochs := [][]float32{
		{2, 4, 6},    // mean = 4
		{10, 20, 30}, // mean = 20
	}

	for _, epoch := range epochs {
		for _, v := range epoch {
			x, _ := tensor.FromFloat32([]float32{v}, []int64{1, 1})
			g.Forward(autograd.NewVariable(x, false))
			g.Collect("out")
		}
		g.Flush("out")
	}

	// Batch buffer should be empty after flush.
	if g.Collected("out") != nil {
		t.Error("batch buffer should be empty after flush")
	}

	// Epoch history should have 2 entries.
	trend := g.Trend("out")
	if trend.Len() != 2 {
		t.Fatalf("trend length: got %d, want 2", trend.Len())
	}
	vals := trend.Values()
	if math.Abs(vals[0]-4) > 1e-4 {
		t.Errorf("epoch 0 mean: got %f, want 4", vals[0])
	}
	if math.Abs(vals[1]-20) > 1e-4 {
		t.Errorf("epoch 1 mean: got %f, want 20", vals[1])
	}
}

func TestFlushAll(t *testing.T) {
	l1, _ := nn.NewLinear(2, 2)
	setLinearWeights(l1, []float32{1, 0, 0, 1}, []float32{0, 0})
	l2, _ := nn.NewLinear(2, 1)
	setLinearWeights(l2, []float32{1, 1}, []float32{0})

	g, err := From(l1).Tag("hidden").Through(l2).Tag("output").Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{3, 5}, []int64{1, 2})
	g.Forward(autograd.NewVariable(x, false))
	g.Collect("hidden", "output")
	g.Flush() // flush all

	if g.Collected("hidden") != nil {
		t.Error("hidden batch buffer should be empty")
	}
	if g.Collected("output") != nil {
		t.Error("output batch buffer should be empty")
	}
	if g.Trend("hidden").Len() != 1 {
		t.Error("hidden epoch history should have 1 entry")
	}
	if g.Trend("output").Len() != 1 {
		t.Error("output epoch history should have 1 entry")
	}
}

func TestFlushHook(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	setLinearWeights(l, []float32{1}, []float32{0})

	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	var flushed map[string]float64
	g.OnFlush(func(f map[string]float64) {
		flushed = f
	})

	// Collect values: 2, 4, 6 → mean = 4.
	for _, v := range []float32{2, 4, 6} {
		x, _ := tensor.FromFloat32([]float32{v}, []int64{1, 1})
		g.Forward(autograd.NewVariable(x, false))
		g.Collect("out")
	}
	g.Flush()

	if flushed == nil {
		t.Fatal("OnFlush hook was not called")
	}
	if math.Abs(flushed["out"]-4) > 1e-4 {
		t.Errorf("flushed mean: got %f, want 4", flushed["out"])
	}
}

func TestTrendFromFlush(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	setLinearWeights(l, []float32{1}, []float32{0})

	g, err := From(l).Tag("loss").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Simulate 5 epochs with decreasing loss.
	epochMeans := []float64{1.0, 0.8, 0.6, 0.4, 0.2}
	for _, mean := range epochMeans {
		x, _ := tensor.FromFloat32([]float32{float32(mean)}, []int64{1, 1})
		g.Forward(autograd.NewVariable(x, false))
		g.Collect("loss")
		g.Flush("loss")
	}

	trend := g.Trend("loss")
	if trend.Len() != 5 {
		t.Fatalf("trend length: got %d, want 5", trend.Len())
	}
	if !trend.Improving(0) {
		t.Error("trend should be improving (decreasing)")
	}
	if math.Abs(trend.Slope(0)-(-0.2)) > 1e-10 {
		t.Errorf("slope: got %f, want -0.2", trend.Slope(0))
	}
	if trend.Stalled(5, 0.01) {
		t.Error("trend should not be stalled")
	}
}

func TestTrendStalledFromGraph(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	setLinearWeights(l, []float32{1}, []float32{0})

	g, err := From(l).Tag("loss").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Simulate 5 epochs with flat loss.
	for i := 0; i < 5; i++ {
		x, _ := tensor.FromFloat32([]float32{0.5}, []int64{1, 1})
		g.Forward(autograd.NewVariable(x, false))
		g.Collect("loss")
		g.Flush("loss")
	}

	trend := g.Trend("loss")
	if !trend.Stalled(5, 0.01) {
		t.Error("trend should be stalled (flat)")
	}
	if !trend.Converged(5, 1e-4) {
		t.Error("trend should be converged (identical values)")
	}
}

func TestResetTrend(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	setLinearWeights(l, []float32{1}, []float32{0})

	g, err := From(l).Tag("loss").Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1}, []int64{1, 1})
	g.Forward(autograd.NewVariable(x, false))
	g.Collect("loss")
	g.Flush("loss")

	if g.Trend("loss").Len() != 1 {
		t.Fatal("expected 1 epoch in trend")
	}

	g.ResetTrend("loss")
	if g.Trend("loss").Len() != 0 {
		t.Error("trend should be empty after reset")
	}
}

func TestResetTrendAll(t *testing.T) {
	l1, _ := nn.NewLinear(2, 2)
	setLinearWeights(l1, []float32{1, 0, 0, 1}, []float32{0, 0})
	l2, _ := nn.NewLinear(2, 1)
	setLinearWeights(l2, []float32{1, 1}, []float32{0})

	g, err := From(l1).Tag("a").Through(l2).Tag("b").Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	g.Forward(autograd.NewVariable(x, false))
	g.Collect("a", "b")
	g.Flush()

	g.ResetTrend() // reset all
	if g.Trend("a").Len() != 0 || g.Trend("b").Len() != 0 {
		t.Error("all trends should be empty after ResetTrend()")
	}
}

func TestSub(t *testing.T) {
	// Build an inner graph.
	innerL, _ := nn.NewLinear(2, 2)
	setLinearWeights(innerL, []float32{1, 0, 0, 1}, []float32{0, 0})
	inner, err := From(innerL).Tag("inner_hidden").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Build outer graph with inner as a sub-graph.
	outerL, _ := nn.NewLinear(2, 1)
	setLinearWeights(outerL, []float32{1, 1}, []float32{0})
	outer, err := From(inner).Tag("sub").Through(outerL).Tag("result").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Sub should return the inner graph.
	sub := outer.Sub("sub")
	if sub == nil {
		t.Fatal("Sub(\"sub\") returned nil")
	}
	if sub != inner {
		t.Fatal("Sub(\"sub\") returned wrong graph")
	}

	// Forward should populate both graphs' tagged outputs.
	x, _ := tensor.FromFloat32([]float32{3, 5}, []int64{1, 2})
	result := outer.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// Outer tagged value.
	if math.Abs(scalarValue(outer.Tagged("result"))-8) > 1e-4 {
		t.Errorf("outer result: got %f, want 8", scalarValue(outer.Tagged("result")))
	}

	// Inner tagged value (via Sub).
	innerHidden := outer.Sub("sub").Tagged("inner_hidden")
	if innerHidden == nil {
		t.Fatal("inner tagged value is nil")
	}
	hdata, _ := innerHidden.Data().Float32Data()
	if !approxEqual(hdata[0], 3, 1e-6) || !approxEqual(hdata[1], 5, 1e-6) {
		t.Errorf("inner hidden: got %v, want [3 5]", hdata)
	}
}

func TestSubCollect(t *testing.T) {
	// Inner graph: identity.
	innerL, _ := nn.NewLinear(1, 1)
	setLinearWeights(innerL, []float32{1}, []float32{0})
	inner, err := From(innerL).Tag("val").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Outer graph: wraps inner, then doubles.
	outerL, _ := nn.NewLinear(1, 1)
	setLinearWeights(outerL, []float32{2}, []float32{0})
	outer, err := From(inner).Tag("inner").Through(outerL).Tag("doubled").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Simulate 3 batches.
	for _, v := range []float32{3, 5, 7} {
		x, _ := tensor.FromFloat32([]float32{v}, []int64{1, 1})
		outer.Forward(autograd.NewVariable(x, false))
		outer.Collect("doubled")
		outer.Sub("inner").Collect("val")
	}

	// Outer collected: 6, 10, 14.
	outerVals := outer.Collected("doubled")
	if len(outerVals) != 3 {
		t.Fatalf("outer collected: got %d, want 3", len(outerVals))
	}
	for i, want := range []float64{6, 10, 14} {
		if math.Abs(outerVals[i]-want) > 1e-4 {
			t.Errorf("outer[%d]: got %f, want %f", i, outerVals[i], want)
		}
	}

	// Inner collected: 3, 5, 7.
	innerVals := outer.Sub("inner").Collected("val")
	if len(innerVals) != 3 {
		t.Fatalf("inner collected: got %d, want 3", len(innerVals))
	}
	for i, want := range []float64{3, 5, 7} {
		if math.Abs(innerVals[i]-want) > 1e-4 {
			t.Errorf("inner[%d]: got %f, want %f", i, innerVals[i], want)
		}
	}
}

func TestSubNil(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Tag exists but node is not a Graph → nil.
	if g.Sub("out") != nil {
		t.Error("Sub on non-graph node should return nil")
	}

	// Unknown tag → nil.
	if g.Sub("unknown") != nil {
		t.Error("Sub on unknown tag should return nil")
	}
}

func TestCollectUnknownTag(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1}, []int64{1, 1})
	g.Forward(autograd.NewVariable(x, false))

	// Collecting an unknown tag should be a no-op (no panic).
	g.Collect("nonexistent")
	if g.Collected("nonexistent") != nil {
		t.Error("unknown tag should not produce collected data")
	}
}

func TestRecord(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Record external values (no Forward needed).
	g.Record("ext_loss", 0.5)
	g.Record("ext_loss", 0.3)
	g.Record("ext_loss", 0.2)

	collected := g.Collected("ext_loss")
	if len(collected) != 3 {
		t.Fatalf("collected: got %d values, want 3", len(collected))
	}
	for i, want := range []float64{0.5, 0.3, 0.2} {
		if math.Abs(collected[i]-want) > 1e-10 {
			t.Errorf("collected[%d]: got %f, want %f", i, collected[i], want)
		}
	}

	// Flush should promote to epoch history.
	g.Flush()
	trend := g.Trend("ext_loss")
	if trend.Len() != 1 {
		t.Fatalf("trend length: got %d, want 1", trend.Len())
	}
	// Mean of 0.5, 0.3, 0.2 = 1.0/3 ≈ 0.3333
	if math.Abs(trend.Values()[0]-1.0/3) > 1e-10 {
		t.Errorf("epoch mean: got %f, want %f", trend.Values()[0], 1.0/3)
	}
}

func TestRecordVariadic(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Record multiple values in one call.
	g.Record("metric", 1.0, 2.0, 3.0)
	collected := g.Collected("metric")
	if len(collected) != 3 {
		t.Fatalf("collected: got %d, want 3", len(collected))
	}

	// Record with no values is a no-op.
	g.Record("empty")
	if g.Collected("empty") != nil {
		t.Error("empty Record should not create buffer entry")
	}
}

func TestRecordAndCollectMix(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	setLinearWeights(l, []float32{1}, []float32{0})

	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Mix Collect (from graph) and Record (external) in same epoch.
	x, _ := tensor.FromFloat32([]float32{4}, []int64{1, 1})
	g.Forward(autograd.NewVariable(x, false))
	g.Collect("out")       // graph value: 4
	g.Record("out", 6)     // external value: 6

	g.Flush()
	trend := g.Trend("out")
	if trend.Len() != 1 {
		t.Fatalf("trend length: got %d, want 1", trend.Len())
	}
	// Mean of 4 and 6 = 5.
	if math.Abs(trend.Values()[0]-5) > 1e-4 {
		t.Errorf("epoch mean: got %f, want 5", trend.Values()[0])
	}
}

func TestRecordFlushTrend(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Simulate 3 epochs of purely external metrics.
	epochs := [][]float64{
		{1.0, 0.8, 0.6},   // mean = 0.8
		{0.5, 0.4, 0.3},   // mean = 0.4
		{0.2, 0.15, 0.1},  // mean = 0.15
	}

	for _, epoch := range epochs {
		for _, v := range epoch {
			g.Record("loss", v)
		}
		g.Flush()
	}

	trend := g.Trend("loss")
	if trend.Len() != 3 {
		t.Fatalf("trend length: got %d, want 3", trend.Len())
	}
	if !trend.Improving(0) {
		t.Error("trend should be improving")
	}
	if math.Abs(trend.Values()[0]-0.8) > 1e-10 {
		t.Errorf("epoch 0: got %f, want 0.8", trend.Values()[0])
	}
	if math.Abs(trend.Values()[1]-0.4) > 1e-10 {
		t.Errorf("epoch 1: got %f, want 0.4", trend.Values()[1])
	}
	if math.Abs(trend.Values()[2]-0.15) > 1e-10 {
		t.Errorf("epoch 2: got %f, want 0.15", trend.Values()[2])
	}
}

func TestFlushEmpty(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Flushing with nothing collected should be a no-op (no panic).
	g.Flush()
	if g.Trend("out").Len() != 0 {
		t.Error("flush with no data should not create trend entries")
	}
}

func TestEndToEndTrainingPattern(t *testing.T) {
	// Full training pattern: 3 epochs, each with 4 batches of decreasing loss.
	l, _ := nn.NewLinear(1, 1)
	setLinearWeights(l, []float32{1}, []float32{0})

	g, err := From(l).Tag("loss").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Simulated epoch means: 0.8, 0.4, 0.2
	batchValues := [][]float32{
		{0.9, 0.8, 0.8, 0.7}, // mean ≈ 0.8
		{0.5, 0.4, 0.4, 0.3}, // mean ≈ 0.4
		{0.3, 0.2, 0.2, 0.1}, // mean ≈ 0.2
	}

	for _, epoch := range batchValues {
		for _, v := range epoch {
			x, _ := tensor.FromFloat32([]float32{v}, []int64{1, 1})
			g.Forward(autograd.NewVariable(x, false))
			g.Collect("loss")
		}
		g.Flush("loss")
	}

	trend := g.Trend("loss")
	if trend.Len() != 3 {
		t.Fatalf("expected 3 epochs, got %d", trend.Len())
	}
	if !trend.Improving(0) {
		t.Error("loss should be improving")
	}
	if trend.Stalled(3, 0.01) {
		t.Error("loss should not be stalled")
	}

	t.Logf("Epoch means: %v", trend.Values())
	t.Logf("Slope: %.4f", trend.Slope(0))
	t.Logf("Mean: %.4f", trend.Mean())
}

func TestFlushCount(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	if g.FlushCount() != 0 {
		t.Error("flush count should be 0 before any flush")
	}

	// 3 flushes with recorded data.
	for i := range 3 {
		g.Record("loss", float64(i))
		g.Flush()
	}

	if g.FlushCount() != 3 {
		t.Errorf("flush count: got %d, want 3", g.FlushCount())
	}

	// Empty flush should not increment.
	g.Flush()
	if g.FlushCount() != 3 {
		t.Errorf("empty flush should not increment: got %d, want 3", g.FlushCount())
	}
}

func TestElapsed(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	if g.Elapsed() != 0 {
		t.Error("elapsed should be 0 before any flush")
	}

	g.Record("loss", 1.0)
	g.Flush()

	// Elapsed should be positive after flush.
	if g.Elapsed() <= 0 {
		t.Error("elapsed should be positive after flush")
	}
}

func TestLastFlushDuration(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	// No flushes → 0.
	if g.LastFlushDuration() != 0 {
		t.Error("should be 0 before any flush")
	}

	g.Record("loss", 1.0)
	g.Flush()

	// Single flush → 0 (need at least 2).
	if g.LastFlushDuration() != 0 {
		t.Error("should be 0 with single flush")
	}

	g.Record("loss", 0.5)
	g.Flush()

	// Two flushes → positive duration.
	if g.LastFlushDuration() < 0 {
		t.Error("should be non-negative with two flushes")
	}
}

func TestETA(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	// No flushes → 0.
	if g.ETA(100) != 0 {
		t.Error("ETA should be 0 before any flush")
	}

	// Single flush → 0 (need 2 for rate estimate).
	g.Record("loss", 1.0)
	g.Flush()
	if g.ETA(100) != 0 {
		t.Error("ETA should be 0 with single flush")
	}

	g.Record("loss", 0.5)
	g.Flush()

	// Two flushes with total=100 → positive ETA.
	eta := g.ETA(100)
	if eta <= 0 {
		t.Errorf("ETA should be positive, got %v", eta)
	}

	// Already done → 0.
	if g.ETA(2) != 0 {
		t.Errorf("ETA should be 0 when done, got %v", g.ETA(2))
	}
}

func TestLatestFromGraph(t *testing.T) {
	l, _ := nn.NewLinear(1, 1)
	g, err := From(l).Tag("out").Build()
	if err != nil {
		t.Fatal(err)
	}

	// Record and flush 3 epochs.
	for _, v := range []float64{1.0, 0.5, 0.2} {
		g.Record("loss", v)
		g.Flush()
	}

	if math.Abs(g.Trend("loss").Latest()-0.2) > 1e-10 {
		t.Errorf("Latest: got %f, want 0.2", g.Trend("loss").Latest())
	}
}
