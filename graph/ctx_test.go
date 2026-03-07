package graph

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// ctxCounter counts how many times Forward is called (atomic for goroutine safety).
type ctxCounter struct {
	calls atomic.Int64
}

func (m *ctxCounter) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	m.calls.Add(1)
	return inputs[0]
}

func (m *ctxCounter) Parameters() []*nn.Parameter { return nil }

// TestForwardCtx_Background verifies ForwardCtx with Background behaves
// identically to Forward.
func TestForwardCtx_Background(t *testing.T) {
	l, _ := nn.NewLinear(2, 2)
	setLinearWeights(l, identityN(2), []float32{0, 0})

	g, err := From(l).Build()
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.FromFloat32([]float32{3, 4}, []int64{1, 2})
	v := autograd.NewVariable(input, false)

	out1 := g.Forward(v)
	out2 := g.ForwardCtx(context.Background(), v)

	d1, _ := out1.Data().Float32Data()
	d2, _ := out2.Data().Float32Data()
	if d1[0] != d2[0] || d1[1] != d2[1] {
		t.Fatalf("Forward and ForwardCtx(Background) differ: %v vs %v", d1, d2)
	}
}

// TestForwardCtx_ForLoopCancel verifies that a For loop respects
// context cancellation and stops early.
func TestForwardCtx_ForLoopCancel(t *testing.T) {
	body := &ctxCounter{}
	g, err := From(body).
		Loop(body).For(1000).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	input, _ := tensor.FromFloat32([]float32{1}, []int64{1, 1})
	result := g.ForwardCtx(ctx, autograd.NewVariable(input, false))

	if result.Err() == nil {
		t.Fatal("expected context error, got nil")
	}
	if !errors.Is(result.Err(), context.Canceled) {
		t.Fatalf("expected context.Canceled, got: %v", result.Err())
	}
	// Body should NOT have run all 1000 iterations.
	if body.calls.Load() >= 1000 {
		t.Fatalf("expected early exit, but body ran %d times", body.calls.Load())
	}
}

// TestForwardCtx_WhileLoopCancel verifies that a While loop respects
// context cancellation.
func TestForwardCtx_WhileLoopCancel(t *testing.T) {
	body := &ctxCounter{}
	neverHalt := ThresholdHalt(1e9) // never halts naturally

	g, err := From(body).
		Loop(body).While(neverHalt, 1000).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	input, _ := tensor.FromFloat32([]float32{1}, []int64{1, 1})
	result := g.ForwardCtx(ctx, autograd.NewVariable(input, false))

	if result.Err() == nil {
		t.Fatal("expected context error")
	}
	if !errors.Is(result.Err(), context.Canceled) {
		t.Fatalf("expected context.Canceled, got: %v", result.Err())
	}
}

// TestForwardCtx_UntilLoopCancel verifies that an Until loop respects
// context cancellation but still runs the body at least once (the
// Until guarantee). Cancellation happens during the loop, not before.
func TestForwardCtx_UntilLoopCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Body cancels ctx after 2 calls (1 from From node + 1 first loop iteration).
	body := &cancelAfterModule{limit: 2, cancel: cancel}
	neverHalt := ThresholdHalt(1e9)

	g, err := From(body).
		Loop(body).Until(neverHalt, 1000).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.FromFloat32([]float32{1}, []int64{1, 1})
	result := g.ForwardCtx(ctx, autograd.NewVariable(input, false))

	// Until guarantees at least one body execution in the loop.
	// From node = 1 call, loop body >= 1 call = at least 2.
	if body.calls.Load() < 2 {
		t.Fatalf("Until should run body at least once in loop, got %d total calls", body.calls.Load())
	}

	// But should not run all 1000 loop iterations.
	if result.Err() == nil {
		t.Fatal("expected context error")
	}
	if !errors.Is(result.Err(), context.Canceled) {
		t.Fatalf("expected context.Canceled, got: %v", result.Err())
	}
}

// TestForwardCtx_MapCancel verifies that Map.Each respects context
// cancellation and stops iterating early.
func TestForwardCtx_MapCancel(t *testing.T) {
	body := &ctxCounter{}

	g, err := From(body).
		Map(body).Each().
		Build()
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// 100 elements along dim 0.
	input, _ := tensor.FromFloat32(make([]float32, 100), []int64{100, 1})
	result := g.ForwardCtx(ctx, autograd.NewVariable(input, false))

	if result.Err() == nil {
		t.Fatal("expected context error")
	}
	if !errors.Is(result.Err(), context.Canceled) {
		t.Fatalf("expected context.Canceled, got: %v", result.Err())
	}
}

// TestForwardCtx_BetweenLevels verifies that context is checked between
// topological levels.
func TestForwardCtx_BetweenLevels(t *testing.T) {
	// Two-level graph: layer1 -> layer2
	layer1 := &ctxCounter{}
	layer2 := &ctxCounter{}

	g, err := From(layer1).
		Through(layer2).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	input, _ := tensor.FromFloat32([]float32{1}, []int64{1, 1})
	result := g.ForwardCtx(ctx, autograd.NewVariable(input, false))

	if result.Err() == nil {
		t.Fatal("expected context error between levels")
	}
	if !errors.Is(result.Err(), context.Canceled) {
		t.Fatalf("expected context.Canceled, got: %v", result.Err())
	}
}

// TestForwardCtx_NormalCompletion verifies that a non-cancelled context
// lets the graph complete normally.
func TestForwardCtx_NormalCompletion(t *testing.T) {
	body := &ctxCounter{}
	g, err := From(body).
		Loop(body).For(5).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.FromFloat32([]float32{42}, []int64{1, 1})
	result := g.ForwardCtx(context.Background(), autograd.NewVariable(input, false))

	if err := result.Err(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	data, _ := result.Data().Float32Data()
	if data[0] != 42 {
		t.Fatalf("expected 42, got %v", data[0])
	}
	// body used in From + Loop, but countingModule is shared.
	// From node runs once, Loop body runs 5 times = 6 total.
	if body.calls.Load() != 6 {
		t.Fatalf("expected 6 calls, got %d", body.calls.Load())
	}
}

// cancelAfterModule cancels a context after n Forward calls.
type cancelAfterModule struct {
	calls  atomic.Int64
	limit  int64
	cancel context.CancelFunc
}

func (m *cancelAfterModule) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	c := m.calls.Add(1)
	if c >= m.limit {
		m.cancel()
	}
	return inputs[0]
}

func (m *cancelAfterModule) Parameters() []*nn.Parameter { return nil }

// TestForwardCtx_LoopPartialCancel verifies that a For loop can run
// some iterations before cancellation.
func TestForwardCtx_LoopPartialCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	body := &cancelAfterModule{limit: 5, cancel: cancel}

	g, err := From(body).
		Loop(body).For(100).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.FromFloat32([]float32{1}, []int64{1, 1})
	result := g.ForwardCtx(ctx, autograd.NewVariable(input, false))

	if result.Err() == nil {
		t.Fatal("expected context error")
	}
	// Should have run ~5 iterations, not all 100.
	totalCalls := body.calls.Load()
	if totalCalls >= 100 {
		t.Fatalf("expected early exit, got %d calls", totalCalls)
	}
}
