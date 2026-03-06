package graph

import (
	"math"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// setLinearWeights replaces a Linear layer's weights and biases with known values.
func setLinearWeights(l *nn.Linear, wData, bData []float32) {
	w, _ := tensor.FromFloat32(wData, []int64{l.Weight.Data().Shape()[0], l.Weight.Data().Shape()[1]})
	l.Weight.SetData(w)
	b, _ := tensor.FromFloat32(bData, l.Bias.Data().Shape())
	l.Bias.SetData(b)
}

// f32 extracts the first float32 value from a variable's data.
func f32(v *autograd.Variable) float32 {
	data, _ := v.Data().Float32Data()
	return data[0]
}

// allF32 extracts all float32 values from a variable's data.
func allF32(v *autograd.Variable) []float32 {
	data, _ := v.Data().Float32Data()
	return data
}

// gradF32 extracts all float32 values from a variable's gradient.
func gradF32(v *autograd.Variable) []float32 {
	data, _ := v.Grad().Float32Data()
	return data
}

// approxEqual checks if two float32 values are close enough.
func approxEqual(a, b float32, eps float64) bool {
	return math.Abs(float64(a-b)) < eps
}

// --- Tests ---

func TestChainForward(t *testing.T) {
	// Build: Linear(2→3) → ReLU → Linear(3→1)
	l1, err := nn.NewLinear(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	l2, err := nn.NewLinear(3, 1)
	if err != nil {
		t.Fatal(err)
	}

	// Set known weights.
	// l1: W=[3,2], b=[3]
	setLinearWeights(l1,
		[]float32{1, 0, 0, 1, 1, 1}, // W: identity + sum row
		[]float32{0, 0, 0},           // b: zeros
	)
	// l2: W=[1,3], b=[1]
	setLinearWeights(l2,
		[]float32{1, 1, 1}, // W: sum all
		[]float32{0},       // b: zero
	)

	// Build graph with fluent API.
	g, err := From(l1).Through(nn.NewReLU()).Through(l2).Build()
	if err != nil {
		t.Fatal(err)
	}

	// Input: [1, 2]
	x, err := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	input := autograd.NewVariable(x, false)

	// Graph forward.
	graphResult := g.Forward(input)
	if err := graphResult.Err(); err != nil {
		t.Fatal(err)
	}

	// Direct computation: l1([1,2]) = [1, 2, 3], relu = [1, 2, 3], l2 = [6]
	directResult := l2.Forward(nn.NewReLU().Forward(l1.Forward(input)))
	if err := directResult.Err(); err != nil {
		t.Fatal(err)
	}

	graphVal := f32(graphResult)
	directVal := f32(directResult)
	if !approxEqual(graphVal, directVal, 1e-6) {
		t.Errorf("graph=%v, direct=%v", graphVal, directVal)
	}
	t.Logf("Chain forward: graph=%.4f, direct=%.4f", graphVal, directVal)
}

func TestChainBackward(t *testing.T) {
	// Two separate but identically-initialized chains:
	// one through the graph, one direct.
	l1g, _ := nn.NewLinear(2, 2)
	l2g, _ := nn.NewLinear(2, 1)
	l1d, _ := nn.NewLinear(2, 2)
	l2d, _ := nn.NewLinear(2, 1)

	w := []float32{1, 0, 0, 1}
	b := []float32{0, 0}
	setLinearWeights(l1g, w, b)
	setLinearWeights(l1d, w, b)

	w2 := []float32{1, 1}
	b2 := []float32{0}
	setLinearWeights(l2g, w2, b2)
	setLinearWeights(l2d, w2, b2)

	g, err := From(l1g).Through(nn.NewReLU()).Through(l2g).Build()
	if err != nil {
		t.Fatal(err)
	}

	// Forward + backward through graph.
	x1, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	gOut := g.Forward(autograd.NewVariable(x1, false))
	if err := gOut.Backward(); err != nil {
		t.Fatal("graph backward:", err)
	}

	// Forward + backward direct.
	x2, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	dOut := l2d.Forward(nn.NewReLU().Forward(l1d.Forward(autograd.NewVariable(x2, false))))
	if err := dOut.Backward(); err != nil {
		t.Fatal("direct backward:", err)
	}

	// Compare gradients on l1 weight.
	gGrad := gradF32(l1g.Weight.Variable)
	dGrad := gradF32(l1d.Weight.Variable)
	for i := range gGrad {
		if !approxEqual(gGrad[i], dGrad[i], 1e-5) {
			t.Errorf("l1.Weight grad[%d]: graph=%v, direct=%v", i, gGrad[i], dGrad[i])
		}
	}

	// Compare gradients on l2 weight.
	gGrad2 := gradF32(l2g.Weight.Variable)
	dGrad2 := gradF32(l2d.Weight.Variable)
	for i := range gGrad2 {
		if !approxEqual(gGrad2[i], dGrad2[i], 1e-5) {
			t.Errorf("l2.Weight grad[%d]: graph=%v, direct=%v", i, gGrad2[i], dGrad2[i])
		}
	}
	t.Log("Chain backward: gradients match")
}

func TestGraphAsModule(t *testing.T) {
	// Inner graph: Linear(2→2) → ReLU
	innerL, _ := nn.NewLinear(2, 2)
	setLinearWeights(innerL, []float32{1, 0, 0, 1}, []float32{0, 0})

	inner, err := From(innerL).Through(nn.NewReLU()).Build()
	if err != nil {
		t.Fatal(err)
	}

	// Outer graph: [inner graph] → Linear(2→1)
	outerL, _ := nn.NewLinear(2, 1)
	setLinearWeights(outerL, []float32{1, 1}, []float32{0})

	outer, err := From(inner).Through(outerL).Build()
	if err != nil {
		t.Fatal(err)
	}

	// Forward.
	x, _ := tensor.FromFloat32([]float32{3, -1}, []int64{1, 2})
	result := outer.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// Expected: inner = relu(identity([3,-1])) = [3, 0], outer = sum = 3
	val := f32(result)
	if !approxEqual(val, 3.0, 1e-6) {
		t.Errorf("nested graph: got %v, want 3.0", val)
	}

	// Parameters: inner has 2 (w+b), outer has 2 (w+b) = 4 total.
	params := outer.Parameters()
	if len(params) != 4 {
		t.Errorf("expected 4 parameters, got %d", len(params))
	}
	t.Logf("Graph-as-Module: value=%.4f, params=%d", val, len(params))
}

func TestAlso(t *testing.T) {
	// Build: Linear(2→2) → Also(Linear(2→2))
	// Result should be: input + transform(input)
	l1, _ := nn.NewLinear(2, 2)
	setLinearWeights(l1, []float32{1, 0, 0, 1}, []float32{0, 0}) // identity

	transform, _ := nn.NewLinear(2, 2)
	setLinearWeights(transform, []float32{2, 0, 0, 2}, []float32{0, 0}) // 2x

	g, err := From(l1).Also(transform).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 3}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// l1([1,3]) = [1,3] (identity)
	// transform([1,3]) = [2,6]
	// Also: [1,3] + [2,6] = [3, 9]
	vals := allF32(result)
	if !approxEqual(vals[0], 3.0, 1e-6) || !approxEqual(vals[1], 9.0, 1e-6) {
		t.Errorf("Also: got %v, want [3, 9]", vals)
	}
	t.Logf("Also (residual): %v", vals)
}

func TestSplitMerge(t *testing.T) {
	// Split into two Linear branches, merge with Add.
	branchA, _ := nn.NewLinear(2, 2)
	setLinearWeights(branchA, []float32{1, 0, 0, 1}, []float32{1, 1}) // identity + 1

	branchB, _ := nn.NewLinear(2, 2)
	setLinearWeights(branchB, []float32{2, 0, 0, 2}, []float32{0, 0}) // 2x

	// Need a starting node to split from.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0}) // identity

	g, err := From(entry).Split(branchA, branchB).Merge(Add()).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// entry([1,2]) = [1,2]
	// branchA([1,2]) = [1+1, 2+1] = [2, 3]
	// branchB([1,2]) = [2, 4]
	// Add: [2+2, 3+4] = [4, 7]
	vals := allF32(result)
	if !approxEqual(vals[0], 4.0, 1e-6) || !approxEqual(vals[1], 7.0, 1e-6) {
		t.Errorf("Split+Merge: got %v, want [4, 7]", vals)
	}
	t.Logf("Split+Merge: %v", vals)
}

func TestSplitMergeBackward(t *testing.T) {
	// Verify gradients flow through both branches.
	branchA, _ := nn.NewLinear(2, 1)
	setLinearWeights(branchA, []float32{1, 0}, []float32{0})

	branchB, _ := nn.NewLinear(2, 1)
	setLinearWeights(branchB, []float32{0, 1}, []float32{0})

	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, err := From(entry).Split(branchA, branchB).Merge(Add()).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{3, 5}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	// branchA: 3*1+5*0 = 3, branchB: 3*0+5*1 = 5, Add: 8
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}
	if !approxEqual(f32(result), 8.0, 1e-6) {
		t.Errorf("forward: got %v, want 8", f32(result))
	}

	if err := result.Backward(); err != nil {
		t.Fatal(err)
	}

	// Both branches should have gradients.
	aGrad := gradF32(branchA.Weight.Variable)
	bGrad := gradF32(branchB.Weight.Variable)
	t.Logf("branchA.Weight grad: %v", aGrad)
	t.Logf("branchB.Weight grad: %v", bGrad)

	if aGrad == nil || bGrad == nil {
		t.Error("expected gradients on both branches")
	}
}

func TestParameters(t *testing.T) {
	l1, _ := nn.NewLinear(2, 3)
	l2, _ := nn.NewLinear(3, 1)

	g, err := From(l1).Through(nn.NewReLU()).Through(l2).Build()
	if err != nil {
		t.Fatal(err)
	}

	params := g.Parameters()
	// l1: weight + bias = 2, relu: 0, l2: weight + bias = 2 → total 4
	if len(params) != 4 {
		t.Errorf("expected 4 parameters, got %d", len(params))
	}
	t.Logf("Parameters: %d total", len(params))
}

func TestErrorWrongInputCount(t *testing.T) {
	l, _ := nn.NewLinear(2, 1)
	g, err := From(l).Build()
	if err != nil {
		t.Fatal(err)
	}

	// Pass 2 inputs when graph expects 1.
	x1, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	x2, _ := tensor.FromFloat32([]float32{3, 4}, []int64{1, 2})
	result := g.Forward(
		autograd.NewVariable(x1, false),
		autograd.NewVariable(x2, false),
	)

	if result.Err() == nil {
		t.Error("expected error for wrong input count")
	}
	t.Logf("Error: %v", result.Err())
}

func TestBuildOpenStreamsError(t *testing.T) {
	l1, _ := nn.NewLinear(2, 2)
	l2, _ := nn.NewLinear(2, 2)
	l3, _ := nn.NewLinear(2, 2)

	// Split without Merge → Build should fail.
	_, err := From(l1).Split(l2, l3).Build()
	if err == nil {
		t.Error("expected error when building with open streams")
	}
	t.Logf("Error: %v", err)
}

func TestParallelLevels(t *testing.T) {
	// Verify that Split branches land in the same topological level
	// and produce correct results under parallel execution.
	// Graph: entry → Split(A, B, C) → Merge(Add)
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	a, _ := nn.NewLinear(2, 2)
	setLinearWeights(a, []float32{1, 0, 0, 0}, []float32{0, 0}) // [x1, 0]

	b, _ := nn.NewLinear(2, 2)
	setLinearWeights(b, []float32{0, 0, 0, 1}, []float32{0, 0}) // [0, x2]

	c, _ := nn.NewLinear(2, 2)
	setLinearWeights(c, []float32{1, 0, 0, 1}, []float32{1, 1}) // identity + 1

	g, err := From(entry).Split(a, b, c).Merge(Add()).Build()
	if err != nil {
		t.Fatal(err)
	}

	// Verify we have a level with 3 parallel nodes.
	found3 := false
	for _, level := range g.levels {
		if len(level) == 3 {
			found3 = true
			break
		}
	}
	if !found3 {
		t.Error("expected a level with 3 parallel nodes")
		for i, level := range g.levels {
			ids := make([]string, len(level))
			for j, n := range level {
				ids[j] = n.id
			}
			t.Logf("  level %d: %v", i, ids)
		}
	}

	// Run forward many times to stress parallel execution.
	for i := range 20 {
		x, _ := tensor.FromFloat32([]float32{float32(i + 1), float32(i + 2)}, []int64{1, 2})
		result := g.Forward(autograd.NewVariable(x, false))
		if err := result.Err(); err != nil {
			t.Fatalf("iteration %d: %v", i, err)
		}
		vals := allF32(result)
		// a([x1,x2]) = [x1, 0]
		// b([x1,x2]) = [0, x2]
		// c([x1,x2]) = [x1+1, x2+1]
		// Add: [2*x1+1, 2*x2+1]
		x1 := float32(i + 1)
		x2 := float32(i + 2)
		if !approxEqual(vals[0], 2*x1+1, 1e-5) || !approxEqual(vals[1], 2*x2+1, 1e-5) {
			t.Errorf("iteration %d: got %v, want [%.0f, %.0f]", i, vals, 2*x1+1, 2*x2+1)
		}
	}
	t.Log("Parallel 3-way split: 20 iterations passed")
}

func TestParallelBackward(t *testing.T) {
	// 3-way split with backward — verify all branches get gradients.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	a, _ := nn.NewLinear(2, 1)
	setLinearWeights(a, []float32{1, 0}, []float32{0})

	b, _ := nn.NewLinear(2, 1)
	setLinearWeights(b, []float32{0, 1}, []float32{0})

	c, _ := nn.NewLinear(2, 1)
	setLinearWeights(c, []float32{1, 1}, []float32{0})

	g, err := From(entry).Split(a, b, c).Merge(Add()).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{2, 3}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	// a: 2, b: 3, c: 5, Add: 10
	if !approxEqual(f32(result), 10.0, 1e-5) {
		t.Errorf("forward: got %v, want 10", f32(result))
	}

	if err := result.Backward(); err != nil {
		t.Fatal(err)
	}

	// All three branches should have gradients.
	for name, layer := range map[string]*nn.Linear{"a": a, "b": b, "c": c} {
		grad := layer.Weight.Grad()
		if grad == nil {
			t.Errorf("branch %s: no gradient", name)
			continue
		}
		t.Logf("branch %s grad: %v", name, gradF32(layer.Weight.Variable))
	}
}

func TestParallelWithRace(t *testing.T) {
	// Build a graph with parallel branches that each do real computation.
	// Run with -race to verify no data races.
	entry, _ := nn.NewLinear(4, 4)

	branches := make([]nn.Module, 4)
	for i := range 4 {
		l, _ := nn.NewLinear(4, 1)
		branches[i] = l
	}

	g, err := From(entry).Split(branches...).Merge(Add()).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{1, 4})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// Sum to scalar for backward.
	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	t.Logf("4-way parallel with backward: output shape %v", result.Data().Shape())
}
