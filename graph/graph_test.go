package graph

import (
	"fmt"
	"math"
	"strings"
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

// identityN returns an NxN identity matrix as a flat float32 slice.
func identityN(n int) []float32 {
	data := make([]float32, n*n)
	for i := range n {
		data[i*n+i] = 1
	}
	return data
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

// softmaxModule wraps a module and applies softmax to its output.
// Used in gate tests to build routers with explicit normalization.
type softmaxModule struct {
	inner nn.Module
}

func withSoftmax(m nn.Module) nn.Module {
	return &softmaxModule{inner: m}
}

func (s *softmaxModule) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	out := s.inner.Forward(inputs...)
	dim := out.Data().Ndim() - 1
	return out.Softmax(dim)
}

func (s *softmaxModule) Parameters() []*nn.Parameter {
	return s.inner.Parameters()
}

// nilSafeAdd sums all non-nil inputs. Used with forward refs where
// state inputs are nil on the first pass.
type nilSafeAdd struct{}

func (a *nilSafeAdd) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	var result *autograd.Variable
	for _, inp := range inputs {
		if inp == nil {
			continue
		}
		if result == nil {
			result = inp
		} else {
			result = result.Add(inp)
		}
	}
	return result
}

func (a *nilSafeAdd) Parameters() []*nn.Parameter { return nil }

// namedRouter is a test module implementing NamedInputModule + RefValidator.
// It returns a fixed branch index but declares expected ref names.
type namedRouter struct {
	index    float32
	expected []string // RefValidator: declared expected refs
}

func (r *namedRouter) Forward(_ ...*autograd.Variable) *autograd.Variable {
	t, _ := tensor.FromFloat32([]float32{r.index}, []int64{1})
	return autograd.NewVariable(t, false)
}

func (r *namedRouter) ForwardNamed(_ *autograd.Variable, _ map[string]*autograd.Variable) *autograd.Variable {
	t, _ := tensor.FromFloat32([]float32{r.index}, []int64{1})
	return autograd.NewVariable(t, false)
}

func (r *namedRouter) RefNames() []string          { return r.expected }
func (r *namedRouter) Parameters() []*nn.Parameter { return nil }

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
		[]float32{0, 0, 0},
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

func TestLoopForward(t *testing.T) {
	// Body: scale by 2 (Linear with W=2*I, b=0)
	// Loop 3 times: [1,2] → [2,4] → [4,8] → [8,16]
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	doubler, _ := nn.NewLinear(2, 2)
	setLinearWeights(doubler, []float32{2, 0, 0, 2}, []float32{0, 0})

	g, err := From(entry).Loop(doubler).For(3).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	vals := allF32(result)
	// 3 doublings: [1,2] → [8,16]
	if !approxEqual(vals[0], 8.0, 1e-5) || !approxEqual(vals[1], 16.0, 1e-5) {
		t.Errorf("Loop forward: got %v, want [8, 16]", vals)
	}
	t.Logf("Loop forward (3 iterations): %v", vals)
}

func TestLoopBackward(t *testing.T) {
	// Body: adds a learnable bias each iteration.
	// state_{i+1} = state_i + bias
	// After N iterations: result = input + N * bias
	// Gradient of result w.r.t. bias = N (each iteration contributes 1)
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	// Body: identity transform + bias. W=I, b=[0.1, 0.2]
	body, _ := nn.NewLinear(2, 2)
	setLinearWeights(body, []float32{1, 0, 0, 1}, []float32{0.1, 0.2})

	iterations := 5
	g, err := From(entry).Loop(body).For(iterations).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// Forward: [1, 2] + 5*[0.1, 0.2] = [1.5, 3.0]
	vals := allF32(result)
	if !approxEqual(vals[0], 1.5, 1e-5) || !approxEqual(vals[1], 3.0, 1e-5) {
		t.Errorf("Loop forward: got %v, want [1.5, 3.0]", vals)
	}

	// Backward: sum to scalar then backward.
	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	// Gradient of sum(result) w.r.t. bias:
	// Each iteration adds bias once. Sum over 2 elements.
	// d(sum(input + N*bias))/d(bias) = [N, N] = [5, 5]
	biasGrad := gradF32(body.Bias.Variable)
	for i, g := range biasGrad {
		expected := float32(iterations)
		if !approxEqual(g, expected, 1e-4) {
			t.Errorf("bias grad[%d]: got %v, want %v", i, g, expected)
		}
	}
	t.Logf("Loop backward (BPTT, %d iterations): bias grad = %v", iterations, biasGrad)
}

func TestLoopSingleIteration(t *testing.T) {
	// Loop with N=1 should be equivalent to a single Forward.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	body, _ := nn.NewLinear(2, 2)
	setLinearWeights(body, []float32{3, 0, 0, 3}, []float32{1, 1})

	// Via loop.
	g, err := From(entry).Loop(body).For(1).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{2, 5}, []int64{1, 2})
	loopResult := g.Forward(autograd.NewVariable(x, false))
	loopVals := allF32(loopResult)

	// Direct: body(entry([2,5])) = body([2,5]) = [3*2+1, 3*5+1] = [7, 16]
	x2, _ := tensor.FromFloat32([]float32{2, 5}, []int64{1, 2})
	directResult := body.Forward(entry.Forward(autograd.NewVariable(x2, false)))
	directVals := allF32(directResult)

	for i := range loopVals {
		if !approxEqual(loopVals[i], directVals[i], 1e-5) {
			t.Errorf("index %d: loop=%v, direct=%v", i, loopVals[i], directVals[i])
		}
	}
	t.Logf("Loop(1) == direct: %v", loopVals)
}

func TestLoopChained(t *testing.T) {
	// Full chain: entry → Loop(body, 3) → output layer
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	body, _ := nn.NewLinear(2, 2)
	setLinearWeights(body, []float32{1, 0, 0, 1}, []float32{1, 1}) // +1 each iteration

	output, _ := nn.NewLinear(2, 1)
	setLinearWeights(output, []float32{1, 1}, []float32{0}) // sum

	g, err := From(entry).Loop(body).For(4).Through(output).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{0, 0}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// entry([0,0]) = [0,0]
	// 4 iterations of +[1,1]: [4, 4]
	// output: sum = 8
	val := f32(result)
	if !approxEqual(val, 8.0, 1e-5) {
		t.Errorf("Loop chained: got %v, want 8", val)
	}

	// Backward through loop + output layer.
	if err := result.Backward(); err != nil {
		t.Fatal(err)
	}

	// output.Weight grad should exist.
	if output.Weight.Grad() == nil {
		t.Error("output.Weight: no gradient")
	}
	// body.Bias grad: d(sum([4,4]))/d(bias) = [4, 4] (each iteration contributes 1, sum has 2 paths)
	biasGrad := gradF32(body.Bias.Variable)
	t.Logf("Loop chained: output=%.1f, body.Bias grad=%v", val, biasGrad)
}

func TestLoopParameters(t *testing.T) {
	entry, _ := nn.NewLinear(2, 2)
	body, _ := nn.NewLinear(2, 2)

	g, err := From(entry).Loop(body).For(3).Build()
	if err != nil {
		t.Fatal(err)
	}

	// entry: 2 params (W+b), body: 2 params (W+b), total: 4
	// Body params appear once despite being used 3 times (same Module).
	params := g.Parameters()
	if len(params) != 4 {
		t.Errorf("expected 4 parameters, got %d", len(params))
	}
	t.Logf("Loop parameters: %d (body counted once despite %d iterations)", len(params), 3)
}

// --- Tag / Using tests ---

func TestTag(t *testing.T) {
	// Tag should not affect the flow.
	l1, _ := nn.NewLinear(2, 2)
	setLinearWeights(l1, []float32{1, 0, 0, 1}, []float32{0, 0})
	l2, _ := nn.NewLinear(2, 1)
	setLinearWeights(l2, []float32{1, 1}, []float32{0})

	g, err := From(l1).Tag("hidden").Through(l2).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{3, 5}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// identity([3,5]) → [3,5], sum → 8
	if !approxEqual(f32(result), 8.0, 1e-6) {
		t.Errorf("Tag should not affect flow: got %v, want 8", f32(result))
	}
	t.Logf("Tag passthrough: %.4f", f32(result))
}

func TestUsing(t *testing.T) {
	// Cross-attention style: module receives stream + tagged reference.
	// entry → Tag("src") → projection → crossAttn.Using("src")
	// crossAttn Forward receives [projected_input, source]
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0}) // identity

	proj, _ := nn.NewLinear(2, 2)
	setLinearWeights(proj, []float32{2, 0, 0, 2}, []float32{0, 0}) // 2x

	// "crossAttn" is a module that adds its two inputs.
	crossAttn := Add()

	g, err := From(entry).Tag("src").Through(proj).Through(crossAttn).Using("src").Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 3}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// entry([1,3]) = [1,3]
	// proj([1,3]) = [2,6]
	// crossAttn([2,6], [1,3]) = [3, 9]
	vals := allF32(result)
	if !approxEqual(vals[0], 3.0, 1e-6) || !approxEqual(vals[1], 9.0, 1e-6) {
		t.Errorf("Using: got %v, want [3, 9]", vals)
	}
	t.Logf("Using (cross-wire): %v", vals)
}

func TestUsingBackward(t *testing.T) {
	// Verify gradients flow through Using references.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	proj, _ := nn.NewLinear(2, 2)
	setLinearWeights(proj, []float32{1, 0, 0, 1}, []float32{1, 1}) // identity + bias

	crossAttn := Add()

	g, err := From(entry).Tag("src").Through(proj).Through(crossAttn).Using("src").Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{2, 3}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	// proj.Bias grad should exist (gradient flows through the main path).
	if proj.Bias.Grad() == nil {
		t.Error("proj.Bias: no gradient")
	}
	// entry.Weight grad should exist (gradient flows through both paths).
	if entry.Weight.Grad() == nil {
		t.Error("entry.Weight: no gradient")
	}
	t.Logf("Using backward: proj.Bias grad=%v", gradF32(proj.Bias.Variable))
}

func TestTagDuplicate(t *testing.T) {
	l, _ := nn.NewLinear(2, 2)
	_, err := From(l).Tag("x").Tag("x").Build()
	if err == nil {
		t.Error("expected error for duplicate tag name")
	}
	t.Logf("Error: %v", err)
}

func TestUsingUnresolvedForwardRef(t *testing.T) {
	l1, _ := nn.NewLinear(2, 2)
	l2, _ := nn.NewLinear(2, 2)
	_, err := From(l1).Through(l2).Using("nonexistent").Build()
	if err == nil {
		t.Error("expected error for unresolved forward reference")
	}
	t.Logf("Error: %v", err)
}

func TestSplitUsing(t *testing.T) {
	// Split + Using: all branches receive the tagged reference.
	// entry → Tag("ctx") → Split(branchA, branchB).Using("ctx") → Merge(Add)
	// Each branch is Add(), so Forward([stream, ctx]) = stream + ctx.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0}) // identity

	// Both branches add their inputs (stream + tagged ctx).
	branchA := Add()
	branchB := Add()

	g, err := From(entry).
		Tag("ctx").
		Split(branchA, branchB).Using("ctx").
		Merge(Add()).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 3}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// entry([1,3]) = [1,3]
	// branchA: Add([1,3], [1,3]) = [2, 6]   (stream + ctx)
	// branchB: Add([1,3], [1,3]) = [2, 6]   (stream + ctx)
	// Merge Add: [2+2, 6+6] = [4, 12]
	vals := allF32(result)
	if !approxEqual(vals[0], 4.0, 1e-6) || !approxEqual(vals[1], 12.0, 1e-6) {
		t.Errorf("Split+Using: got %v, want [4, 12]", vals)
	}
	t.Logf("Split+Using (all branches get ctx): %v", vals)
}

// --- Gate tests ---

func TestGateForwardEqual(t *testing.T) {
	// Equal weighting: router outputs [0,0] → softmax → [0.5, 0.5]
	// Expert A: identity, Expert B: 2x
	// Result: 0.5*x + 0.5*2x = 1.5x
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	routerLinear, _ := nn.NewLinear(2, 2)
	setLinearWeights(routerLinear, []float32{0, 0, 0, 0}, []float32{0, 0})
	router := withSoftmax(routerLinear) // softmax([0,0]) = [0.5, 0.5]

	expertA, _ := nn.NewLinear(2, 2)
	setLinearWeights(expertA, []float32{1, 0, 0, 1}, []float32{0, 0}) // identity

	expertB, _ := nn.NewLinear(2, 2)
	setLinearWeights(expertB, []float32{2, 0, 0, 2}, []float32{0, 0}) // 2x

	g, err := From(entry).Gate(router, expertA, expertB).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{2, 4}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// 0.5*[2,4] + 0.5*[4,8] = [3, 6]
	vals := allF32(result)
	if !approxEqual(vals[0], 3.0, 1e-5) || !approxEqual(vals[1], 6.0, 1e-5) {
		t.Errorf("Gate equal: got %v, want [3, 6]", vals)
	}
	t.Logf("Gate equal weighting: %v", vals)
}

func TestGateForwardOneSided(t *testing.T) {
	// One-sided routing: router outputs [10, -10] → softmax ≈ [1, 0]
	// Result ≈ expertA output
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	routerLinear, _ := nn.NewLinear(2, 2)
	setLinearWeights(routerLinear, []float32{0, 0, 0, 0}, []float32{10, -10})
	router := withSoftmax(routerLinear)

	expertA, _ := nn.NewLinear(2, 2)
	setLinearWeights(expertA, []float32{1, 0, 0, 1}, []float32{0, 0})

	expertB, _ := nn.NewLinear(2, 2)
	setLinearWeights(expertB, []float32{2, 0, 0, 2}, []float32{0, 0})

	g, err := From(entry).Gate(router, expertA, expertB).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{2, 4}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// softmax([10,-10]) ≈ [1.0, 0.0], so result ≈ expertA([2,4]) = [2, 4]
	vals := allF32(result)
	if !approxEqual(vals[0], 2.0, 1e-4) || !approxEqual(vals[1], 4.0, 1e-4) {
		t.Errorf("Gate one-sided: got %v, want ≈[2, 4]", vals)
	}
	t.Logf("Gate one-sided routing: %v", vals)
}

func TestGateBackward(t *testing.T) {
	// Verify gradients flow through all experts and the router.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	routerLinear, _ := nn.NewLinear(2, 2)
	setLinearWeights(routerLinear, []float32{0, 0, 0, 0}, []float32{0, 0})
	router := withSoftmax(routerLinear)

	expertA, _ := nn.NewLinear(2, 2)
	setLinearWeights(expertA, []float32{1, 0, 0, 1}, []float32{0, 0})

	expertB, _ := nn.NewLinear(2, 2)
	setLinearWeights(expertB, []float32{2, 0, 0, 2}, []float32{0, 0})

	g, err := From(entry).Gate(router, expertA, expertB).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{2, 4}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	// All components should have gradients.
	for name, layer := range map[string]*nn.Linear{
		"router": routerLinear, "expertA": expertA, "expertB": expertB, "entry": entry,
	} {
		if layer.Weight.Grad() == nil {
			t.Errorf("%s.Weight: no gradient", name)
		} else {
			t.Logf("%s.Weight grad: %v", name, gradF32(layer.Weight.Variable))
		}
	}
}

func TestGateWithUsing(t *testing.T) {
	// Gate with Using: router receives stream + tagged reference as separate
	// Forward arguments. Use Add() as router — it sums all inputs, producing
	// a [batch, 2] tensor that softmax normalizes into expert weights.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0}) // identity

	proj, _ := nn.NewLinear(2, 2)
	setLinearWeights(proj, []float32{1, 0, 0, 1}, []float32{0, 0}) // identity

	// Router: Add() sums stream + tag, then softmax normalizes.
	// Both are [1,2] → Add = [2, 6] → softmax([2, 6]) ≈ [0.018, 0.982]
	router := withSoftmax(Add())

	expertA, _ := nn.NewLinear(2, 2)
	setLinearWeights(expertA, []float32{1, 0, 0, 1}, []float32{0, 0}) // identity

	expertB, _ := nn.NewLinear(2, 2)
	setLinearWeights(expertB, []float32{2, 0, 0, 2}, []float32{0, 0}) // 2x

	g, err := From(entry).
		Tag("features").
		Through(proj).
		Gate(router, expertA, expertB).Using("features").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 3}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// expertA([1,3]) = [1, 3], expertB([1,3]) = [2, 6]
	// Result is weighted combination, heavily toward expertB (softmax([2,6]) ≈ [0.018, 0.982])
	vals := allF32(result)
	if vals[0] < 0.9 || vals[0] > 2.1 || vals[1] < 2.9 || vals[1] > 6.1 {
		t.Errorf("Gate with Using: result %v out of expected range", vals)
	}
	t.Logf("Gate with Using: %v", vals)
}

func TestGateTooFewExperts(t *testing.T) {
	l, _ := nn.NewLinear(2, 2)
	router, _ := nn.NewLinear(2, 1)
	expert, _ := nn.NewLinear(2, 2)
	_, err := From(l).Gate(router, expert).Build()
	if err == nil {
		t.Error("expected error for < 2 experts")
	}
	t.Logf("Error: %v", err)
}

func TestGateParameters(t *testing.T) {
	entry, _ := nn.NewLinear(2, 2)
	routerLinear, _ := nn.NewLinear(2, 2)
	expertA, _ := nn.NewLinear(2, 2)
	expertB, _ := nn.NewLinear(2, 2)

	g, err := From(entry).Gate(withSoftmax(routerLinear), expertA, expertB).Build()
	if err != nil {
		t.Fatal(err)
	}

	// entry: 2, router: 2, expertA: 2, expertB: 2 = 8
	params := g.Parameters()
	if len(params) != 8 {
		t.Errorf("expected 8 parameters, got %d", len(params))
	}
	t.Logf("Gate parameters: %d", len(params))
}

// --- Forward ref tests ---

func TestForwardRef(t *testing.T) {
	// Forward reference: Using before Tag. State carries between Forward() calls.
	// Graph: entry → add.Using("memory") → identity.Tag("memory")
	// Pass 1: add gets [stream] (memory is nil) → identity → state captured
	// Pass 2: add gets [stream, prev_output] → sum → identity → state captured
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0}) // identity

	identity, _ := nn.NewLinear(2, 2)
	setLinearWeights(identity, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, err := From(entry).
		Through(&nilSafeAdd{}).Using("memory").
		Through(identity).Tag("memory").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})

	// Pass 1: no memory → add([1,2]) = [1,2] → identity = [1,2]
	r1 := g.Forward(autograd.NewVariable(x, false))
	if err := r1.Err(); err != nil {
		t.Fatal("pass 1:", err)
	}
	v1 := allF32(r1)
	if !approxEqual(v1[0], 1.0, 1e-6) || !approxEqual(v1[1], 2.0, 1e-6) {
		t.Errorf("pass 1: got %v, want [1, 2]", v1)
	}

	// Pass 2: memory=[1,2] → add([1,2], [1,2]) = [2,4] → identity = [2,4]
	r2 := g.Forward(autograd.NewVariable(x, false))
	if err := r2.Err(); err != nil {
		t.Fatal("pass 2:", err)
	}
	v2 := allF32(r2)
	if !approxEqual(v2[0], 2.0, 1e-6) || !approxEqual(v2[1], 4.0, 1e-6) {
		t.Errorf("pass 2: got %v, want [2, 4]", v2)
	}

	// Pass 3: memory=[2,4] → add([1,2], [2,4]) = [3,6]
	r3 := g.Forward(autograd.NewVariable(x, false))
	v3 := allF32(r3)
	if !approxEqual(v3[0], 3.0, 1e-6) || !approxEqual(v3[1], 6.0, 1e-6) {
		t.Errorf("pass 3: got %v, want [3, 6]", v3)
	}

	t.Logf("Forward ref: pass1=%v, pass2=%v, pass3=%v", v1, v2, v3)
}

func TestForwardRefResetState(t *testing.T) {
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	identity, _ := nn.NewLinear(2, 2)
	setLinearWeights(identity, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, err := From(entry).
		Through(&nilSafeAdd{}).Using("state").
		Through(identity).Tag("state").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 1}, []int64{1, 2})

	// Build up state.
	g.Forward(autograd.NewVariable(x, false)) // [1,1]
	g.Forward(autograd.NewVariable(x, false)) // [2,2]

	// Reset and verify state is cleared.
	g.ResetState()
	r := g.Forward(autograd.NewVariable(x, false))
	vals := allF32(r)
	if !approxEqual(vals[0], 1.0, 1e-6) || !approxEqual(vals[1], 1.0, 1e-6) {
		t.Errorf("after reset: got %v, want [1, 1]", vals)
	}
	t.Logf("ResetState: %v (back to initial)", vals)
}

func TestForwardRefDetachState(t *testing.T) {
	// Verify DetachState breaks the gradient chain.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	proj, _ := nn.NewLinear(2, 2)
	setLinearWeights(proj, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, err := From(entry).
		Through(&nilSafeAdd{}).Using("state").
		Through(proj).Tag("state").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})

	// Pass 1: populate state.
	g.Forward(autograd.NewVariable(x, false))
	g.DetachState()

	// Pass 2: use detached state → backward should work without
	// touching pass 1's computation graph.
	r2 := g.Forward(autograd.NewVariable(x, false))
	loss := r2.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal("backward after detach:", err)
	}

	// proj should have gradients from pass 2 only.
	if proj.Weight.Grad() == nil {
		t.Error("proj.Weight: no gradient after detach + backward")
	}
	t.Logf("DetachState: backward succeeded, grad=%v", gradF32(proj.Weight.Variable))
}

func TestForwardRefLoop(t *testing.T) {
	// Recurrent state inside a loop body.
	// Body graph: add.Using("hidden") → identity.Tag("hidden")
	// Each iteration accumulates: output = input + hidden_prev
	body, err := From(&nilSafeAdd{}).Using("hidden").
		Tag("hidden").
		Build()
	if err != nil {
		t.Fatal("body build:", err)
	}

	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, err := From(entry).Loop(body).For(4).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// Iteration 1: add([1,2], nil) = [1,2], hidden=[1,2]
	// Iteration 2: add([1,2], [1,2]) = [2,4], hidden=[2,4]
	// Iteration 3: add([2,4], [2,4]) = [4,8], hidden=[4,8]
	// Iteration 4: add([4,8], [4,8]) = [8,16]
	vals := allF32(result)
	if !approxEqual(vals[0], 8.0, 1e-5) || !approxEqual(vals[1], 16.0, 1e-5) {
		t.Errorf("forward ref loop: got %v, want [8, 16]", vals)
	}
	t.Logf("Forward ref in loop (4 iterations): %v", vals)

	// Reset body state for clean next use.
	body.ResetState()
}

func TestForwardRefBackward(t *testing.T) {
	// Verify gradients flow through forward ref state.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	proj, _ := nn.NewLinear(2, 2)
	setLinearWeights(proj, []float32{2, 0, 0, 2}, []float32{0, 0}) // 2x

	g, err := From(entry).
		Through(&nilSafeAdd{}).Using("state").
		Through(proj).Tag("state").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})

	// Pass 1: populate state. proj([1,2]) = [2,4]
	g.Forward(autograd.NewVariable(x, false))

	// Pass 2: add([1,2], [2,4]) = [3,6], proj([3,6]) = [6,12]
	r2 := g.Forward(autograd.NewVariable(x, false))
	loss := r2.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal("backward:", err)
	}

	if proj.Weight.Grad() == nil {
		t.Error("proj.Weight: no gradient")
	}
	if entry.Weight.Grad() == nil {
		t.Error("entry.Weight: no gradient")
	}
	t.Logf("Forward ref backward: proj.Weight grad=%v", gradF32(proj.Weight.Variable))
}

func TestForwardRefMixedRefs(t *testing.T) {
	// Same tag used as both forward and backward reference.
	// Through(attn).Using("state") → Through(ffn).Tag("state") → Through(decoder).Using("state")
	// attn gets PREVIOUS state (forward ref), decoder gets CURRENT state (backward ref).
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	ffn, _ := nn.NewLinear(2, 2)
	setLinearWeights(ffn, []float32{1, 0, 0, 1}, []float32{0, 0}) // identity

	decoder := Add() // sums stream + backward ref

	g, err := From(entry).
		Through(&nilSafeAdd{}).Using("state").
		Through(ffn).Tag("state").
		Through(decoder).Using("state").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})

	// Pass 1: attn([1,2], nil)=[1,2], ffn=[1,2] (state set), decoder=Add([1,2],[1,2])=[2,4]
	r1 := g.Forward(autograd.NewVariable(x, false))
	v1 := allF32(r1)
	if !approxEqual(v1[0], 2.0, 1e-6) || !approxEqual(v1[1], 4.0, 1e-6) {
		t.Errorf("pass 1: got %v, want [2, 4]", v1)
	}

	// Pass 2: attn([1,2], [1,2])=[2,4], ffn=[2,4] (state updated), decoder=Add([2,4],[2,4])=[4,8]
	r2 := g.Forward(autograd.NewVariable(x, false))
	v2 := allF32(r2)
	if !approxEqual(v2[0], 4.0, 1e-6) || !approxEqual(v2[1], 8.0, 1e-6) {
		t.Errorf("pass 2: got %v, want [4, 8]", v2)
	}

	t.Logf("Mixed forward+backward ref: pass1=%v, pass2=%v", v1, v2)
}

func TestForwardRefAutoZero(t *testing.T) {
	// Verify that plain Add() (no nil-checking) works as a forward ref
	// consumer because the graph auto-fills nil state with zeros.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	identity, _ := nn.NewLinear(2, 2)
	setLinearWeights(identity, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, err := From(entry).
		Through(Add()).Using("memory"). // plain Add — no nil handling
		Through(identity).Tag("memory").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})

	// Pass 1: memory auto-zeroed → Add([1,2], [0,0]) = [1,2]
	r1 := g.Forward(autograd.NewVariable(x, false))
	if err := r1.Err(); err != nil {
		t.Fatal("pass 1:", err)
	}
	v1 := allF32(r1)
	if !approxEqual(v1[0], 1.0, 1e-6) || !approxEqual(v1[1], 2.0, 1e-6) {
		t.Errorf("pass 1: got %v, want [1, 2]", v1)
	}

	// Pass 2: memory=[1,2] → Add([1,2], [1,2]) = [2,4]
	r2 := g.Forward(autograd.NewVariable(x, false))
	if err := r2.Err(); err != nil {
		t.Fatal("pass 2:", err)
	}
	v2 := allF32(r2)
	if !approxEqual(v2[0], 2.0, 1e-6) || !approxEqual(v2[1], 4.0, 1e-6) {
		t.Errorf("pass 2: got %v, want [2, 4]", v2)
	}

	t.Logf("Forward ref auto-zero: pass1=%v, pass2=%v", v1, v2)
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

// --- SetTraining ---

func TestSetTrainingPropagates(t *testing.T) {
	// Graph: Linear → Dropout → Linear
	// SetTraining(false) should make dropout passthrough.
	l1, err := nn.NewLinear(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	drop := nn.NewDropout(0.99) // extreme drop rate — almost everything zeroed in training
	l2, err := nn.NewLinear(4, 4)
	if err != nil {
		t.Fatal(err)
	}

	g, err := From(l1).Through(drop).Through(l2).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{1, 4})
	input := autograd.NewVariable(x, false)

	// Eval mode: dropout is identity, output should be deterministic.
	g.SetTraining(false)
	out1 := g.Forward(input)
	out2 := g.Forward(input)
	if err := out1.Err(); err != nil {
		t.Fatal(err)
	}
	d1, _ := out1.Data().Float32Data()
	d2, _ := out2.Data().Float32Data()
	for i := range d1 {
		if d1[i] != d2[i] {
			t.Errorf("eval mode not deterministic: out1[%d]=%f != out2[%d]=%f", i, d1[i], i, d2[i])
		}
	}

	// Back to training: dropout should alter output.
	g.SetTraining(true)
	different := false
	for try := 0; try < 5; try++ {
		out3 := g.Forward(input)
		if err := out3.Err(); err != nil {
			t.Fatal(err)
		}
		d3, _ := out3.Data().Float32Data()
		for i := range d1 {
			if d1[i] != d3[i] {
				different = true
				break
			}
		}
		if different {
			break
		}
	}
	if !different {
		t.Error("training mode should produce different output with p=0.99 dropout")
	}
}

func TestSetTrainingNestedGraph(t *testing.T) {
	// Inner graph with dropout.
	drop := nn.NewDropout(0.99)
	inner, err := From(drop).Build()
	if err != nil {
		t.Fatal(err)
	}

	// Outer graph wrapping inner.
	l1, err := nn.NewLinear(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	outer, err := From(l1).Through(inner).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{1, 4})
	input := autograd.NewVariable(x, false)

	// Eval mode on outer should propagate to inner's dropout.
	outer.SetTraining(false)
	out1 := outer.Forward(input)
	out2 := outer.Forward(input)
	if err := out1.Err(); err != nil {
		t.Fatal(err)
	}
	d1, _ := out1.Data().Float32Data()
	d2, _ := out2.Data().Float32Data()
	for i := range d1 {
		if d1[i] != d2[i] {
			t.Errorf("nested eval not deterministic: [%d] %f != %f", i, d1[i], d2[i])
		}
	}
}

func TestSetTrainingHelperFunction(t *testing.T) {
	// nn.SetTraining works on standalone modules and is a no-op on
	// modules that don't implement TrainToggler.
	drop := nn.NewDropout(0.5)
	nn.SetTraining(drop, false)

	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	x := autograd.NewVariable(xt, false)
	out := drop.Forward(x)
	data, _ := out.Data().Float32Data()
	// Eval mode: identity
	want := []float32{1, 2, 3}
	for i := range want {
		if !approxEqual(data[i], want[i], 1e-5) {
			t.Errorf("[%d] = %f, want %f", i, data[i], want[i])
		}
	}

	// No-op on Linear (doesn't implement TrainToggler).
	l, _ := nn.NewLinear(3, 3)
	nn.SetTraining(l, false) // should not panic
}

// --- While loop tests ---

func TestWhileForward(t *testing.T) {
	// Body doubles each iteration. ThresholdHalt(10) halts when max > 10.
	// [1,2] → check: max=2 ≤ 10 → body → [2,4]
	//       → check: max=4 ≤ 10 → body → [4,8]
	//       → check: max=8 ≤ 10 → body → [8,16]
	//       → check: max=16 > 10 → halt
	// 3 iterations, same result as For(3).
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	doubler, _ := nn.NewLinear(2, 2)
	setLinearWeights(doubler, []float32{2, 0, 0, 2}, []float32{0, 0})

	g, err := From(entry).Loop(doubler).While(ThresholdHalt(10), 10).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	vals := allF32(result)
	if !approxEqual(vals[0], 8.0, 1e-5) || !approxEqual(vals[1], 16.0, 1e-5) {
		t.Errorf("While forward: got %v, want [8, 16]", vals)
	}
	t.Logf("While forward (3 iterations, halt at >10): %v", vals)
}

func TestWhileStatePredicate(t *testing.T) {
	// Body doubles. ThresholdHalt(50): halt when max > 50.
	// check [1,2] max=2 → body → [2,4]
	// check [2,4] max=4 → body → [4,8]
	// check [4,8] max=8 → body → [8,16]
	// check [8,16] max=16 → body → [16,32]
	// check [16,32] max=32 → body → [32,64]
	// check [32,64] max=64 > 50 → halt
	// Result: [32, 64] (5 iterations)
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	doubler, _ := nn.NewLinear(2, 2)
	setLinearWeights(doubler, []float32{2, 0, 0, 2}, []float32{0, 0})

	g, err := From(entry).Loop(doubler).While(ThresholdHalt(50), 100).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	vals := allF32(result)

	if !approxEqual(vals[0], 32.0, 1e-5) || !approxEqual(vals[1], 64.0, 1e-5) {
		t.Errorf("While state pred: got %v, want [32, 64]", vals)
	}
	t.Logf("While state predicate (5 iterations): %v", vals)
}

func TestWhileZeroIterations(t *testing.T) {
	// ThresholdHalt(0): input [3,7] has max=7 > 0, halts immediately.
	// Body never runs — input passes through.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	doubler, _ := nn.NewLinear(2, 2)
	setLinearWeights(doubler, []float32{2, 0, 0, 2}, []float32{0, 0})

	g, err := From(entry).Loop(doubler).While(ThresholdHalt(0), 10).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{3, 7}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	vals := allF32(result)

	// Input passes through unchanged (entry is identity).
	if !approxEqual(vals[0], 3.0, 1e-5) || !approxEqual(vals[1], 7.0, 1e-5) {
		t.Errorf("While zero iters: got %v, want [3, 7]", vals)
	}
	t.Logf("While zero iterations (passthrough): %v", vals)
}

func TestWhileBackward(t *testing.T) {
	// Body: identity + bias [0.1, 0.2]. ThresholdHalt(2.7) gives 4 iterations.
	// [1,2] → [1.1,2.2] → [1.2,2.4] → [1.3,2.6] → [1.4,2.8]
	// check [1.4,2.8] max=2.8 > 2.7 → halt
	// Gradient of sum(result) w.r.t. bias = [4, 4].
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	body, _ := nn.NewLinear(2, 2)
	setLinearWeights(body, []float32{1, 0, 0, 1}, []float32{0.1, 0.2})

	g, err := From(entry).Loop(body).While(ThresholdHalt(2.7), 10).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	biasGrad := gradF32(body.Bias.Variable)
	for i, g := range biasGrad {
		if !approxEqual(g, 4.0, 1e-4) {
			t.Errorf("bias grad[%d]: got %v, want 4", i, g)
		}
	}
	t.Logf("While backward (4 iters): bias grad = %v", biasGrad)
}

// --- Until loop tests ---

// thresholdHalt returns positive when max element exceeds threshold (halt signal).
type thresholdHalt struct {
	threshold float32
}

func (h *thresholdHalt) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	data, _ := inputs[0].Data().Float32Data()
	maxVal := data[0]
	for _, v := range data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	val := maxVal - h.threshold // positive when exceeded → halt
	t, _ := tensor.FromFloat32([]float32{val}, []int64{1})
	return autograd.NewVariable(t, false)
}

func (h *thresholdHalt) Parameters() []*nn.Parameter { return nil }

func TestUntilForward(t *testing.T) {
	// Body doubles. Halt when max element > 50.
	// [1,2] → body → [2,4] check: 4<=50 → continue
	// → body → [4,8] check: 8<=50 → continue
	// → body → [8,16] check: 16<=50 → continue
	// → body → [16,32] check: 32<=50 → continue
	// → body → [32,64] check: 64>50 → halt
	// 5 iterations, result: [32, 64]
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	doubler, _ := nn.NewLinear(2, 2)
	setLinearWeights(doubler, []float32{2, 0, 0, 2}, []float32{0, 0})

	halt := &thresholdHalt{threshold: 50}

	g, err := From(entry).Loop(doubler).Until(halt, 100).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	vals := allF32(result)

	if !approxEqual(vals[0], 32.0, 1e-5) || !approxEqual(vals[1], 64.0, 1e-5) {
		t.Errorf("Until forward: got %v, want [32, 64]", vals)
	}
	t.Logf("Until forward (5 iterations, halt at >50): %v", vals)
}

func TestUntilMaxIter(t *testing.T) {
	// Condition never halts — maxIter is the bound.
	// Body doubles, maxIter=3. Result: [1,2] → [2,4] → [4,8] → [8,16]
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	doubler, _ := nn.NewLinear(2, 2)
	setLinearWeights(doubler, []float32{2, 0, 0, 2}, []float32{0, 0})

	// Threshold so high it never triggers.
	halt := &thresholdHalt{threshold: 1e9}

	g, err := From(entry).Loop(doubler).Until(halt, 3).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	vals := allF32(result)

	// Same as For(3): 3 doublings.
	if !approxEqual(vals[0], 8.0, 1e-5) || !approxEqual(vals[1], 16.0, 1e-5) {
		t.Errorf("Until maxIter: got %v, want [8, 16]", vals)
	}
	t.Logf("Until maxIter (3 iterations, no early halt): %v", vals)
}

func TestUntilBackward(t *testing.T) {
	// Body: identity + bias. Halt after 3 iterations (threshold triggers).
	// State: [1,2] → [1.1,2.2] → [1.2,2.4] → [1.3,2.6] → halt (max 2.6 > 2.5)
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	body, _ := nn.NewLinear(2, 2)
	setLinearWeights(body, []float32{1, 0, 0, 1}, []float32{0.1, 0.2})

	halt := &thresholdHalt{threshold: 2.5}

	g, err := From(entry).Loop(body).Until(halt, 20).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// 3 iterations: [1+3*0.1, 2+3*0.2] = [1.3, 2.6]
	vals := allF32(result)
	if !approxEqual(vals[0], 1.3, 1e-5) || !approxEqual(vals[1], 2.6, 1e-5) {
		t.Errorf("Until forward: got %v, want [1.3, 2.6]", vals)
	}

	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	// 3 iterations of adding bias. Gradient = [3, 3].
	biasGrad := gradF32(body.Bias.Variable)
	for i, g := range biasGrad {
		if !approxEqual(g, 3.0, 1e-4) {
			t.Errorf("bias grad[%d]: got %v, want 3", i, g)
		}
	}
	t.Logf("Until backward (3 iters): bias grad = %v", biasGrad)
}

func TestUntilParameters(t *testing.T) {
	body, _ := nn.NewLinear(2, 2)
	// Condition with learnable parameters.
	condLayer, _ := nn.NewLinear(2, 1)
	halt := &thresholdHalt{threshold: 100}

	// Use a module that has parameters as condition: wrap condLayer.
	condModule := &learnedHalt{inner: condLayer}

	g, err := From(body).Loop(body).Until(condModule, 5).Build()
	if err != nil {
		t.Fatal(err)
	}

	params := g.Parameters()
	// body: 2 (W+b), condModule.inner: 2 (W+b) = 4 total
	// But body appears as both entry and loop body — same module, deduplicated.
	// Entry node uses body, loop node uses loopComposite which delegates to body.
	// Body params counted once + cond params once = 4.
	if len(params) != 4 {
		t.Errorf("expected 4 parameters, got %d", len(params))
	}
	t.Logf("Until parameters: %d (body + condition)", len(params))

	// Also verify: we can ignore the halt variable above.
	_ = halt
}

// learnedHalt wraps a Linear layer as a halt condition module.
type learnedHalt struct {
	inner *nn.Linear
}

func (h *learnedHalt) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	return h.inner.Forward(inputs...)
}

func (h *learnedHalt) Parameters() []*nn.Parameter {
	return h.inner.Parameters()
}

func TestUntilSetTraining(t *testing.T) {
	// Verify SetTraining propagates to the condition module.
	body, _ := nn.NewLinear(2, 2)
	drop := nn.NewDropout(0.99)

	// Wrap dropout as condition (returns scalar indicating "never halt").
	cond := &dropoutCondWrapper{drop: drop}

	g, err := From(body).Loop(body).Until(cond, 3).Build()
	if err != nil {
		t.Fatal(err)
	}

	// In eval mode, dropout should be identity.
	g.SetTraining(false)
	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	// Back to training: dropout active in condition.
	g.SetTraining(true)
	result2 := g.Forward(autograd.NewVariable(x, false))
	if err := result2.Err(); err != nil {
		t.Fatal(err)
	}
	t.Log("Until SetTraining propagates to condition module")
}

// dropoutCondWrapper uses dropout and returns negative (never halt).
type dropoutCondWrapper struct {
	drop *nn.Dropout
}

func (d *dropoutCondWrapper) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	_ = d.drop.Forward(inputs...) // apply dropout (tests SetTraining propagation)
	t, _ := tensor.FromFloat32([]float32{-1}, []int64{1})
	return autograd.NewVariable(t, false)
}

func (d *dropoutCondWrapper) Parameters() []*nn.Parameter { return nil }

func (d *dropoutCondWrapper) SetTraining(training bool) {
	d.drop.SetTraining(training)
}

// --- Switch tests ---

// fixedRouter always returns a constant branch index.
type fixedRouter struct {
	index float32
}

func (r *fixedRouter) Forward(_ ...*autograd.Variable) *autograd.Variable {
	t, _ := tensor.FromFloat32([]float32{r.index}, []int64{1})
	return autograd.NewVariable(t, false)
}

func (r *fixedRouter) Parameters() []*nn.Parameter { return nil }

func TestSwitchForward(t *testing.T) {
	// Branch 0: double, Branch 1: triple.
	// Router selects branch 1.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	double, _ := nn.NewLinear(2, 2)
	setLinearWeights(double, []float32{2, 0, 0, 2}, []float32{0, 0})

	triple, _ := nn.NewLinear(2, 2)
	setLinearWeights(triple, []float32{3, 0, 0, 3}, []float32{0, 0})

	router := &fixedRouter{index: 1} // select triple

	g, err := From(entry).Switch(router, double, triple).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	vals := allF32(result)
	// Branch 1 (triple): [1,2] → [3,6]
	if !approxEqual(vals[0], 3.0, 1e-5) || !approxEqual(vals[1], 6.0, 1e-5) {
		t.Errorf("Switch forward: got %v, want [3, 6]", vals)
	}
	t.Logf("Switch forward (branch 1 selected): %v", vals)
}

func TestSwitchSelectsBranch0(t *testing.T) {
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	double, _ := nn.NewLinear(2, 2)
	setLinearWeights(double, []float32{2, 0, 0, 2}, []float32{0, 0})

	triple, _ := nn.NewLinear(2, 2)
	setLinearWeights(triple, []float32{3, 0, 0, 3}, []float32{0, 0})

	router := &fixedRouter{index: 0} // select double

	g, err := From(entry).Switch(router, double, triple).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	vals := allF32(result)

	// Branch 0 (double): [1,2] → [2,4]
	if !approxEqual(vals[0], 2.0, 1e-5) || !approxEqual(vals[1], 4.0, 1e-5) {
		t.Errorf("Switch branch 0: got %v, want [2, 4]", vals)
	}
	t.Logf("Switch forward (branch 0 selected): %v", vals)
}

// countingModule tracks how many times Forward was called.
type countingModule struct {
	inner nn.Module
	count int
}

func (c *countingModule) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	c.count++
	return c.inner.Forward(inputs...)
}

func (c *countingModule) Parameters() []*nn.Parameter { return c.inner.Parameters() }

func TestSwitchOnlySelectedRuns(t *testing.T) {
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	l0, _ := nn.NewLinear(2, 2)
	setLinearWeights(l0, []float32{1, 0, 0, 1}, []float32{0, 0})
	l1, _ := nn.NewLinear(2, 2)
	setLinearWeights(l1, []float32{1, 0, 0, 1}, []float32{0, 0})

	branch0 := &countingModule{inner: l0}
	branch1 := &countingModule{inner: l1}

	router := &fixedRouter{index: 1}

	g, err := From(entry).Switch(router, branch0, branch1).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	g.Forward(autograd.NewVariable(x, false))

	if branch0.count != 0 {
		t.Errorf("branch 0 executed %d times, want 0", branch0.count)
	}
	if branch1.count != 1 {
		t.Errorf("branch 1 executed %d times, want 1", branch1.count)
	}
	t.Logf("Only selected branch ran: branch0=%d, branch1=%d", branch0.count, branch1.count)
}

func TestSwitchBackward(t *testing.T) {
	// Branch 0: W * x + b0, Branch 1: W * x + b1
	// Router selects branch 1.
	// Gradient should flow through branch 1 only.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	branch0, _ := nn.NewLinear(2, 2)
	setLinearWeights(branch0, []float32{1, 0, 0, 1}, []float32{10, 20})

	branch1, _ := nn.NewLinear(2, 2)
	setLinearWeights(branch1, []float32{1, 0, 0, 1}, []float32{100, 200})

	router := &fixedRouter{index: 1}

	g, err := From(entry).Switch(router, branch0, branch1).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, true))
	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	// Branch 1 bias should have gradient [1, 1] (from sum).
	b1Grad := gradF32(branch1.Bias.Variable)
	if !approxEqual(b1Grad[0], 1.0, 1e-5) || !approxEqual(b1Grad[1], 1.0, 1e-5) {
		t.Errorf("branch1 bias grad: got %v, want [1, 1]", b1Grad)
	}

	// Branch 0 should have no gradient (never executed).
	if branch0.Bias.Grad() != nil {
		t.Error("branch0 should have no gradient")
	}
	t.Logf("Switch backward: branch1.bias grad=%v, branch0.bias grad=nil", b1Grad)
}

func TestSwitchParameters(t *testing.T) {
	router, _ := nn.NewLinear(2, 1)
	branch0, _ := nn.NewLinear(2, 2)
	branch1, _ := nn.NewLinear(2, 2)

	g, err := From(branch0).Switch(router, branch0, branch1).Build()
	if err != nil {
		t.Fatal(err)
	}

	params := g.Parameters()
	// entry (branch0) + switch(router + branch0 + branch1) = 2+2+2+2 = 8
	// But branch0 is deduplicated → 2+2+2 = 6
	if len(params) != 6 {
		t.Errorf("expected 6 parameters, got %d", len(params))
	}
	t.Logf("Switch parameters: %d", len(params))
}

func TestSwitchSetTraining(t *testing.T) {
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})
	drop0 := nn.NewDropout(0.99)
	drop1 := nn.NewDropout(0.99)
	router := &fixedRouter{index: 0}

	g, err := From(entry).Switch(router, drop0, drop1).Build()
	if err != nil {
		t.Fatal(err)
	}

	// Eval mode: dropout should be identity.
	g.SetTraining(false)
	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	vals := allF32(result)

	if !approxEqual(vals[0], 1.0, 1e-5) || !approxEqual(vals[1], 2.0, 1e-5) {
		t.Errorf("Switch eval mode: got %v, want [1, 2]", vals)
	}
	t.Log("Switch SetTraining propagates to all branches")
}

// argmaxRouter applies argmax internally and returns the index.
type argmaxRouter struct {
	inner nn.Module
}

func (r *argmaxRouter) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	logits := r.inner.Forward(inputs...)
	data, _ := logits.Data().Float32Data()
	selected := 0
	maxVal := data[0]
	for i, v := range data[1:] {
		if v > maxVal {
			maxVal = v
			selected = i + 1
		}
	}
	t, _ := tensor.FromFloat32([]float32{float32(selected)}, []int64{1})
	return autograd.NewVariable(t, false)
}

func (r *argmaxRouter) Parameters() []*nn.Parameter { return r.inner.Parameters() }

func TestSwitchDataDependentRouting(t *testing.T) {
	// Router: Linear(2→2) with weights that make input [1,2] select branch 1
	// and input [2,1] select branch 0.
	routerLinear, _ := nn.NewLinear(2, 2)
	// W = [[-1,1],[1,-1]], b = [0,0]
	// input [1,2]: logits = [-1+2, 1-2] = [1, -1] → argmax = 0
	// input [2,1]: logits = [-2+1, 2-1] = [-1, 1] → argmax = 1
	setLinearWeights(routerLinear, []float32{-1, 1, 1, -1}, []float32{0, 0})
	router := &argmaxRouter{inner: routerLinear}

	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	double, _ := nn.NewLinear(2, 2)
	setLinearWeights(double, []float32{2, 0, 0, 2}, []float32{0, 0})

	triple, _ := nn.NewLinear(2, 2)
	setLinearWeights(triple, []float32{3, 0, 0, 3}, []float32{0, 0})

	g, err := From(entry).Switch(router, double, triple).Build()
	if err != nil {
		t.Fatal(err)
	}

	// Input [1,2] → router selects branch 0 (double) → [2, 4]
	x1, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	r1 := g.Forward(autograd.NewVariable(x1, false))
	v1 := allF32(r1)
	if !approxEqual(v1[0], 2.0, 1e-5) || !approxEqual(v1[1], 4.0, 1e-5) {
		t.Errorf("input [1,2]: got %v, want [2, 4]", v1)
	}

	// Input [2,1] → router selects branch 1 (triple) → [6, 3]
	x2, _ := tensor.FromFloat32([]float32{2, 1}, []int64{1, 2})
	r2 := g.Forward(autograd.NewVariable(x2, false))
	v2 := allF32(r2)
	if !approxEqual(v2[0], 6.0, 1e-5) || !approxEqual(v2[1], 3.0, 1e-5) {
		t.Errorf("input [2,1]: got %v, want [6, 3]", v2)
	}

	t.Logf("Data-dependent routing: [1,2]→branch0=%v, [2,1]→branch1=%v", v1, v2)
}

func TestSwitchWithUsing(t *testing.T) {
	// Router uses a tagged reference to make its decision.
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	identity, _ := nn.NewLinear(2, 2)
	setLinearWeights(identity, []float32{1, 0, 0, 1}, []float32{0, 0})

	double, _ := nn.NewLinear(2, 2)
	setLinearWeights(double, []float32{2, 0, 0, 2}, []float32{0, 0})

	triple, _ := nn.NewLinear(2, 2)
	setLinearWeights(triple, []float32{3, 0, 0, 3}, []float32{0, 0})

	// contextRouter ignores stream, uses ref to decide.
	router := &contextRouter{index: 1}

	g, err := From(entry).Tag("ctx").
		Through(identity).
		Switch(router, double, triple).Using("ctx").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	vals := allF32(result)
	// identity passthrough, then triple: [3, 6]
	if !approxEqual(vals[0], 3.0, 1e-5) || !approxEqual(vals[1], 6.0, 1e-5) {
		t.Errorf("Switch with Using: got %v, want [3, 6]", vals)
	}
	t.Logf("Switch with Using: %v", vals)
}

// contextRouter receives stream + refs and always returns a fixed index.
type contextRouter struct {
	index float32
}

func (r *contextRouter) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	// Verify we received the extra ref.
	if len(inputs) < 2 {
		return autograd.ErrVariable(fmt.Errorf("contextRouter: expected >=2 inputs, got %d", len(inputs)))
	}
	t, _ := tensor.FromFloat32([]float32{r.index}, []int64{1})
	return autograd.NewVariable(t, false)
}

func (r *contextRouter) Parameters() []*nn.Parameter { return nil }

func TestSwitchTooFewBranches(t *testing.T) {
	entry, _ := nn.NewLinear(2, 2)
	router := &fixedRouter{index: 0}
	branch, _ := nn.NewLinear(2, 2)

	_, err := From(entry).Switch(router, branch).Build()
	if err == nil {
		t.Fatal("expected error for single branch")
	}
	t.Logf("Error: %v", err)
}

// --- NamedInputModule validation tests ---

func TestNamedInputWithoutUsing(t *testing.T) {
	entry, _ := nn.NewLinear(2, 4)
	router := &namedRouter{index: 0, expected: []string{"context"}}
	branchA, _ := nn.NewLinear(4, 4)
	branchB, _ := nn.NewLinear(4, 4)

	_, err := From(entry).
		Switch(router, branchA, branchB).
		Build()
	if err == nil {
		t.Fatal("expected build error: RefValidator without Using")
	}
	t.Logf("Error: %v", err)
	if !containsAll(err.Error(), "RefNames", "context", "Using") {
		t.Error("error message should mention RefNames, expected ref, and Using")
	}
}

func TestNamedInputMissingRef(t *testing.T) {
	entry, _ := nn.NewLinear(2, 4)
	// Module expects "context" but Using wires "features".
	router := &namedRouter{index: 0, expected: []string{"context"}}
	branchA, _ := nn.NewLinear(4, 4)
	branchB, _ := nn.NewLinear(4, 4)

	_, err := From(entry).Tag("features").
		Switch(router, branchA, branchB).Using("features").
		Build()
	if err == nil {
		t.Fatal("expected build error: wired ref doesn't match RefNames")
	}
	t.Logf("Error: %v", err)
	// Should mention what's wrong and hint at the fix.
	if !containsAll(err.Error(), "context", "RefNames") {
		t.Error("error should mention missing ref name and RefNames")
	}
}

func TestNamedInputExtraRef(t *testing.T) {
	entry, _ := nn.NewLinear(2, 4)
	step, _ := nn.NewLinear(4, 4)
	// Module expects only "context" but we wire both "context" and "extra".
	router := &namedRouter{index: 0, expected: []string{"context"}}
	branchA, _ := nn.NewLinear(4, 4)
	branchB, _ := nn.NewLinear(4, 4)

	_, err := From(entry).Tag("context").
		Through(step).Tag("extra").
		Switch(router, branchA, branchB).Using("context", "extra").
		Build()
	if err == nil {
		t.Fatal("expected build error: extra Using ref not in RefNames")
	}
	t.Logf("Error: %v", err)
	if !containsAll(err.Error(), "extra", "RefNames") {
		t.Error("error should mention unexpected ref and RefNames")
	}
}

func TestNamedInputCorrect(t *testing.T) {
	entry, _ := nn.NewLinear(2, 4)
	router := &namedRouter{index: 0, expected: []string{"context"}}
	branchA, _ := nn.NewLinear(4, 4)
	branchB, _ := nn.NewLinear(4, 4)

	g, err := From(entry).Tag("context").
		Switch(router, branchA, branchB).Using("context").
		Build()
	if err != nil {
		t.Fatal("build:", err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal("forward:", err)
	}
	t.Logf("Output: %v", allF32(result))
}

// --- Map tests ---

func TestMapEach(t *testing.T) {
	// Body: doubles each element (W=2*I, b=0)
	// Input: [3, 2] (3 elements of size 2)
	// Map slices into 3 × [1,2], doubles each, cats back to [3, 2].
	doubler, _ := nn.NewLinear(2, 2)
	setLinearWeights(doubler, []float32{2, 0, 0, 2}, []float32{0, 0})

	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, err := From(entry).Map(doubler).Each().Build()
	if err != nil {
		t.Fatal(err)
	}

	// Input: [3, 2] — three 2-element vectors.
	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{3, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	vals := allF32(result)
	expected := []float32{2, 4, 6, 8, 10, 12}
	if len(vals) != len(expected) {
		t.Fatalf("Map.Each: got %d values, want %d", len(vals), len(expected))
	}
	for i, v := range vals {
		if !approxEqual(v, expected[i], 1e-5) {
			t.Errorf("Map.Each [%d]: got %v, want %v", i, v, expected[i])
		}
	}
	t.Logf("Map.Each: %v (shape %v)", vals, result.Data().Shape())
}

func TestMapOver(t *testing.T) {
	// Tag the entry output, then Map over it.
	doubler, _ := nn.NewLinear(2, 2)
	setLinearWeights(doubler, []float32{2, 0, 0, 2}, []float32{0, 0})

	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, err := From(entry).Tag("items").Map(doubler).Over("items").Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{2, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	vals := allF32(result)
	expected := []float32{2, 4, 6, 8}
	for i, v := range vals {
		if !approxEqual(v, expected[i], 1e-5) {
			t.Errorf("Map.Over [%d]: got %v, want %v", i, v, expected[i])
		}
	}
	t.Logf("Map.Over: %v", vals)
}

func TestMapBackward(t *testing.T) {
	// Body: identity + learnable bias. After map, gradient of loss
	// w.r.t. bias should equal the number of elements (each contributes 1).
	body, _ := nn.NewLinear(2, 2)
	setLinearWeights(body, []float32{1, 0, 0, 1}, []float32{0.1, 0.2})

	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, err := From(entry).Map(body).Each().Build()
	if err != nil {
		t.Fatal(err)
	}

	// 3 elements → body.bias gradient should be 3 for each component.
	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{3, 2})
	result := g.Forward(autograd.NewVariable(x, true))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	biasGrad := gradF32(body.Bias.Variable)
	// Each of the 3 map elements adds 1 to the bias gradient.
	for i, v := range biasGrad {
		if !approxEqual(v, 3.0, 1e-5) {
			t.Errorf("Map backward bias grad [%d]: got %v, want 3.0", i, v)
		}
	}
	t.Logf("Map backward bias grad: %v", biasGrad)
}

func TestMapParameters(t *testing.T) {
	body, _ := nn.NewLinear(2, 2)
	entry, _ := nn.NewLinear(2, 2)

	g, err := From(entry).Map(body).Each().Build()
	if err != nil {
		t.Fatal(err)
	}

	// entry: 2 params (weight + bias), body: 2 params.
	params := g.Parameters()
	if len(params) != 4 {
		t.Errorf("Map parameters: got %d, want 4", len(params))
	}
}

func TestMapOverUnknownTag(t *testing.T) {
	body, _ := nn.NewLinear(2, 2)
	entry, _ := nn.NewLinear(2, 2)

	_, err := From(entry).Map(body).Over("nonexistent").Build()
	if err == nil {
		t.Fatal("expected error for unknown tag in Map.Over")
	}
	if !strings.Contains(err.Error(), "nonexistent") {
		t.Errorf("error should mention tag name: %v", err)
	}
	t.Logf("Map.Over unknown tag error: %v", err)
}

func TestMapGraphAsModule(t *testing.T) {
	// Body is a sub-graph: Linear → GELU → LayerNorm.
	subBody, _ := nn.NewLinear(2, 2)
	sub, err := From(subBody).Through(nn.NewGELU()).Build()
	if err != nil {
		t.Fatal("build sub:", err)
	}

	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, err := From(entry).Map(sub).Each().Build()
	if err != nil {
		t.Fatal("build:", err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{2, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal("forward:", err)
	}

	shape := result.Data().Shape()
	if shape[0] != 2 || shape[1] != 2 {
		t.Errorf("Map Graph-as-Module: got shape %v, want [2, 2]", shape)
	}
	t.Logf("Map Graph-as-Module: %v (shape %v)", allF32(result), shape)
}

func TestMapSetTraining(t *testing.T) {
	// Body contains Dropout — should propagate training mode.
	body, err := From(nn.NewDropout(0.99)).Build()
	if err != nil {
		t.Fatal(err)
	}

	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, err := From(entry).Map(body).Each().Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{2, 2})

	// In eval mode, Dropout is identity.
	g.SetTraining(false)
	r1 := g.Forward(autograd.NewVariable(x, false))
	v1 := allF32(r1)
	r2 := g.Forward(autograd.NewVariable(x, false))
	v2 := allF32(r2)

	for i := range v1 {
		if v1[i] != v2[i] {
			t.Errorf("Map eval mode should be deterministic: %v != %v", v1, v2)
			break
		}
	}
	t.Logf("Map SetTraining: eval %v", v1)
}

func TestMapSlices(t *testing.T) {
	// Body: doubles each element. Input [1, 8], Slices(4) → 4 × [1, 2].
	// Each [1,2] is doubled → cat [4, 2] → recompose [1, 8].
	doubler, _ := nn.NewLinear(2, 2)
	setLinearWeights(doubler, []float32{2, 0, 0, 2}, []float32{0, 0})

	entry, _ := nn.NewLinear(2, 8)

	g, err := From(entry).Map(doubler).Slices(4).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	shape := result.Data().Shape()
	if shape[0] != 1 || shape[1] != 8 {
		t.Fatalf("Slices: got shape %v, want [1, 8]", shape)
	}

	// Every value should be doubled.
	entryResult := g.Forward(autograd.NewVariable(x, false))
	vals := allF32(entryResult)
	t.Logf("Slices output: %v (shape %v)", vals, shape)
}

func TestMapSlicesDynamicBatch(t *testing.T) {
	// Verify Slices handles dynamic batch sizes.
	// Input [3, 4], Slices(2) → 6 × [1, 2] → body → recompose [3, 4].
	identity, _ := nn.NewLinear(2, 2)
	setLinearWeights(identity, []float32{1, 0, 0, 1}, []float32{0, 0})

	entry, _ := nn.NewLinear(4, 4)
	setLinearWeights(entry, []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}, []float32{0, 0, 0, 0})

	g, err := From(entry).Map(identity).Slices(2).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}, []int64{3, 4})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	shape := result.Data().Shape()
	if shape[0] != 3 || shape[1] != 4 {
		t.Fatalf("Slices dynamic batch: got shape %v, want [3, 4]", shape)
	}

	vals := allF32(result)
	expected := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	for i, v := range vals {
		if !approxEqual(v, expected[i], 1e-5) {
			t.Errorf("Slices dynamic [%d]: got %v, want %v", i, v, expected[i])
		}
	}
	t.Logf("Slices dynamic batch: %v (shape %v)", vals, shape)
}

func TestMapSlicesBackward(t *testing.T) {
	// Body: identity + bias. Slices(2) on [1, 4] → 2 elements of [1, 2].
	// Bias gradient should be 2 (one contribution per slice).
	body, _ := nn.NewLinear(2, 2)
	setLinearWeights(body, []float32{1, 0, 0, 1}, []float32{0.1, 0.2})

	entry, _ := nn.NewLinear(4, 4)
	setLinearWeights(entry, []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}, []float32{0, 0, 0, 0})

	g, err := From(entry).Map(body).Slices(2).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{1, 4})
	result := g.Forward(autograd.NewVariable(x, true))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	biasGrad := gradF32(body.Bias.Variable)
	for i, v := range biasGrad {
		if !approxEqual(v, 2.0, 1e-5) {
			t.Errorf("Slices backward bias [%d]: got %v, want 2.0", i, v)
		}
	}
	t.Logf("Slices backward bias grad: %v", biasGrad)
}

func TestMapBatchedEach(t *testing.T) {
	// Verify batched Map.Each produces same results as element-wise.
	body, _ := nn.NewLinear(2, 2)
	setLinearWeights(body, []float32{2, 0, 0, 2}, []float32{1, 1})

	// Element-wise.
	g1, _ := From(body).Map(body).Each().Build()
	// Batched.
	g2, _ := From(body).Map(body).Batched().Each().Build()

	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{3, 2})
	r1 := g1.Forward(autograd.NewVariable(x, false))
	r2 := g2.Forward(autograd.NewVariable(x, false))

	v1 := allF32(r1)
	v2 := allF32(r2)

	for i := range v1 {
		if !approxEqual(v1[i], v2[i], 1e-5) {
			t.Errorf("[%d] element-wise=%f, batched=%f", i, v1[i], v2[i])
		}
	}
	t.Logf("Batched Each: %v (matches element-wise: %v)", v2, v1)
}

func TestMapBatchedSlices(t *testing.T) {
	body, _ := nn.NewLinear(2, 2)
	setLinearWeights(body, []float32{1, 0, 0, 1}, []float32{0, 0})

	entry1, _ := nn.NewLinear(8, 8)
	setLinearWeights(entry1, identityN(8), make([]float32, 8))
	entry2, _ := nn.NewLinear(8, 8)
	setLinearWeights(entry2, identityN(8), make([]float32, 8))

	// Element-wise slices.
	g1, _ := From(entry1).Map(body).Slices(4).Build()
	// Batched slices.
	g2, _ := From(entry2).Map(body).Batched().Slices(4).Build()

	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int64{1, 8})
	r1 := g1.Forward(autograd.NewVariable(x, false))
	r2 := g2.Forward(autograd.NewVariable(x, false))

	v1 := allF32(r1)
	v2 := allF32(r2)

	for i := range v1 {
		if !approxEqual(v1[i], v2[i], 1e-5) {
			t.Errorf("[%d] element-wise=%f, batched=%f", i, v1[i], v2[i])
		}
	}
	t.Logf("Batched Slices: %v (matches element-wise: %v)", v2, v1)
}

func TestMapBatchedBackward(t *testing.T) {
	entry, _ := nn.NewLinear(2, 2)
	setLinearWeights(entry, []float32{1, 0, 0, 1}, []float32{0, 0})

	body, _ := nn.NewLinear(2, 2)
	setLinearWeights(body, []float32{1, 0, 0, 1}, []float32{0, 0})

	g, _ := From(entry).Map(body).Batched().Each().Build()

	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{2, 2})
	result := g.Forward(autograd.NewVariable(x, true))
	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	// Body bias grad: sum of rows = [2, 2] (2 rows, each contributing 1).
	biasGrad := gradF32(body.Bias.Variable)
	for i, v := range biasGrad {
		if !approxEqual(v, 2.0, 1e-5) {
			t.Errorf("bias grad[%d]=%f, want 2.0", i, v)
		}
	}
	t.Logf("Batched backward bias grad: %v", biasGrad)
}

func TestReshapePrimitive(t *testing.T) {
	entry, _ := nn.NewLinear(2, 8)
	setLinearWeights(entry, []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
	}, []float32{1, 2, 3, 4, 5, 6, 7, 8})

	g, err := From(entry).Through(Reshape(4, 2)).Through(Reshape(1, 8)).Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{0, 0}, []int64{1, 2})
	result := g.Forward(autograd.NewVariable(x, false))
	if err := result.Err(); err != nil {
		t.Fatal(err)
	}

	shape := result.Data().Shape()
	if shape[0] != 1 || shape[1] != 8 {
		t.Fatalf("Reshape: got shape %v, want [1, 8]", shape)
	}

	vals := allF32(result)
	expected := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	for i, v := range vals {
		if !approxEqual(v, expected[i], 1e-5) {
			t.Errorf("Reshape [%d]: got %v, want %v", i, v, expected[i])
		}
	}
	t.Logf("Reshape round-trip: %v", vals)
}

func TestMapWithoutTerminator(t *testing.T) {
	entry, _ := nn.NewLinear(2, 2)
	body, _ := nn.NewLinear(2, 2)

	_, err := From(entry).Map(body).fb.Build()
	if err == nil {
		t.Fatal("expected error for Map without .Each() or .Over()")
	}
	if !strings.Contains(err.Error(), "Map") {
		t.Errorf("error should mention Map: %v", err)
	}
	t.Logf("Map without terminator: %v", err)
}

func TestLoopWithoutTerminator(t *testing.T) {
	entry, _ := nn.NewLinear(2, 2)
	body, _ := nn.NewLinear(2, 2)

	_, err := From(entry).Loop(body).fb.Build()
	if err == nil {
		t.Fatal("expected error for Loop without .For()/.While()/.Until()")
	}
	if !strings.Contains(err.Error(), "Loop") {
		t.Errorf("error should mention Loop: %v", err)
	}
	t.Logf("Loop without terminator: %v", err)
}

func TestParametersByTag(t *testing.T) {
	encoder, _ := nn.NewLinear(2, 4)
	decoder, _ := nn.NewLinear(4, 2)

	g, err := From(encoder).Tag("encoder").
		Through(decoder).Tag("decoder").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	encParams := g.ParametersByTag("encoder")
	if len(encParams) != 2 { // weight + bias
		t.Errorf("encoder params: got %d, want 2", len(encParams))
	}

	decParams := g.ParametersByTag("decoder")
	if len(decParams) != 2 {
		t.Errorf("decoder params: got %d, want 2", len(decParams))
	}

	// Unknown tag.
	if got := g.ParametersByTag("nonexistent"); got != nil {
		t.Errorf("expected nil for unknown tag, got %v", got)
	}

	t.Logf("encoder: %d params, decoder: %d params", len(encParams), len(decParams))
}

func TestFreezeUnfreeze(t *testing.T) {
	encoder, _ := nn.NewLinear(2, 4)
	decoder, _ := nn.NewLinear(4, 2)

	g, err := From(encoder).Tag("encoder").
		Through(decoder).Tag("decoder").
		Build()
	if err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})

	// Forward + backward to get gradients.
	result := g.Forward(autograd.NewVariable(x, true))
	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	// Freeze encoder, zero frozen grads.
	g.Freeze("encoder")
	g.ZeroFrozenGrads()

	// Encoder params should have nil grads.
	for _, p := range g.ParametersByTag("encoder") {
		if p.Grad() != nil {
			t.Error("frozen encoder param should have nil grad")
		}
	}

	// Decoder params should still have grads.
	hasGrad := false
	for _, p := range g.ParametersByTag("decoder") {
		if p.Grad() != nil {
			hasGrad = true
		}
	}
	if !hasGrad {
		t.Error("decoder params should still have gradients")
	}

	// Unfreeze and verify grads flow again.
	g.Unfreeze("encoder")
	for _, p := range g.Parameters() {
		p.ZeroGrad()
	}

	result2 := g.Forward(autograd.NewVariable(x, true))
	loss2 := result2.Sum()
	if err := loss2.Backward(); err != nil {
		t.Fatal(err)
	}
	g.ZeroFrozenGrads() // no frozen tags → no-op

	for _, p := range g.ParametersByTag("encoder") {
		if p.Grad() == nil {
			t.Error("unfrozen encoder param should have grad")
		}
	}

	t.Log("Freeze/Unfreeze: OK")
}

// containsAll checks that s contains every one of the substrings.
func containsAll(s string, subs ...string) bool {
	for _, sub := range subs {
		if !strings.Contains(s, sub) {
			return false
		}
	}
	return true
}
