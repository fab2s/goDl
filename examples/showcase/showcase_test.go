package showcase

import (
	"os"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

func allF32(v *autograd.Variable) []float32 {
	data, _ := v.Data().Float32Data()
	return data
}

func makeInput() *autograd.Variable {
	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	return autograd.NewVariable(x, false)
}

func makeInputGrad() *autograd.Variable {
	x, _ := tensor.FromFloat32([]float32{1, 2}, []int64{1, 2})
	return autograd.NewVariable(x, true)
}

func TestBuild(t *testing.T) {
	g, err := BuildShowcase()
	if err != nil {
		t.Fatal("build:", err)
	}

	result := g.Forward(makeInput())
	if err := result.Err(); err != nil {
		t.Fatal("forward:", err)
	}

	vals := allF32(result)
	if len(vals) != 2 {
		t.Fatalf("expected 2 outputs, got %d", len(vals))
	}
	t.Logf("Output: %v (shape %v)", vals, result.Data().Shape())
}

func TestForwardRefCarriesState(t *testing.T) {
	g, err := BuildShowcase()
	if err != nil {
		t.Fatal(err)
	}

	r1 := g.Forward(makeInput())
	if err := r1.Err(); err != nil {
		t.Fatal("pass 1:", err)
	}
	v1 := allF32(r1)

	r2 := g.Forward(makeInput())
	if err := r2.Err(); err != nil {
		t.Fatal("pass 2:", err)
	}
	v2 := allF32(r2)

	// Memory state from pass 1 should make pass 2 different.
	same := true
	for i := range v1 {
		if v1[i] != v2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("pass 2 should differ from pass 1 (forward ref memory)")
	}
	t.Logf("Pass 1: %v, Pass 2: %v", v1, v2)
}

func TestResetState(t *testing.T) {
	g, err := BuildShowcase()
	if err != nil {
		t.Fatal(err)
	}

	// Eval mode for deterministic comparison (Dropout is identity).
	g.SetTraining(false)

	r1 := g.Forward(makeInput())
	v1 := allF32(r1)

	// Advance state.
	g.Forward(makeInput())

	// Reset and verify we're back to pass-1 behavior.
	g.ResetState()
	r3 := g.Forward(makeInput())
	v3 := allF32(r3)

	for i := range v1 {
		if v1[i] != v3[i] {
			t.Errorf("[%d] after reset: %v != %v", i, v3[i], v1[i])
		}
	}
	t.Logf("After ResetState: %v matches pass 1: %v", v3, v1)
}

func TestDetachState(t *testing.T) {
	g, err := BuildShowcase()
	if err != nil {
		t.Fatal(err)
	}

	g.Forward(makeInput()) // build state
	g.DetachState()        // break gradient chain

	result := g.Forward(makeInput())
	if err := result.Err(); err != nil {
		t.Fatal("forward after detach:", err)
	}
	t.Logf("After DetachState: %v", allF32(result))
}

func TestBackward(t *testing.T) {
	g, err := BuildShowcase()
	if err != nil {
		t.Fatal(err)
	}

	result := g.Forward(makeInputGrad())
	if err := result.Err(); err != nil {
		t.Fatal("forward:", err)
	}

	loss := result.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatal("backward:", err)
	}

	// Spot-check that parameters received gradients.
	params := g.Parameters()
	withGrad := 0
	for _, p := range params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		data, _ := grad.Float32Data()
		for _, v := range data {
			if v != 0 {
				withGrad++
				break
			}
		}
	}
	t.Logf("Parameters with non-zero gradients: %d/%d", withGrad, len(params))
	if withGrad == 0 {
		t.Error("no parameters received gradients")
	}
}

func TestParameters(t *testing.T) {
	g, err := BuildShowcase()
	if err != nil {
		t.Fatal(err)
	}

	params := g.Parameters()
	// Top-level (12 × 2 params each = 24):
	//   embed + LayerNorm + residual + softmaxRouter
	//   + expertA + expertB + switchBranchA
	//   + mapEachLinear + mapOverLinear
	//   + whileBody + untilBody + head
	// Sub-graphs:
	//   Split readHead×2 (h=8): Linear(2) + LayerNorm(2) = 4 each = 8
	//   Map.Slices readHead×1 (dim=2): Linear(2) + LayerNorm(2) = 4
	//   ffnBlock×2 (loopFor + switchBranchB): Linear(2) + GELU(0) + LayerNorm(2) = 4 each = 8
	// heavyPathSelector: 0, Reshape×2: 0
	// Total: 24 + 8 + 4 + 8 = 44
	if len(params) != 44 {
		t.Errorf("expected 44 parameters, got %d", len(params))
	}
	t.Logf("Total parameters: %d", len(params))

	// All should be distinct (no accidental sharing).
	seen := make(map[*nn.Parameter]bool)
	for _, p := range params {
		if seen[p] {
			t.Error("duplicate parameter detected")
		}
		seen[p] = true
	}
}

func TestSetTraining(t *testing.T) {
	g, err := BuildShowcase()
	if err != nil {
		t.Fatal(err)
	}

	g.SetTraining(false)
	result := g.Forward(makeInput())
	if err := result.Err(); err != nil {
		t.Fatal("eval forward:", err)
	}

	g.SetTraining(true)
	result2 := g.Forward(makeInput())
	if err := result2.Err(); err != nil {
		t.Fatal("train forward:", err)
	}

	t.Logf("Eval: %v, Train: %v", allF32(result), allF32(result2))
}

func TestDOT(t *testing.T) {
	g, err := BuildShowcase()
	if err != nil {
		t.Fatal(err)
	}

	// Write DOT file.
	dot := g.DOT()
	if len(dot) == 0 {
		t.Fatal("DOT output is empty")
	}
	if err := os.WriteFile("showcase.dot", []byte(dot), 0600); err != nil {
		t.Fatal("write dot:", err)
	}
	t.Logf("Wrote showcase.dot (%d bytes)", len(dot))

	// Render SVG via g.SVG(path).
	svg, err := g.SVG("showcase.svg")
	if err != nil {
		t.Skip("SVG:", err)
	}
	t.Logf("Wrote showcase.svg (%d bytes)", len(svg))
}
