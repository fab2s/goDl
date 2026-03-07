package nn_test

import (
	"math"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

const gcEps = 1e-3
const gcAtol = 1e-2

// moduleGradCheck verifies gradients of an nn.Module numerically.
// Checks both input gradients (for float inputs) and parameter gradients.
// f must return a scalar loss.
func moduleGradCheck(t *testing.T, name string, m nn.Module,
	inputs []*tensor.Tensor,
	f func(nn.Module, []*autograd.Variable) *autograd.Variable,
	atol float64) {
	t.Helper()

	for _, p := range m.Parameters() {
		p.ZeroGrad()
	}

	// Analytical forward + backward.
	vars := make([]*autograd.Variable, len(inputs))
	for i, inp := range inputs {
		vars[i] = autograd.NewVariable(inp, true)
	}
	loss := f(m, vars)
	if err := loss.Err(); err != nil {
		t.Fatalf("%s: forward error: %v", name, err)
	}
	if err := loss.Backward(); err != nil {
		t.Fatalf("%s: backward error: %v", name, err)
	}

	// Check input gradients (skip non-float inputs like Int64 indices).
	for vi, inp := range inputs {
		origData, err := inp.Float32Data()
		if err != nil {
			continue
		}
		grad := vars[vi].Grad()
		if grad == nil {
			continue
		}
		analyticalData, _ := grad.Float32Data()
		shape := inp.Shape()
		numel := len(origData)

		for ei := 0; ei < numel; ei++ {
			plus := make([]float32, numel)
			copy(plus, origData)
			plus[ei] += float32(gcEps)
			tPlus, _ := tensor.FromFloat32(plus, shape)

			minus := make([]float32, numel)
			copy(minus, origData)
			minus[ei] -= float32(gcEps)
			tMinus, _ := tensor.FromFloat32(minus, shape)

			var fPlus, fMinus float64
			autograd.NoGrad(func() {
				vp := gcReplaceVar(inputs, vi, tPlus)
				fPlus = gcScalar(t, name, f(m, vp))
				vm := gcReplaceVar(inputs, vi, tMinus)
				fMinus = gcScalar(t, name, f(m, vm))
			})

			numerical := (fPlus - fMinus) / (2 * gcEps)
			analytical := float64(analyticalData[ei])
			if math.Abs(numerical-analytical) > atol {
				t.Errorf("%s: input %d elem %d: analytical=%.6f numerical=%.6f (diff=%.6f)",
					name, vi, ei, analytical, numerical, math.Abs(numerical-analytical))
			}
		}
	}

	// Check parameter gradients.
	for pi, p := range m.Parameters() {
		grad := p.Grad()
		if grad == nil {
			t.Errorf("%s: param %d (%s) has nil gradient", name, pi, p.Name)
			continue
		}
		analyticalData, _ := grad.Float32Data()
		origData, _ := p.Data().Float32Data()
		origTensor := p.Data()
		shape := p.Data().Shape()
		numel := len(origData)

		for ei := 0; ei < numel; ei++ {
			plus := make([]float32, numel)
			copy(plus, origData)
			plus[ei] += float32(gcEps)
			tPlus, _ := tensor.FromFloat32(plus, shape)

			var fPlus float64
			p.SetData(tPlus)
			autograd.NoGrad(func() {
				fPlus = gcScalar(t, name, f(m, gcWrap(inputs)))
			})

			minus := make([]float32, numel)
			copy(minus, origData)
			minus[ei] -= float32(gcEps)
			tMinus, _ := tensor.FromFloat32(minus, shape)

			var fMinus float64
			p.SetData(tMinus)
			autograd.NoGrad(func() {
				fMinus = gcScalar(t, name, f(m, gcWrap(inputs)))
			})

			p.SetData(origTensor)

			numerical := (fPlus - fMinus) / (2 * gcEps)
			analytical := float64(analyticalData[ei])
			if math.Abs(numerical-analytical) > atol {
				t.Errorf("%s: param %d (%s) elem %d: analytical=%.6f numerical=%.6f (diff=%.6f)",
					name, pi, p.Name, ei, analytical, numerical, math.Abs(numerical-analytical))
			}
		}
	}
}

func gcWrap(inputs []*tensor.Tensor) []*autograd.Variable {
	vars := make([]*autograd.Variable, len(inputs))
	for i, inp := range inputs {
		vars[i] = autograd.NewVariable(inp, false)
	}
	return vars
}

func gcReplaceVar(inputs []*tensor.Tensor, idx int, replacement *tensor.Tensor) []*autograd.Variable {
	vars := make([]*autograd.Variable, len(inputs))
	for i, inp := range inputs {
		if i == idx {
			vars[i] = autograd.NewVariable(replacement, false)
		} else {
			vars[i] = autograd.NewVariable(inp, false)
		}
	}
	return vars
}

func gcScalar(t *testing.T, name string, v *autograd.Variable) float64 {
	t.Helper()
	if err := v.Err(); err != nil {
		t.Fatalf("%s: forward error in gradcheck: %v", name, err)
	}
	data, err := v.Data().Float32Data()
	if err != nil {
		t.Fatalf("%s: scalar data: %v", name, err)
	}
	if len(data) != 1 {
		t.Fatalf("%s: expected scalar, got %d elements", name, len(data))
	}
	return float64(data[0])
}

func sumLoss(m nn.Module, v []*autograd.Variable) *autograd.Variable {
	return m.Forward(v...).Sum()
}

// --- Module gradient checks ---

func TestModuleGradCheckLinear(t *testing.T) {
	m, err := nn.NewLinear(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	x, _ := tensor.FromFloat32([]float32{0.5, -0.3, 0.8, -0.2, 0.6, 0.1}, []int64{2, 3})
	moduleGradCheck(t, "Linear", m, []*tensor.Tensor{x}, sumLoss, gcAtol)
}

func TestModuleGradCheckLayerNorm(t *testing.T) {
	m, err := nn.NewLayerNorm(4)
	if err != nil {
		t.Fatal(err)
	}
	x, _ := tensor.FromFloat32([]float32{
		1.0, 2.0, 3.0, 4.0,
		-1.0, 0.5, 1.5, 3.0,
	}, []int64{2, 4})
	moduleGradCheck(t, "LayerNorm", m, []*tensor.Tensor{x}, sumLoss, gcAtol)
}

func TestModuleGradCheckBatchNorm(t *testing.T) {
	m, err := nn.NewBatchNorm(3)
	if err != nil {
		t.Fatal(err)
	}
	x, _ := tensor.FromFloat32([]float32{
		1.0, 2.0, 3.0,
		-1.0, 0.0, 1.0,
		0.5, 1.5, 2.5,
	}, []int64{3, 3})
	moduleGradCheck(t, "BatchNorm", m, []*tensor.Tensor{x}, sumLoss, gcAtol)
}

func TestModuleGradCheckConv2d(t *testing.T) {
	m, err := nn.NewConv2d(1, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	// Small values to keep float32 numerical precision tight.
	x, _ := tensor.FromFloat32([]float32{
		0.1, 0.2, 0.3, 0.4,
		0.5, 0.6, 0.7, 0.8,
		0.9, 0.1, 0.2, 0.3,
		0.4, 0.5, 0.6, 0.7,
	}, []int64{1, 1, 4, 4})
	moduleGradCheck(t, "Conv2d", m, []*tensor.Tensor{x}, sumLoss, gcAtol)
}

func TestModuleGradCheckGRUCell(t *testing.T) {
	m, err := nn.NewGRUCell(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	x, _ := tensor.FromFloat32([]float32{0.5, -0.3, 0.2, 0.8}, []int64{2, 2})
	h, _ := tensor.FromFloat32([]float32{0.1, -0.1, 0.3, 0.2}, []int64{2, 2})
	moduleGradCheck(t, "GRUCell", m, []*tensor.Tensor{x, h}, sumLoss, gcAtol)
}

func TestModuleGradCheckGRUCellNilHidden(t *testing.T) {
	m, err := nn.NewGRUCell(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	x, _ := tensor.FromFloat32([]float32{0.5, -0.3, 0.2, 0.8}, []int64{2, 2})
	moduleGradCheck(t, "GRUCell_nil_h", m, []*tensor.Tensor{x}, sumLoss, gcAtol)
}

func TestModuleGradCheckLSTMCell(t *testing.T) {
	m, err := nn.NewLSTMCell(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	x, _ := tensor.FromFloat32([]float32{0.5, -0.3, 0.2, 0.8}, []int64{2, 2})
	// State = cat(h, c) → [batch, 2*hiddenSize] = [2, 4]
	state, _ := tensor.FromFloat32([]float32{
		0.1, -0.1, 0.2, 0.3,
		0.3, 0.2, -0.1, 0.1,
	}, []int64{2, 4})
	moduleGradCheck(t, "LSTMCell", m, []*tensor.Tensor{x, state}, sumLoss, gcAtol)
}

func TestModuleGradCheckLSTMCellNilState(t *testing.T) {
	m, err := nn.NewLSTMCell(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	x, _ := tensor.FromFloat32([]float32{0.5, -0.3, 0.2, 0.8}, []int64{2, 2})
	moduleGradCheck(t, "LSTMCell_nil_state", m, []*tensor.Tensor{x}, sumLoss, gcAtol)
}

func TestModuleGradCheckEmbedding(t *testing.T) {
	m, err := nn.NewEmbedding(5, 3)
	if err != nil {
		t.Fatal(err)
	}
	// Int64 indices — input gradcheck is skipped (non-float), only param gradients checked.
	idx, _ := tensor.FromInt64([]int64{0, 2, 4, 1}, []int64{2, 2})
	moduleGradCheck(t, "Embedding", m, []*tensor.Tensor{idx}, sumLoss, gcAtol)
}

// CrossEntropyLoss has a MaxDim detachment that's worth verifying.
func TestModuleGradCheckCrossEntropyLoss(t *testing.T) {
	pred, _ := tensor.FromFloat32([]float32{
		2.0, 1.0, 0.1,
		0.5, 2.5, 1.0,
	}, []int64{2, 3})
	target, _ := tensor.FromFloat32([]float32{
		1, 0, 0,
		0, 1, 0,
	}, []int64{2, 3})

	// Analytical.
	pVar := autograd.NewVariable(pred, true)
	tVar := autograd.NewVariable(target, false)
	loss := nn.CrossEntropyLoss(pVar, tVar)
	if err := loss.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	analyticalData, _ := pVar.Grad().Float32Data()
	origData, _ := pred.Float32Data()
	shape := pred.Shape()
	numel := len(origData)

	for ei := 0; ei < numel; ei++ {
		plus := make([]float32, numel)
		copy(plus, origData)
		plus[ei] += float32(gcEps)
		tPlus, _ := tensor.FromFloat32(plus, shape)

		minus := make([]float32, numel)
		copy(minus, origData)
		minus[ei] -= float32(gcEps)
		tMinus, _ := tensor.FromFloat32(minus, shape)

		var fPlus, fMinus float64
		autograd.NoGrad(func() {
			pv := autograd.NewVariable(tPlus, false)
			tv := autograd.NewVariable(target, false)
			fPlus = gcScalar(t, "CE", nn.CrossEntropyLoss(pv, tv))

			pm := autograd.NewVariable(tMinus, false)
			fMinus = gcScalar(t, "CE", nn.CrossEntropyLoss(pm, tv))
		})

		numerical := (fPlus - fMinus) / (2 * gcEps)
		analytical := float64(analyticalData[ei])
		if math.Abs(numerical-analytical) > gcAtol {
			t.Errorf("CrossEntropyLoss: elem %d: analytical=%.6f numerical=%.6f (diff=%.6f)",
				ei, analytical, numerical, math.Abs(numerical-analytical))
		}
	}
}
