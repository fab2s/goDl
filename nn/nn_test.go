package nn_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

const eps = 1e-4

func mustData(t *testing.T, ts *tensor.Tensor) []float32 {
	t.Helper()
	data, err := ts.Float32Data()
	if err != nil {
		t.Fatalf("Float32Data: %v", err)
	}
	return data
}

// --- Linear layer ---

func TestLinearForward(t *testing.T) {
	linear, err := nn.NewLinear(3, 2)
	if err != nil {
		t.Fatalf("NewLinear: %v", err)
	}

	// Check parameter shapes
	params := linear.Parameters()
	if len(params) != 2 {
		t.Fatalf("expected 2 parameters, got %d", len(params))
	}

	wShape := params[0].Data().Shape()
	if wShape[0] != 2 || wShape[1] != 3 {
		t.Errorf("weight shape = %v, want [2 3]", wShape)
	}
	bShape := params[1].Data().Shape()
	if bShape[0] != 2 {
		t.Errorf("bias shape = %v, want [2]", bShape)
	}

	// Forward pass
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{1, 3})
	defer xt.Release()
	x := autograd.NewVariable(xt, false)

	out := linear.Forward(x)
	if err := out.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	outShape := out.Data().Shape()
	if outShape[0] != 1 || outShape[1] != 2 {
		t.Errorf("output shape = %v, want [1 2]", outShape)
	}
}

func TestLinearBackward(t *testing.T) {
	linear, err := nn.NewLinear(3, 2)
	if err != nil {
		t.Fatalf("NewLinear: %v", err)
	}

	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{1, 3})
	defer xt.Release()
	x := autograd.NewVariable(xt, true)

	out := linear.Forward(x)
	loss := out.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	// All parameters should have gradients
	for _, p := range linear.Parameters() {
		if p.Grad() == nil {
			t.Errorf("parameter %s has nil gradient", p.Name)
		}
	}
	// Input should have gradient too
	if x.Grad() == nil {
		t.Error("input x has nil gradient")
	}
}

// --- MSE Loss ---

func TestMSELoss(t *testing.T) {
	// pred = [1, 2, 3], target = [1, 2, 3] → loss = 0
	pt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer pt.Release()
	tt2, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer tt2.Release()

	pred := autograd.NewVariable(pt, true)
	target := autograd.NewVariable(tt2, false)

	loss := nn.MSELoss(pred, target)
	lossData := mustData(t, loss.Data())
	if math.Abs(float64(lossData[0])) > eps {
		t.Errorf("MSE of identical tensors = %f, want 0", lossData[0])
	}

	// pred = [3, 2, 1], target = [1, 2, 3] → MSE = ((2² + 0 + 2²) / 3) = 8/3
	pt2, _ := tensor.FromFloat32([]float32{3, 2, 1}, []int64{3})
	defer pt2.Release()
	pred2 := autograd.NewVariable(pt2, true)

	loss2 := nn.MSELoss(pred2, target)
	lossData2 := mustData(t, loss2.Data())
	expected := float32(8.0 / 3.0)
	if math.Abs(float64(lossData2[0]-expected)) > eps {
		t.Errorf("MSE = %f, want %f", lossData2[0], expected)
	}

	// Backward should work
	if err := loss2.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}
	if pred2.Grad() == nil {
		t.Error("pred gradient is nil")
	}
}

// --- Cross-entropy Loss ---

func TestCrossEntropyLoss(t *testing.T) {
	// Simple 2-class case: logits = [2, 1], target = [1, 0] (class 0)
	// softmax([2,1]) = [exp(2)/(exp(2)+exp(1)), exp(1)/(exp(2)+exp(1))]
	//                = [0.7311, 0.2689]
	// CE = -log(0.7311) = 0.3133
	pt, _ := tensor.FromFloat32([]float32{2, 1}, []int64{1, 2})
	defer pt.Release()
	tt2, _ := tensor.FromFloat32([]float32{1, 0}, []int64{1, 2})
	defer tt2.Release()

	pred := autograd.NewVariable(pt, true)
	target := autograd.NewVariable(tt2, false)

	loss := nn.CrossEntropyLoss(pred, target)
	if err := loss.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}

	lossData := mustData(t, loss.Data())
	expected := float32(0.3133)
	if math.Abs(float64(lossData[0]-expected)) > 0.01 {
		t.Errorf("CrossEntropy = %f, want ~%f", lossData[0], expected)
	}

	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}
	if pred.Grad() == nil {
		t.Error("pred gradient is nil")
	}
}

// --- SGD Optimizer ---

func TestSGDStep(t *testing.T) {
	// Create a simple parameter and manually set gradient
	wt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer wt.Release()
	p := nn.NewParameter(wt, "w")

	// Simulate gradient = [0.1, 0.2, 0.3]
	gt, _ := tensor.FromFloat32([]float32{0.1, 0.2, 0.3}, []int64{3})
	defer gt.Release()

	// Do a forward+backward to set the gradient
	x := autograd.NewVariable(gt, false)
	loss := p.Variable.Mul(x).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	opt := nn.NewSGD([]*nn.Parameter{p}, 0.1, 0)
	opt.Step()

	// w_new = w - lr * grad = [1, 2, 3] - 0.1 * [0.1, 0.2, 0.3]
	//       = [0.99, 1.98, 2.97]
	data := mustData(t, p.Data())
	want := []float32{0.99, 1.98, 2.97}
	for i := range want {
		if math.Abs(float64(data[i]-want[i])) > eps {
			t.Errorf("w[%d] = %f, want %f", i, data[i], want[i])
		}
	}
}

// --- Adam Optimizer ---

func TestAdamStep(t *testing.T) {
	wt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer wt.Release()
	p := nn.NewParameter(wt, "w")

	gt, _ := tensor.FromFloat32([]float32{0.1, 0.2, 0.3}, []int64{3})
	defer gt.Release()
	x := autograd.NewVariable(gt, false)
	loss := p.Variable.Mul(x).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	opt := nn.NewAdam([]*nn.Parameter{p}, 0.01)
	opt.Step()

	// After one Adam step, parameters should have moved.
	data := mustData(t, p.Data())
	for i, orig := range []float32{1, 2, 3} {
		if math.Abs(float64(data[i]-orig)) < 1e-6 {
			t.Errorf("w[%d] unchanged after Adam step", i)
		}
	}
	// With default betas, first step moves each param by ~lr (≈0.01).
	for i, orig := range []float32{1, 2, 3} {
		diff := math.Abs(float64(data[i] - orig))
		if diff > 0.02 {
			t.Errorf("w[%d] moved too far: %f (diff=%f)", i, data[i], diff)
		}
	}
}

func TestAdamTraining(t *testing.T) {
	// Train y = 2x + 1 with Adam — should converge faster than SGD.
	linear, err := nn.NewLinear(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	opt := nn.NewAdam(linear.Parameters(), 0.05)

	xData, _ := tensor.FromFloat32([]float32{0, 0.25, 0.5, 0.75, 1.0}, []int64{5, 1})
	defer xData.Release()
	yData, _ := tensor.FromFloat32([]float32{1, 1.5, 2, 2.5, 3}, []int64{5, 1})
	defer yData.Release()
	target := autograd.NewVariable(yData, false)

	var finalLoss float32
	for epoch := range 300 {
		pred := linear.Forward(autograd.NewVariable(xData, false))
		loss := nn.MSELoss(pred, target)
		if err := loss.Backward(); err != nil {
			t.Fatalf("epoch %d: %v", epoch, err)
		}
		opt.Step()
		opt.ZeroGrad()
		finalLoss = mustData(t, loss.Data())[0]
	}

	if finalLoss > 0.01 {
		t.Errorf("Adam final loss = %f, want < 0.01", finalLoss)
	}

	testX, _ := tensor.FromFloat32([]float32{0.5}, []int64{1, 1})
	defer testX.Release()
	pred := linear.Forward(autograd.NewVariable(testX, false))
	predVal := mustData(t, pred.Data())[0]
	if math.Abs(float64(predVal-2.0)) > 0.1 {
		t.Errorf("Adam pred(0.5) = %f, want ~2.0", predVal)
	}
	t.Logf("Adam: loss=%.6f, pred(0.5)=%.4f", finalLoss, predVal)
}

func TestAdamWTraining(t *testing.T) {
	// AdamW with weight decay — should also converge, weights should be smaller.
	linear, err := nn.NewLinear(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	opt := nn.NewAdamW(linear.Parameters(), 0.05, 0.1)

	xData, _ := tensor.FromFloat32([]float32{0, 0.25, 0.5, 0.75, 1.0}, []int64{5, 1})
	defer xData.Release()
	yData, _ := tensor.FromFloat32([]float32{1, 1.5, 2, 2.5, 3}, []int64{5, 1})
	defer yData.Release()
	target := autograd.NewVariable(yData, false)

	var finalLoss float32
	for epoch := range 300 {
		pred := linear.Forward(autograd.NewVariable(xData, false))
		loss := nn.MSELoss(pred, target)
		if err := loss.Backward(); err != nil {
			t.Fatalf("epoch %d: %v", epoch, err)
		}
		opt.Step()
		opt.ZeroGrad()
		finalLoss = mustData(t, loss.Data())[0]
	}

	if finalLoss > 0.05 {
		t.Errorf("AdamW final loss = %f, want < 0.05", finalLoss)
	}
	t.Logf("AdamW: loss=%.6f", finalLoss)
}

// --- End-to-end training loop ---

func TestTrainingLoop(t *testing.T) {
	// Train a linear layer to learn y = 2*x + 1
	// Input: [0, 0.25, 0.5, 0.75, 1.0]
	// Target: [1, 1.5, 2, 2.5, 3]
	linear, err := nn.NewLinear(1, 1)
	if err != nil {
		t.Fatalf("NewLinear: %v", err)
	}
	opt := nn.NewSGD(linear.Parameters(), 0.1, 0.9)

	xData, _ := tensor.FromFloat32([]float32{0, 0.25, 0.5, 0.75, 1.0}, []int64{5, 1})
	defer xData.Release()
	yData, _ := tensor.FromFloat32([]float32{1, 1.5, 2, 2.5, 3}, []int64{5, 1})
	defer yData.Release()

	target := autograd.NewVariable(yData, false)
	var finalLoss float32

	for epoch := range 200 {
		x := autograd.NewVariable(xData, false)
		pred := linear.Forward(x)
		loss := nn.MSELoss(pred, target)

		if err := loss.Backward(); err != nil {
			t.Fatalf("epoch %d backward: %v", epoch, err)
		}

		opt.Step()
		opt.ZeroGrad()

		lossVal := mustData(t, loss.Data())
		finalLoss = lossVal[0]
	}

	// Loss should be very small after 200 epochs
	if finalLoss > 0.01 {
		t.Errorf("final loss = %f, expected < 0.01", finalLoss)
	}

	// Test prediction: x=0.5 should give ~2.0
	testX, _ := tensor.FromFloat32([]float32{0.5}, []int64{1, 1})
	defer testX.Release()
	pred := linear.Forward(autograd.NewVariable(testX, false))
	predData := mustData(t, pred.Data())
	if math.Abs(float64(predData[0]-2.0)) > 0.1 {
		t.Errorf("pred(0.5) = %f, want ~2.0", predData[0])
	}
}

// --- Activations ---

func TestGELU(t *testing.T) {
	gelu := nn.NewGELU()
	xt, _ := tensor.FromFloat32([]float32{-1, 0, 1, 2}, []int64{4})
	defer xt.Release()
	x := autograd.NewVariable(xt, true)

	out := gelu.Forward(x)
	data := mustData(t, out.Data())
	// GELU(-1) ≈ -0.1588, GELU(0) = 0, GELU(1) ≈ 0.8412, GELU(2) ≈ 1.9545
	want := []float32{-0.1588, 0, 0.8412, 1.9545}
	for i := range want {
		if math.Abs(float64(data[i]-want[i])) > 0.01 {
			t.Errorf("GELU[%d] = %f, want ~%f", i, data[i], want[i])
		}
	}

	// Backward should work
	loss := out.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}
	if x.Grad() == nil {
		t.Error("GELU gradient is nil")
	}
}

func TestSiLU(t *testing.T) {
	silu := nn.NewSiLU()
	xt, _ := tensor.FromFloat32([]float32{-1, 0, 1, 2}, []int64{4})
	defer xt.Release()
	x := autograd.NewVariable(xt, true)

	out := silu.Forward(x)
	data := mustData(t, out.Data())
	// SiLU(x) = x * sigmoid(x)
	// SiLU(-1) ≈ -0.2689, SiLU(0) = 0, SiLU(1) ≈ 0.7311, SiLU(2) ≈ 1.7616
	want := []float32{-0.2689, 0, 0.7311, 1.7616}
	for i := range want {
		if math.Abs(float64(data[i]-want[i])) > 0.01 {
			t.Errorf("SiLU[%d] = %f, want ~%f", i, data[i], want[i])
		}
	}

	loss := out.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}
	if x.Grad() == nil {
		t.Error("SiLU gradient is nil")
	}
}

func TestSoftmaxModule(t *testing.T) {
	sm := nn.NewSoftmax(1)
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3, 1, 2, 3}, []int64{2, 3})
	defer xt.Release()
	x := autograd.NewVariable(xt, true)

	out := sm.Forward(x)
	data := mustData(t, out.Data())
	// Each row should sum to 1
	for row := 0; row < 2; row++ {
		sum := float64(data[row*3]) + float64(data[row*3+1]) + float64(data[row*3+2])
		if math.Abs(sum-1.0) > eps {
			t.Errorf("row %d sum = %f, want 1.0", row, sum)
		}
	}
}

// --- Dropout ---

func TestDropoutTraining(t *testing.T) {
	drop := nn.NewDropout(0.5)
	xt, _ := tensor.FromFloat32([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, []int64{1, 10})
	defer xt.Release()
	x := autograd.NewVariable(xt, false)

	out := drop.Forward(x)
	if err := out.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	data := mustData(t, out.Data())

	// In inverted dropout with p=0.5, surviving elements are scaled by 2.
	// Each element should be either 0 or 2.
	zeros, twos := 0, 0
	for _, v := range data {
		switch {
		case math.Abs(float64(v)) < eps:
			zeros++
		case math.Abs(float64(v)-2.0) < eps:
			twos++
		default:
			t.Errorf("unexpected value %f (want 0 or 2)", v)
		}
	}
	if zeros == 0 || twos == 0 {
		t.Logf("warning: all zeros=%d or all twos=%d (possible but unlikely)", zeros, twos)
	}
}

func TestDropoutEval(t *testing.T) {
	drop := nn.NewDropout(0.5)
	drop.SetTraining(false)

	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer xt.Release()
	x := autograd.NewVariable(xt, false)

	out := drop.Forward(x)
	data := mustData(t, out.Data())
	// In eval mode, dropout is identity
	want := []float32{1, 2, 3}
	for i := range want {
		if math.Abs(float64(data[i]-want[i])) > eps {
			t.Errorf("eval dropout[%d] = %f, want %f", i, data[i], want[i])
		}
	}
}

func TestDropoutBackward(t *testing.T) {
	drop := nn.NewDropout(0.0) // p=0 means no dropout, all elements survive
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer xt.Release()
	x := autograd.NewVariable(xt, true)

	loss := drop.Forward(x).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}
	if x.Grad() == nil {
		t.Error("input gradient is nil")
	}
}

// --- Embedding ---

func TestEmbeddingForward(t *testing.T) {
	emb, err := nn.NewEmbedding(5, 3) // 5 words, 3-dim embeddings
	if err != nil {
		t.Fatal(err)
	}

	// Look up indices [0, 2, 4]
	idx, _ := tensor.FromInt64([]int64{0, 2, 4}, []int64{3})
	defer idx.Release()

	out := emb.Forward(autograd.NewVariable(idx, false))
	if err := out.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	shape := out.Data().Shape()
	if shape[0] != 3 || shape[1] != 3 {
		t.Errorf("shape = %v, want [3 3]", shape)
	}
}

func TestEmbedding2D(t *testing.T) {
	emb, err := nn.NewEmbedding(10, 4)
	if err != nil {
		t.Fatal(err)
	}

	// 2D indices: [batch=2, seq=3]
	idx, _ := tensor.FromInt64([]int64{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	defer idx.Release()

	out := emb.Forward(autograd.NewVariable(idx, false))
	if err := out.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	shape := out.Data().Shape()
	if shape[0] != 2 || shape[1] != 3 || shape[2] != 4 {
		t.Errorf("shape = %v, want [2 3 4]", shape)
	}
}

func TestEmbeddingBackward(t *testing.T) {
	emb, err := nn.NewEmbedding(5, 3)
	if err != nil {
		t.Fatal(err)
	}

	idx, _ := tensor.FromInt64([]int64{1, 1, 3}, []int64{3})
	defer idx.Release()

	loss := emb.Forward(autograd.NewVariable(idx, false)).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	// Weight gradient: rows 1 and 3 should have gradients, rest zero
	grad := mustData(t, emb.Weight.Grad())
	// Row 0: zeros
	for i := 0; i < 3; i++ {
		if math.Abs(float64(grad[i])) > eps {
			t.Errorf("grad[0][%d] = %f, want 0", i, grad[i])
		}
	}
	// Row 1: index 1 appears twice → grad = [2, 2, 2]
	for i := 3; i < 6; i++ {
		if math.Abs(float64(grad[i]-2.0)) > eps {
			t.Errorf("grad[1][%d] = %f, want 2", i-3, grad[i])
		}
	}
}

// --- LayerNorm ---

func TestLayerNormForward(t *testing.T) {
	ln, err := nn.NewLayerNorm(4)
	if err != nil {
		t.Fatal(err)
	}

	// Input: [2, 4]
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int64{2, 4})
	defer xt.Release()

	out := ln.Forward(autograd.NewVariable(xt, false))
	if err := out.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	shape := out.Data().Shape()
	if shape[0] != 2 || shape[1] != 4 {
		t.Errorf("shape = %v, want [2 4]", shape)
	}

	// After normalization, each row should have mean ≈ 0 and std ≈ 1
	data := mustData(t, out.Data())
	for row := 0; row < 2; row++ {
		var sum float64
		for i := 0; i < 4; i++ {
			sum += float64(data[row*4+i])
		}
		mean := sum / 4.0
		if math.Abs(mean) > 0.01 {
			t.Errorf("row %d mean = %f, want ~0", row, mean)
		}
	}
}

func TestLayerNormBackward(t *testing.T) {
	ln, err := nn.NewLayerNorm(3)
	if err != nil {
		t.Fatal(err)
	}

	xt, _ := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	defer xt.Release()
	x := autograd.NewVariable(xt, true)

	loss := ln.Forward(x).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	// All parameters and input should have gradients
	if x.Grad() == nil {
		t.Error("input gradient is nil")
	}
	for _, p := range ln.Parameters() {
		if p.Grad() == nil {
			t.Errorf("parameter %s has nil gradient", p.Name)
		}
	}
}

// --- GRUCell ---

func TestGRUCellForward(t *testing.T) {
	gru, err := nn.NewGRUCell(4, 3) // input=4, hidden=3
	if err != nil {
		t.Fatal(err)
	}

	// Check parameters: 6 Linear layers × 2 params each = 12
	params := gru.Parameters()
	if len(params) != 12 {
		t.Errorf("GRU params = %d, want 12", len(params))
	}

	// Forward with nil hidden state (first call)
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int64{2, 4})
	defer xt.Release()
	x := autograd.NewVariable(xt, false)

	h := gru.Forward(x) // nil h → zero init
	if err := h.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	shape := h.Data().Shape()
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("hidden shape = %v, want [2 3]", shape)
	}
}

func TestGRUCellWithHidden(t *testing.T) {
	gru, err := nn.NewGRUCell(4, 3)
	if err != nil {
		t.Fatal(err)
	}

	xt, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{1, 4})
	defer xt.Release()
	ht, _ := tensor.FromFloat32([]float32{0.5, 0.5, 0.5}, []int64{1, 3})
	defer ht.Release()

	x := autograd.NewVariable(xt, false)
	h := autograd.NewVariable(ht, false)

	hNew := gru.Forward(x, h)
	if err := hNew.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	shape := hNew.Data().Shape()
	if shape[0] != 1 || shape[1] != 3 {
		t.Errorf("hidden shape = %v, want [1 3]", shape)
	}
}

func TestGRUCellBackward(t *testing.T) {
	gru, err := nn.NewGRUCell(4, 3)
	if err != nil {
		t.Fatal(err)
	}

	xt, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{1, 4})
	defer xt.Release()
	x := autograd.NewVariable(xt, true)

	// Two steps: h1 = gru(x, nil), h2 = gru(x, h1)
	h1 := gru.Forward(x)
	h2 := gru.Forward(x, h1)
	loss := h2.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	// Input should have gradients (through both steps)
	if x.Grad() == nil {
		t.Error("input gradient is nil")
	}
	// All parameters should have gradients
	for _, p := range gru.Parameters() {
		if p.Grad() == nil {
			t.Errorf("parameter %s has nil gradient", p.Name)
		}
	}
}

// --- LSTMCell ---

func TestLSTMCellForward(t *testing.T) {
	lstm, err := nn.NewLSTMCell(4, 3) // input=4, hidden=3
	if err != nil {
		t.Fatal(err)
	}

	// 8 Linear layers × 2 params = 16
	params := lstm.Parameters()
	if len(params) != 16 {
		t.Errorf("LSTM params = %d, want 16", len(params))
	}

	// Forward with nil state (first call)
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int64{2, 4})
	defer xt.Release()

	state := lstm.Forward(autograd.NewVariable(xt, false))
	if err := state.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	shape := state.Data().Shape()
	// Output is cat(h, c) → [batch, 2*hidden]
	if shape[0] != 2 || shape[1] != 6 {
		t.Errorf("state shape = %v, want [2 6]", shape)
	}
}

func TestLSTMCellWithState(t *testing.T) {
	lstm, err := nn.NewLSTMCell(4, 3)
	if err != nil {
		t.Fatal(err)
	}

	xt, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{1, 4})
	defer xt.Release()
	// state = cat(h, c), each [1, 3]
	st, _ := tensor.FromFloat32([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int64{1, 6})
	defer st.Release()

	x := autograd.NewVariable(xt, false)
	state := autograd.NewVariable(st, false)

	newState := lstm.Forward(x, state)
	if err := newState.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	shape := newState.Data().Shape()
	if shape[0] != 1 || shape[1] != 6 {
		t.Errorf("state shape = %v, want [1 6]", shape)
	}
}

func TestLSTMCellBackward(t *testing.T) {
	lstm, err := nn.NewLSTMCell(4, 3)
	if err != nil {
		t.Fatal(err)
	}

	xt, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{1, 4})
	defer xt.Release()
	x := autograd.NewVariable(xt, true)

	// Two steps: state1 = lstm(x, nil), state2 = lstm(x, state1)
	state1 := lstm.Forward(x)
	state2 := lstm.Forward(x, state1)
	loss := state2.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	if x.Grad() == nil {
		t.Error("input gradient is nil")
	}
	for _, p := range lstm.Parameters() {
		if p.Grad() == nil {
			t.Errorf("parameter %s has nil gradient", p.Name)
		}
	}
}

func TestLSTMCellRecurrent(t *testing.T) {
	lstm, err := nn.NewLSTMCell(1, 4)
	if err != nil {
		t.Fatal(err)
	}
	proj, err := nn.NewLinear(4, 1)
	if err != nil {
		t.Fatal(err)
	}

	params := append(lstm.Parameters(), proj.Parameters()...)
	opt := nn.NewAdam(params, 0.01)

	var finalLoss float32
	for epoch := range 100 {
		x0t, _ := tensor.FromFloat32([]float32{2.0}, []int64{1, 1})
		x1t, _ := tensor.FromFloat32([]float32{0.0}, []int64{1, 1})
		tgt, _ := tensor.FromFloat32([]float32{2.0}, []int64{1, 1})

		x0 := autograd.NewVariable(x0t, false)
		x1 := autograd.NewVariable(x1t, false)
		target := autograd.NewVariable(tgt, false)

		state := lstm.Forward(x0)
		state = lstm.Forward(x1, state)
		state = lstm.Forward(x1, state)
		// Extract h from state (first half)
		h := state.Narrow(1, 0, 4)
		out := proj.Forward(h)

		loss := nn.MSELoss(out, target)
		if err := loss.Backward(); err != nil {
			t.Fatalf("epoch %d: %v", epoch, err)
		}
		opt.Step()
		opt.ZeroGrad()
		finalLoss = mustData(t, loss.Data())[0]

		x0t.Release()
		x1t.Release()
		tgt.Release()
	}

	if finalLoss > 0.1 {
		t.Errorf("LSTM recurrent final loss = %f, want < 0.1", finalLoss)
	}
	t.Logf("LSTM recurrent: loss=%.6f", finalLoss)
}

func TestGRUCellRecurrent(t *testing.T) {
	// Train a GRU to remember its input over 3 steps, then output it
	gru, err := nn.NewGRUCell(1, 4)
	if err != nil {
		t.Fatal(err)
	}
	proj, err := nn.NewLinear(4, 1)
	if err != nil {
		t.Fatal(err)
	}

	params := append(gru.Parameters(), proj.Parameters()...)
	opt := nn.NewAdam(params, 0.01)

	var finalLoss float32
	for epoch := range 100 {
		// Input: x=2.0 at step 0, then zeros
		x0t, _ := tensor.FromFloat32([]float32{2.0}, []int64{1, 1})
		x1t, _ := tensor.FromFloat32([]float32{0.0}, []int64{1, 1})
		tgt, _ := tensor.FromFloat32([]float32{2.0}, []int64{1, 1})

		x0 := autograd.NewVariable(x0t, false)
		x1 := autograd.NewVariable(x1t, false)
		target := autograd.NewVariable(tgt, false)

		h := gru.Forward(x0)
		h = gru.Forward(x1, h)
		h = gru.Forward(x1, h)
		out := proj.Forward(h)

		loss := nn.MSELoss(out, target)
		if err := loss.Backward(); err != nil {
			t.Fatalf("epoch %d: %v", epoch, err)
		}
		opt.Step()
		opt.ZeroGrad()
		finalLoss = mustData(t, loss.Data())[0]

		x0t.Release()
		x1t.Release()
		tgt.Release()
	}

	if finalLoss > 0.1 {
		t.Errorf("GRU recurrent final loss = %f, want < 0.1", finalLoss)
	}
	t.Logf("GRU recurrent: loss=%.6f", finalLoss)
}

// --- Conv2d ---

func TestConv2dForward(t *testing.T) {
	conv, err := nn.NewConv2d(1, 1, 3) // 1 in, 1 out, 3x3 kernel
	if err != nil {
		t.Fatal(err)
	}

	// Set known weights: 3x3 kernel of all ones, no bias.
	wData := make([]float32, 9)
	for i := range wData {
		wData[i] = 1
	}
	wt, _ := tensor.FromFloat32(wData, []int64{1, 1, 3, 3})
	conv.Weight.SetData(wt)
	bt, _ := tensor.FromFloat32([]float32{0}, []int64{1})
	conv.Bias.SetData(bt)

	// Input: 1 batch, 1 channel, 4x4 of all ones.
	ones := make([]float32, 16)
	for i := range ones {
		ones[i] = 1
	}
	xt, _ := tensor.FromFloat32(ones, []int64{1, 1, 4, 4})
	x := autograd.NewVariable(xt, false)

	out := conv.Forward(x)
	if err := out.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}

	// With 3x3 kernel of 1s, no padding: output is 2x2, each element = 9.
	shape := out.Data().Shape()
	if shape[0] != 1 || shape[1] != 1 || shape[2] != 2 || shape[3] != 2 {
		t.Fatalf("shape = %v, want [1, 1, 2, 2]", shape)
	}
	data := mustData(t, out.Data())
	for i, v := range data {
		if math.Abs(float64(v)-9.0) > eps {
			t.Errorf("output[%d] = %f, want 9.0", i, v)
		}
	}
}

func TestConv2dPadding(t *testing.T) {
	conv, err := nn.NewConv2d(1, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	conv.Padding = [2]int64{1, 1} // same-padding for 3x3

	// Input: 1x1x4x4
	ones := make([]float32, 16)
	for i := range ones {
		ones[i] = 1
	}
	xt, _ := tensor.FromFloat32(ones, []int64{1, 1, 4, 4})
	x := autograd.NewVariable(xt, false)

	out := conv.Forward(x)
	if err := out.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}

	// With padding=1, output should be 4x4 (same as input spatial dims).
	shape := out.Data().Shape()
	if shape[2] != 4 || shape[3] != 4 {
		t.Fatalf("padded shape = %v, want [1, 1, 4, 4]", shape)
	}
}

func TestConv2dBackward(t *testing.T) {
	conv, err := nn.NewConv2d(1, 2, 3) // 1→2 channels
	if err != nil {
		t.Fatal(err)
	}

	ones := make([]float32, 16)
	for i := range ones {
		ones[i] = 1
	}
	xt, _ := tensor.FromFloat32(ones, []int64{1, 1, 4, 4})
	x := autograd.NewVariable(xt, true)

	loss := conv.Forward(x).Sum()
	if err := loss.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	// Weight and bias should have gradients.
	for _, p := range conv.Parameters() {
		if p.Grad() == nil {
			t.Errorf("parameter %q has nil gradient", p.Name)
		}
	}
	// Input should have gradient.
	if x.Grad() == nil {
		t.Error("input gradient is nil")
	}

	// Weight grad shape should match weight shape.
	wGrad := conv.Weight.Grad()
	wShape := conv.Weight.Data().Shape()
	gShape := wGrad.Shape()
	for i := range wShape {
		if wShape[i] != gShape[i] {
			t.Errorf("weight grad shape %v != weight shape %v", gShape, wShape)
			break
		}
	}
}

func TestConv2dMultiChannel(t *testing.T) {
	// 3 input channels, 8 output channels, 3x3 kernel.
	conv, err := nn.NewConv2d(3, 8, 3)
	if err != nil {
		t.Fatal(err)
	}
	conv.Padding = [2]int64{1, 1}

	// Batch of 2, 3 channels, 8x8 spatial.
	xt, _ := tensor.Rand([]int64{2, 3, 8, 8})
	if err := xt.Err(); err != nil {
		t.Fatal(err)
	}
	x := autograd.NewVariable(xt, true)

	loss := conv.Forward(x).Sum()
	if err := loss.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := conv.Forward(x).Data().Shape()
	if shape[0] != 2 || shape[1] != 8 || shape[2] != 8 || shape[3] != 8 {
		t.Fatalf("shape = %v, want [2, 8, 8, 8]", shape)
	}

	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}
	if x.Grad() == nil {
		t.Error("input gradient is nil")
	}
}

// --- BatchNorm ---

func TestBatchNormTraining(t *testing.T) {
	bn, err := nn.NewBatchNorm(3)
	if err != nil {
		t.Fatal(err)
	}

	// Batch of 4 samples, 3 features.
	xt, _ := tensor.FromFloat32([]float32{
		1, 2, 3,
		5, 6, 7,
		3, 4, 5,
		7, 8, 9,
	}, []int64{4, 3})
	x := autograd.NewVariable(xt, false)

	out := bn.Forward(x)
	if err := out.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}

	data := mustData(t, out.Data())
	shape := out.Data().Shape()
	if shape[0] != 4 || shape[1] != 3 {
		t.Fatalf("shape = %v, want [4, 3]", shape)
	}

	// After batch norm, each feature column should have ~0 mean.
	for col := 0; col < 3; col++ {
		sum := float64(0)
		for row := 0; row < 4; row++ {
			sum += float64(data[row*3+col])
		}
		mean := sum / 4
		if math.Abs(mean) > 1e-3 {
			t.Errorf("feature %d mean = %f, want ~0", col, mean)
		}
	}
}

func TestBatchNormEval(t *testing.T) {
	bn, err := nn.NewBatchNorm(2)
	if err != nil {
		t.Fatal(err)
	}

	// Run a few training passes to populate running stats.
	for i := 0; i < 10; i++ {
		xt, _ := tensor.FromFloat32([]float32{
			float32(i), float32(i + 1),
			float32(i + 2), float32(i + 3),
			float32(i + 4), float32(i + 5),
			float32(i + 6), float32(i + 7),
		}, []int64{4, 2})
		bn.Forward(autograd.NewVariable(xt, false))
	}

	// Switch to eval mode.
	bn.SetTraining(false)

	xt, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{2, 2})
	x := autograd.NewVariable(xt, false)

	// Eval mode should be deterministic.
	out1 := bn.Forward(x)
	out2 := bn.Forward(x)
	if err := out1.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	d1 := mustData(t, out1.Data())
	d2 := mustData(t, out2.Data())
	for i := range d1 {
		if math.Abs(float64(d1[i]-d2[i])) > eps {
			t.Errorf("eval not deterministic: [%d] %f != %f", i, d1[i], d2[i])
		}
	}
}

func TestBatchNormRunningStats(t *testing.T) {
	bn, err := nn.NewBatchNorm(2)
	if err != nil {
		t.Fatal(err)
	}

	// Running mean starts at 0, running var starts at 1.
	rmBefore := mustData(t, bn.RunningMean)
	rvBefore := mustData(t, bn.RunningVar)
	for i := range rmBefore {
		if rmBefore[i] != 0 {
			t.Errorf("initial running mean[%d] = %f, want 0", i, rmBefore[i])
		}
		if rvBefore[i] != 1 {
			t.Errorf("initial running var[%d] = %f, want 1", i, rvBefore[i])
		}
	}

	// One training forward pass.
	xt, _ := tensor.FromFloat32([]float32{
		10, 20,
		30, 40,
	}, []int64{2, 2})
	bn.Forward(autograd.NewVariable(xt, false))

	// Running mean should have moved toward batch mean [20, 30].
	// momentum=0.1: running_mean = 0.9*0 + 0.1*[20,30] = [2, 3]
	rmAfter := mustData(t, bn.RunningMean)
	wantMean := []float32{2, 3}
	for i := range wantMean {
		if math.Abs(float64(rmAfter[i]-wantMean[i])) > eps {
			t.Errorf("running mean[%d] = %f, want %f", i, rmAfter[i], wantMean[i])
		}
	}

	// Running var should have moved toward corrected batch var.
	// Biased var = [100, 100], Bessel correction = 2/1 = 2, unbiased = [200, 200]
	// momentum=0.1: running_var = 0.9*1 + 0.1*200 = 20.9
	rvAfter := mustData(t, bn.RunningVar)
	wantVar := float32(20.9)
	for i := range rvAfter {
		if math.Abs(float64(rvAfter[i]-wantVar)) > eps {
			t.Errorf("running var[%d] = %f, want %f", i, rvAfter[i], wantVar)
		}
	}
}

func TestBatchNormBackward(t *testing.T) {
	bn, err := nn.NewBatchNorm(2)
	if err != nil {
		t.Fatal(err)
	}

	xt, _ := tensor.FromFloat32([]float32{
		1, 2,
		3, 4,
		5, 6,
	}, []int64{3, 2})
	x := autograd.NewVariable(xt, true)

	loss := bn.Forward(x).Sum()
	if err := loss.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	// Weight and bias should have gradients.
	for _, p := range bn.Parameters() {
		if p.Grad() == nil {
			t.Errorf("parameter %q has nil gradient", p.Name)
		}
	}
	// Input should have gradients.
	if x.Grad() == nil {
		t.Error("input gradient is nil")
	}
}

// --- Weight initialization ---

// initStats computes mean and variance of a float32 slice.
func initStats(data []float32) (mean, variance float64) {
	n := float64(len(data))
	for _, v := range data {
		mean += float64(v)
	}
	mean /= n
	for _, v := range data {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= n
	return mean, variance
}

func TestKaimingUniform(t *testing.T) {
	fanIn := int64(256)
	w, err := nn.KaimingUniform([]int64{512, 256}, fanIn)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := w.Float32Data()
	mean, variance := initStats(data)
	// Expected: U(-bound, bound) where bound = sqrt(6/256) ≈ 0.153
	// Uniform variance = bound^2/3 = 6/(256*3) = 0.0078
	bound := math.Sqrt(6.0 / float64(fanIn))
	expectedVar := bound * bound / 3.0
	if math.Abs(mean) > 0.01 {
		t.Errorf("mean = %f, want ~0", mean)
	}
	if math.Abs(variance-expectedVar)/expectedVar > 0.15 {
		t.Errorf("variance = %f, want ~%f", variance, expectedVar)
	}
}

func TestKaimingNormal(t *testing.T) {
	fanIn := int64(256)
	w, err := nn.KaimingNormal([]int64{512, 256}, fanIn)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := w.Float32Data()
	mean, variance := initStats(data)
	// Expected: N(0, std) where std = sqrt(2/256) ≈ 0.088
	expectedVar := 2.0 / float64(fanIn)
	if math.Abs(mean) > 0.01 {
		t.Errorf("mean = %f, want ~0", mean)
	}
	if math.Abs(variance-expectedVar)/expectedVar > 0.15 {
		t.Errorf("variance = %f, want ~%f", variance, expectedVar)
	}
}

func TestXavierUniform(t *testing.T) {
	fanIn, fanOut := int64(256), int64(512)
	w, err := nn.XavierUniform([]int64{512, 256}, fanIn, fanOut)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := w.Float32Data()
	mean, variance := initStats(data)
	bound := math.Sqrt(6.0 / float64(fanIn+fanOut))
	expectedVar := bound * bound / 3.0
	if math.Abs(mean) > 0.01 {
		t.Errorf("mean = %f, want ~0", mean)
	}
	if math.Abs(variance-expectedVar)/expectedVar > 0.15 {
		t.Errorf("variance = %f, want ~%f", variance, expectedVar)
	}
}

func TestXavierNormal(t *testing.T) {
	fanIn, fanOut := int64(256), int64(512)
	w, err := nn.XavierNormal([]int64{512, 256}, fanIn, fanOut)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := w.Float32Data()
	mean, variance := initStats(data)
	expectedVar := 2.0 / float64(fanIn+fanOut)
	if math.Abs(mean) > 0.01 {
		t.Errorf("mean = %f, want ~0", mean)
	}
	if math.Abs(variance-expectedVar)/expectedVar > 0.15 {
		t.Errorf("variance = %f, want ~%f", variance, expectedVar)
	}
}

// --- Checkpoint save/load ---

func TestSaveLoadParameters(t *testing.T) {
	l, err := nn.NewLinear(4, 3)
	if err != nil {
		t.Fatal(err)
	}
	params := l.Parameters()
	origW := mustData(t, params[0].Data())
	origB := mustData(t, params[1].Data())

	// Save.
	var buf bytes.Buffer
	if err := nn.SaveParameters(&buf, params); err != nil {
		t.Fatalf("save: %v", err)
	}

	// Load into fresh model with different random weights.
	l2, err := nn.NewLinear(4, 3)
	if err != nil {
		t.Fatal(err)
	}
	if err := nn.LoadParameters(&buf, l2.Parameters()); err != nil {
		t.Fatalf("load: %v", err)
	}

	loadedW := mustData(t, l2.Parameters()[0].Data())
	loadedB := mustData(t, l2.Parameters()[1].Data())
	for i := range origW {
		if origW[i] != loadedW[i] {
			t.Errorf("weight[%d]: saved=%f loaded=%f", i, origW[i], loadedW[i])
		}
	}
	for i := range origB {
		if origB[i] != loadedB[i] {
			t.Errorf("bias[%d]: saved=%f loaded=%f", i, origB[i], loadedB[i])
		}
	}
}

func TestSaveLoadForwardMatch(t *testing.T) {
	l, err := nn.NewLinear(3, 2)
	if err != nil {
		t.Fatal(err)
	}

	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{1, 3})
	x := autograd.NewVariable(xt, false)
	d1 := mustData(t, l.Forward(x).Data())

	var buf bytes.Buffer
	if err := nn.SaveParameters(&buf, l.Parameters()); err != nil {
		t.Fatal(err)
	}

	l2, err := nn.NewLinear(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := nn.LoadParameters(&buf, l2.Parameters()); err != nil {
		t.Fatal(err)
	}

	d2 := mustData(t, l2.Forward(x).Data())
	for i := range d1 {
		if d1[i] != d2[i] {
			t.Errorf("output[%d]: original=%f loaded=%f", i, d1[i], d2[i])
		}
	}
}

func TestLoadCountMismatch(t *testing.T) {
	l1, _ := nn.NewLinear(4, 3)
	var buf bytes.Buffer
	_ = nn.SaveParameters(&buf, l1.Parameters())

	l2, _ := nn.NewLinear(4, 3)
	err := nn.LoadParameters(&buf, l2.Parameters()[:1])
	if err == nil {
		t.Error("expected error for parameter count mismatch")
	}
}

func TestLoadShapeMismatch(t *testing.T) {
	l1, _ := nn.NewLinear(4, 3)
	var buf bytes.Buffer
	_ = nn.SaveParameters(&buf, l1.Parameters())

	l2, _ := nn.NewLinear(4, 5)
	err := nn.LoadParameters(&buf, l2.Parameters())
	if err == nil {
		t.Error("expected error for shape mismatch")
	}
}
