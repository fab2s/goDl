package nn_test

import (
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
