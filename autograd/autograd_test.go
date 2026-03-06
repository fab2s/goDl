package autograd_test

import (
	"math"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

const eps = 1e-5

func assertClose(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > eps {
			t.Errorf("%s[%d] = %f, want %f", name, i, got[i], want[i])
		}
	}
}

func mustData(t *testing.T, ts *tensor.Tensor) []float32 {
	t.Helper()
	data, err := ts.Float32Data()
	if err != nil {
		t.Fatalf("Float32Data: %v", err)
	}
	return data
}

// --- Add ---

func TestAddBackward(t *testing.T) {
	// z = x + y, loss = sum(z)
	// dL/dx = [1, 1, 1], dL/dy = [1, 1, 1]
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer xt.Release()
	yt, _ := tensor.FromFloat32([]float32{4, 5, 6}, []int64{3})
	defer yt.Release()

	x := autograd.NewVariable(xt, true)
	y := autograd.NewVariable(yt, true)

	loss := x.Add(y).Sum()
	if err := loss.Err(); err != nil {
		t.Fatalf("forward: %v", err)
	}
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	assertClose(t, "dx", mustData(t, x.Grad()), []float32{1, 1, 1})
	assertClose(t, "dy", mustData(t, y.Grad()), []float32{1, 1, 1})
}

// --- Sub ---

func TestSubBackward(t *testing.T) {
	// z = x - y, loss = sum(z)
	// dL/dx = [1, 1, 1], dL/dy = [-1, -1, -1]
	xt, _ := tensor.FromFloat32([]float32{5, 6, 7}, []int64{3})
	defer xt.Release()
	yt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer yt.Release()

	x := autograd.NewVariable(xt, true)
	y := autograd.NewVariable(yt, true)

	loss := x.Sub(y).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	assertClose(t, "dx", mustData(t, x.Grad()), []float32{1, 1, 1})
	assertClose(t, "dy", mustData(t, y.Grad()), []float32{-1, -1, -1})
}

// --- Mul ---

func TestMulBackward(t *testing.T) {
	// z = x * y, loss = sum(z)
	// dL/dx = y = [4, 5, 6], dL/dy = x = [1, 2, 3]
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer xt.Release()
	yt, _ := tensor.FromFloat32([]float32{4, 5, 6}, []int64{3})
	defer yt.Release()

	x := autograd.NewVariable(xt, true)
	y := autograd.NewVariable(yt, true)

	loss := x.Mul(y).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	assertClose(t, "dx", mustData(t, x.Grad()), []float32{4, 5, 6})
	assertClose(t, "dy", mustData(t, y.Grad()), []float32{1, 2, 3})
}

// --- Matmul ---

func TestMatmulBackward(t *testing.T) {
	// x [1,3] @ w [3,2] = z [1,2], loss = sum(z)
	// x = [1, 2, 3], w = [[1, 0], [0, 1], [0, 0]]
	// z = [1, 2]
	// dL/dz = [1, 1]
	// dL/dx = dL/dz @ w^T = [1,1] @ [[1,0,0],[0,1,0]] = [1, 1, 0]
	// dL/dw = x^T @ dL/dz = [[1],[2],[3]] @ [[1,1]] = [[1,1],[2,2],[3,3]]
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{1, 3})
	defer xt.Release()
	wt, _ := tensor.FromFloat32([]float32{1, 0, 0, 1, 0, 0}, []int64{3, 2})
	defer wt.Release()

	x := autograd.NewVariable(xt, true)
	w := autograd.NewVariable(wt, true)

	loss := x.Matmul(w).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	assertClose(t, "dx", mustData(t, x.Grad()), []float32{1, 1, 0})
	assertClose(t, "dw", mustData(t, w.Grad()), []float32{1, 1, 2, 2, 3, 3})
}

// --- ReLU ---

func TestReLUBackward(t *testing.T) {
	// x = [-2, -1, 0, 1, 2], z = relu(x) = [0, 0, 0, 1, 2]
	// loss = sum(z) = 3
	// dL/dx = [0, 0, 0, 1, 1]  (gradient flows where x > 0)
	xt, _ := tensor.FromFloat32([]float32{-2, -1, 0, 1, 2}, []int64{5})
	defer xt.Release()

	x := autograd.NewVariable(xt, true)
	loss := x.ReLU().Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	assertClose(t, "dx", mustData(t, x.Grad()), []float32{0, 0, 0, 1, 1})
}

// --- Sigmoid ---

func TestSigmoidBackward(t *testing.T) {
	// x = [0], sigmoid(0) = 0.5
	// dσ/dx = σ(x) * (1 - σ(x)) = 0.5 * 0.5 = 0.25
	// loss = sum(sigmoid(x)) = 0.5
	// dL/dx = 0.25
	xt, _ := tensor.FromFloat32([]float32{0}, []int64{1})
	defer xt.Release()

	x := autograd.NewVariable(xt, true)
	loss := x.Sigmoid().Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	assertClose(t, "dx", mustData(t, x.Grad()), []float32{0.25})
}

// --- Tanh ---

func TestTanhBackward(t *testing.T) {
	// x = [0], tanh(0) = 0
	// dtanh/dx = 1 - tanh²(x) = 1 - 0 = 1
	// loss = sum(tanh(x)) = 0
	// dL/dx = 1
	xt, _ := tensor.FromFloat32([]float32{0}, []int64{1})
	defer xt.Release()

	x := autograd.NewVariable(xt, true)
	loss := x.Tanh().Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	assertClose(t, "dx", mustData(t, x.Grad()), []float32{1.0})
}

// --- Chain rule: linear layer ---

func TestLinearLayerBackward(t *testing.T) {
	// y = ReLU(x @ W + b), loss = sum(y)
	// x [1,3], W [3,2], b [1,2]
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{1, 3})
	defer xt.Release()
	wt, _ := tensor.FromFloat32([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int64{3, 2})
	defer wt.Release()
	bt, _ := tensor.FromFloat32([]float32{-1, 0.5}, []int64{1, 2})
	defer bt.Release()

	x := autograd.NewVariable(xt, true)
	w := autograd.NewVariable(wt, true)
	b := autograd.NewVariable(bt, true)

	// Forward: z = x @ W + b, y = ReLU(z)
	z := x.Matmul(w).Add(b)
	y := z.ReLU()
	loss := y.Sum()

	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	// Verify gradients exist and have correct shapes
	if x.Grad() == nil {
		t.Fatal("x.Grad() is nil")
	}
	if w.Grad() == nil {
		t.Fatal("w.Grad() is nil")
	}
	if b.Grad() == nil {
		t.Fatal("b.Grad() is nil")
	}

	// x @ W = [1*0.1+2*0.3+3*0.5, 1*0.2+2*0.4+3*0.6] = [2.2, 2.8]
	// + b = [1.2, 3.3], ReLU = [1.2, 3.3] (both positive)
	// dL/dy = [1, 1]
	// dReLU/dz = [1, 1] (both positive)
	// dL/dz = [1, 1]
	// dL/db = [1, 1]
	assertClose(t, "db", mustData(t, b.Grad()), []float32{1, 1})

	// dL/dW = x^T @ dL/dz = [[1],[2],[3]] @ [[1,1]] = [[1,1],[2,2],[3,3]]
	assertClose(t, "dw", mustData(t, w.Grad()), []float32{1, 1, 2, 2, 3, 3})

	// dL/dx = dL/dz @ W^T = [1,1] @ [[0.1,0.3,0.5],[0.2,0.4,0.6]]
	//       = [0.3, 0.7, 1.1]
	assertClose(t, "dx", mustData(t, x.Grad()), []float32{0.3, 0.7, 1.1})
}

// --- Gradient accumulation: variable used twice ---

func TestGradientAccumulation(t *testing.T) {
	// loss = sum(x * x) = sum(x²)
	// dL/dx = 2x = [2, 4, 6]
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer xt.Release()

	x := autograd.NewVariable(xt, true)

	loss := x.Mul(x).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	assertClose(t, "dx", mustData(t, x.Grad()), []float32{2, 4, 6})
}

// --- NoGrad context ---

func TestNoGrad(t *testing.T) {
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer xt.Release()

	x := autograd.NewVariable(xt, true)

	autograd.NoGrad(func() {
		y := x.Add(x)
		// y should not have a gradFn
		if y.RequiresGrad() {
			t.Error("variable inside NoGrad should not require grad")
		}
	})
}

// --- Variable without grad ---

func TestNoGradVariable(t *testing.T) {
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer xt.Release()
	yt, _ := tensor.FromFloat32([]float32{4, 5, 6}, []int64{3})
	defer yt.Release()

	x := autograd.NewVariable(xt, true)
	y := autograd.NewVariable(yt, false) // no grad

	loss := x.Add(y).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	if x.Grad() == nil {
		t.Fatal("x should have gradient")
	}
	assertClose(t, "dx", mustData(t, x.Grad()), []float32{1, 1, 1})

	if y.Grad() != nil {
		t.Error("y should not have gradient (requiresGrad=false)")
	}
}

// --- Detach ---

func TestDetach(t *testing.T) {
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer xt.Release()

	x := autograd.NewVariable(xt, true)
	y := x.Mul(x) // y depends on x

	detached := y.Detach()
	if detached.RequiresGrad() {
		t.Error("detached variable should not require grad")
	}
	if !detached.IsLeaf() {
		t.Error("detached variable should be a leaf")
	}
}

// --- ZeroGrad ---

func TestZeroGrad(t *testing.T) {
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer xt.Release()
	yt, _ := tensor.FromFloat32([]float32{4, 5, 6}, []int64{3})
	defer yt.Release()

	x := autograd.NewVariable(xt, true)
	y := autograd.NewVariable(yt, false)

	// First backward
	loss := x.Add(y).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward 1: %v", err)
	}
	assertClose(t, "dx after first", mustData(t, x.Grad()), []float32{1, 1, 1})

	// Zero grad and backward again — gradient should be fresh, not accumulated
	x.ZeroGrad()
	if x.Grad() != nil {
		t.Error("grad should be nil after ZeroGrad")
	}

	loss2 := x.Add(y).Sum()
	if err := loss2.Backward(); err != nil {
		t.Fatalf("backward 2: %v", err)
	}
	assertClose(t, "dx after second", mustData(t, x.Grad()), []float32{1, 1, 1})
}

// --- Broadcast: add with different shapes ---

func TestBroadcastAddBackward(t *testing.T) {
	// x [2,3] + b [1,3] → result [2,3]
	// dL/db should be sum over dim 0 = [1, 1, 1] * 2 rows = [2, 2, 2]?
	// No: dL/dz = ones [2,3], unbroadcast to [1,3] → sum dim 0 = [2, 2, 2]
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	defer xt.Release()
	bt, _ := tensor.FromFloat32([]float32{0.1, 0.2, 0.3}, []int64{1, 3})
	defer bt.Release()

	x := autograd.NewVariable(xt, true)
	b := autograd.NewVariable(bt, true)

	loss := x.Add(b).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	// dx: ones [2,3] → [1,1,1,1,1,1]
	assertClose(t, "dx", mustData(t, x.Grad()), []float32{1, 1, 1, 1, 1, 1})
	// db: sum over dim 0 of ones [2,3] → [2, 2, 2]
	assertClose(t, "db", mustData(t, b.Grad()), []float32{2, 2, 2})
}

// --- MulScalar ---

func TestMulScalarBackward(t *testing.T) {
	// z = x * 3, loss = sum(z)
	// dL/dx = [3, 3, 3]
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer xt.Release()

	x := autograd.NewVariable(xt, true)
	loss := x.MulScalar(3).Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	assertClose(t, "dx", mustData(t, x.Grad()), []float32{3, 3, 3})
}
