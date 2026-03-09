package autograd_test

import (
	"sync"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

func TestScopeFreesIntermediates(t *testing.T) {
	skipIfDeviceUnavailable(t)
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3}, tensor.WithDevice(testDevice))
	yt, _ := tensor.FromFloat32([]float32{4, 5, 6}, []int64{3}, tensor.WithDevice(testDevice))

	x := autograd.NewVariable(xt, true)
	y := autograd.NewVariable(yt, true)

	scope := autograd.NewScope()

	sum := x.Add(y)      // intermediate — tracked
	loss := sum.Sum()     // intermediate — tracked

	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}

	// Verify gradients are computed correctly.
	assertClose(t, "dx", mustData(t, x.Grad()), []float32{1, 1, 1})
	assertClose(t, "dy", mustData(t, y.Grad()), []float32{1, 1, 1})

	// Read values before close.
	if loss.Data() == nil {
		t.Fatal("loss data should be non-nil before close")
	}

	scope.Close()

	// After close, intermediate data is nil.
	if sum.Data() != nil {
		t.Error("sum.Data() should be nil after scope.Close()")
	}
	if loss.Data() != nil {
		t.Error("loss.Data() should be nil after scope.Close()")
	}

	// Leaf parameters (inputs) are NOT freed — they were created via
	// NewVariable, not by an op.
	if x.Data() == nil {
		t.Error("x.Data() should survive scope.Close()")
	}
	if y.Data() == nil {
		t.Error("y.Data() should survive scope.Close()")
	}

	// Gradients on leaf parameters survive.
	if x.Grad() == nil {
		t.Error("x.Grad() should survive scope.Close()")
	}
}

func TestScopePreservesLeafParameters(t *testing.T) {
	skipIfDeviceUnavailable(t)
	// Simulate a multi-batch training loop: parameters must survive
	// across scope boundaries.
	wt, _ := tensor.FromFloat32([]float32{0.5, -0.3, 0.8}, []int64{3}, tensor.WithDevice(testDevice))
	w := autograd.NewVariable(wt, true)

	for batch := range 3 {
		scope := autograd.NewScope()

		xt, _ := tensor.FromFloat32([]float32{float32(batch + 1), 2, 3}, []int64{3}, tensor.WithDevice(testDevice))
		x := autograd.NewVariable(xt, false)

		loss := x.Mul(w).Sum()
		if err := loss.Err(); err != nil {
			t.Fatalf("batch %d forward: %v", batch, err)
		}
		if err := loss.Backward(); err != nil {
			t.Fatalf("batch %d backward: %v", batch, err)
		}

		// Read metric before close.
		_ = loss.Item()

		scope.Close()

		// Parameter must still be alive.
		if w.Data() == nil {
			t.Fatalf("batch %d: parameter data is nil after scope.Close()", batch)
		}
		if w.Grad() == nil {
			t.Fatalf("batch %d: parameter grad is nil after scope.Close()", batch)
		}

		w.ZeroGrad()
	}
}

func TestScopeNoScopeStillWorks(t *testing.T) {
	skipIfDeviceUnavailable(t)
	// Without a scope, everything should work as before (GC handles cleanup).
	xt, _ := tensor.FromFloat32([]float32{1, 2}, []int64{2}, tensor.WithDevice(testDevice))
	x := autograd.NewVariable(xt, true)

	loss := x.Sum()
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward: %v", err)
	}
	assertClose(t, "dx", mustData(t, x.Grad()), []float32{1, 1})
}

func TestScopeConcurrentTracking(t *testing.T) {
	skipIfDeviceUnavailable(t)
	scope := autograd.NewScope()

	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3}, tensor.WithDevice(testDevice))
	x := autograd.NewVariable(xt, true)

	var wg sync.WaitGroup
	for range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Each goroutine creates intermediates.
			_ = x.MulScalar(2.0).Sum()
		}()
	}
	wg.Wait()

	scope.Close()

	// Parameter survives.
	if x.Data() == nil {
		t.Error("x.Data() should survive concurrent scope.Close()")
	}
}

func TestScopeNoGrad(t *testing.T) {
	skipIfDeviceUnavailable(t)
	// Variables created inside NoGrad are still op results — they should
	// be tracked and freed by the scope.
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3}, tensor.WithDevice(testDevice))
	x := autograd.NewVariable(xt, false)

	scope := autograd.NewScope()

	var sum *autograd.Variable
	autograd.NoGrad(func() {
		sum = x.Add(x)
	})

	if sum.Data() == nil {
		t.Fatal("sum.Data() should be non-nil before close")
	}

	scope.Close()

	if sum.Data() != nil {
		t.Error("sum.Data() should be nil after scope.Close()")
	}
	// Input created via NewVariable is preserved.
	if x.Data() == nil {
		t.Error("x.Data() should survive scope.Close()")
	}
}

func TestScopeDoubleClose(t *testing.T) {
	skipIfDeviceUnavailable(t)
	xt, _ := tensor.FromFloat32([]float32{1, 2}, []int64{2}, tensor.WithDevice(testDevice))
	x := autograd.NewVariable(xt, true)

	scope := autograd.NewScope()
	_ = x.Sum()
	scope.Close()

	// Second close should be a no-op (vars already cleared).
	scope.Close()

	if x.Data() == nil {
		t.Error("x.Data() should survive double close")
	}
}
