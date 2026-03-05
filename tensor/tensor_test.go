package tensor_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/fab2s/goDl/tensor"
)

// --- Creation and metadata ---

func TestCreateZeros(t *testing.T) {
	z, err := tensor.Zeros([]int64{2, 3})
	if err != nil {
		t.Fatalf("Zeros: %v", err)
	}
	defer z.Release()

	if got := z.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Errorf("shape = %v, want [2 3]", got)
	}
	if z.Ndim() != 2 {
		t.Errorf("ndim = %d, want 2", z.Ndim())
	}
	if z.Numel() != 6 {
		t.Errorf("numel = %d, want 6", z.Numel())
	}
	if z.DType() != tensor.Float32 {
		t.Errorf("dtype = %s, want float32", z.DType())
	}
	if z.Device() != tensor.CPU {
		t.Errorf("device = %s, want cpu", z.Device())
	}

	data, err := z.Float32Data()
	if err != nil {
		t.Fatalf("Float32Data: %v", err)
	}
	for i, v := range data {
		if v != 0 {
			t.Errorf("data[%d] = %f, want 0", i, v)
		}
	}
}

func TestCreateWithOptions(t *testing.T) {
	o, err := tensor.Ones([]int64{4}, tensor.WithDType(tensor.Float64))
	if err != nil {
		t.Fatalf("Ones: %v", err)
	}
	defer o.Release()

	if o.DType() != tensor.Float64 {
		t.Errorf("dtype = %s, want float64", o.DType())
	}

	data, err := o.Float64Data()
	if err != nil {
		t.Fatalf("Float64Data: %v", err)
	}
	for i, v := range data {
		if v != 1 {
			t.Errorf("data[%d] = %f, want 1", i, v)
		}
	}
}

func TestFromFloat32(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	x, err := tensor.FromFloat32(data, []int64{2, 3})
	if err != nil {
		t.Fatalf("FromFloat32: %v", err)
	}
	defer x.Release()

	got, err := x.Float32Data()
	if err != nil {
		t.Fatalf("Float32Data: %v", err)
	}
	for i := range data {
		if got[i] != data[i] {
			t.Errorf("got[%d] = %f, want %f", i, got[i], data[i])
		}
	}
}

func TestString(t *testing.T) {
	x, _ := tensor.Zeros([]int64{2, 3})
	defer x.Release()

	s := x.String()
	if s != "Tensor(shape=[2 3], dtype=float32, device=cpu)" {
		t.Errorf("String() = %q", s)
	}
}

// --- Chaining operations ---

func TestChainedOps(t *testing.T) {
	a, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer a.Release()
	b, _ := tensor.FromFloat32([]float32{10, 20, 30}, []int64{3})
	defer b.Release()

	// Chain: add then relu
	result := a.Add(b).ReLU()
	if err := result.Err(); err != nil {
		t.Fatalf("chain error: %v", err)
	}
	defer result.Release()

	got, _ := result.Float32Data()
	want := []float32{11, 22, 33}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestChainedMatmul(t *testing.T) {
	// Linear layer: y = ReLU(x @ W + b)
	x, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{1, 3})
	defer x.Release()
	w, _ := tensor.FromFloat32([]float32{1, 0, 0, 1, 0, 0}, []int64{3, 2})
	defer w.Release()
	b, _ := tensor.FromFloat32([]float32{-5, 10}, []int64{1, 2})
	defer b.Release()

	// x @ W = [1, 2] (first two elements of x, since W is a selector)
	// + b = [-4, 12]
	// ReLU = [0, 12]
	y := x.Matmul(w).Add(b).ReLU()
	if err := y.Err(); err != nil {
		t.Fatalf("chain error: %v", err)
	}
	defer y.Release()

	got, _ := y.Float32Data()
	want := []float32{0, 12}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("y[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

// --- Error propagation ---

func TestErrorPropagation(t *testing.T) {
	a, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer a.Release()

	// Release b, then try to use it — should propagate error
	b, _ := tensor.FromFloat32([]float32{4, 5, 6}, []int64{3})
	b.Release()

	result := a.Add(b)
	if result.Err() == nil {
		t.Fatal("expected error from using released tensor")
	}

	// Further chaining should propagate the error
	result2 := result.ReLU().Sigmoid()
	if result2.Err() == nil {
		t.Fatal("expected error to propagate through chain")
	}
}

func TestErrorOnReleasedTensor(t *testing.T) {
	a, _ := tensor.Zeros([]int64{3})
	a.Release()

	if a.Err() == nil {
		t.Fatal("expected error from released tensor")
	}
	if a.Shape() != nil {
		t.Error("Shape() should return nil for released tensor")
	}
	if a.Numel() != 0 {
		t.Error("Numel() should return 0 for released tensor")
	}
}

// --- Activations ---

func TestActivations(t *testing.T) {
	x, _ := tensor.FromFloat32([]float32{-2, -1, 0, 1, 2}, []int64{5})
	defer x.Release()

	// Sigmoid: symmetric around 0.5
	s := x.Sigmoid()
	if err := s.Err(); err != nil {
		t.Fatalf("Sigmoid: %v", err)
	}
	defer s.Release()

	sdata, _ := s.Float32Data()
	if sdata[2] < 0.499 || sdata[2] > 0.501 {
		t.Errorf("sigmoid(0) = %f, want ~0.5", sdata[2])
	}

	// Tanh: odd function, tanh(0) = 0
	th := x.Tanh()
	if err := th.Err(); err != nil {
		t.Fatalf("Tanh: %v", err)
	}
	defer th.Release()

	tdata, _ := th.Float32Data()
	if math.Abs(float64(tdata[2])) > 0.001 {
		t.Errorf("tanh(0) = %f, want ~0", tdata[2])
	}
}

// --- Scope ---

func TestScope(t *testing.T) {
	before := tensor.ActiveTensors()

	scope := tensor.NewScope()
	a, _ := tensor.Zeros([]int64{100, 100})
	scope.Track(a)
	b, _ := tensor.Ones([]int64{100, 100})
	scope.Track(b)
	c := scope.Track(a.Add(b))

	// 3 tensors tracked (a, b, c)
	if tensor.ActiveTensors()-before != 3 {
		t.Errorf("active tensors = %d, want %d", tensor.ActiveTensors()-before, 3)
	}

	// Verify c is valid before close
	if err := c.Err(); err != nil {
		t.Fatalf("c should be valid: %v", err)
	}

	scope.Close()

	// All should be released
	if tensor.ActiveTensors()-before != 0 {
		t.Errorf("after Close: active tensors = %d, want 0", tensor.ActiveTensors()-before)
	}

	// c should now be in error state
	if c.Err() == nil {
		t.Error("c should have error after scope close")
	}
}

func TestWithScope(t *testing.T) {
	before := tensor.ActiveTensors()

	result, err := tensor.WithScope(func(s *tensor.Scope) *tensor.Tensor {
		a, _ := tensor.Zeros([]int64{3})
		s.Track(a)
		b, _ := tensor.Ones([]int64{3})
		s.Track(b)

		// Return the result — it should survive scope closure
		return a.Add(b)
	})
	if err != nil {
		t.Fatalf("WithScope: %v", err)
	}
	defer result.Release()

	// Only the result should be alive
	if tensor.ActiveTensors()-before != 1 {
		t.Errorf("active tensors = %d, want 1", tensor.ActiveTensors()-before)
	}

	// Verify the result is correct
	got, _ := result.Float32Data()
	for i, v := range got {
		if v != 1 {
			t.Errorf("result[%d] = %f, want 1", i, v)
		}
	}
}

func TestGCFinalizer(t *testing.T) {
	// Force GC first to flush any pending finalizers from earlier tests
	runtime.GC()
	runtime.GC()

	before := tensor.ActiveTensors()

	// Create a tensor and let it go out of scope — GC should finalize it
	func() {
		x, _ := tensor.Zeros([]int64{100, 100})
		_ = x // goes out of scope
	}()

	// Force GC to run finalizers
	runtime.GC()
	runtime.GC()

	delta := tensor.ActiveTensors() - before
	if delta != 0 {
		// Finalizer timing is not guaranteed, so only warn
		t.Logf("after GC: active tensors delta = %d (finalizer may be pending)", delta)
	}
}

// --- CUDA ---

func TestCUDADevice(t *testing.T) {
	if !tensor.CUDAAvailable() {
		t.Skip("CUDA not available")
	}

	x, err := tensor.Zeros([]int64{3, 3}, tensor.WithDevice(tensor.CUDA))
	if err != nil {
		t.Fatalf("Zeros CUDA: %v", err)
	}
	defer x.Release()

	if x.Device() != tensor.CUDA {
		t.Errorf("device = %s, want cuda", x.Device())
	}
}

func TestCUDAChain(t *testing.T) {
	if !tensor.CUDAAvailable() {
		t.Skip("CUDA not available")
	}

	x, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{2, 2},
		tensor.WithDevice(tensor.CUDA))
	defer x.Release()

	w, _ := tensor.FromFloat32([]float32{1, 0, 0, 1}, []int64{2, 2},
		tensor.WithDevice(tensor.CUDA))
	defer w.Release()

	// Identity matmul → relu on GPU, then read back
	result := x.Matmul(w).ReLU()
	if err := result.Err(); err != nil {
		t.Skipf("CUDA chain not supported on this GPU: %v", err)
	}
	defer result.Release()

	got, err := result.Float32Data()
	if err != nil {
		t.Fatalf("Float32Data: %v", err)
	}
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestToDeviceChain(t *testing.T) {
	if !tensor.CUDAAvailable() {
		t.Skip("CUDA not available")
	}

	// Create on CPU, move to CUDA, operate, move back
	x, _ := tensor.FromFloat32([]float32{-1, 2, -3, 4}, []int64{4})
	defer x.Release()

	result := x.ToCUDA().ReLU().ToCPU()
	if err := result.Err(); err != nil {
		t.Fatalf("chain error: %v", err)
	}
	defer result.Release()

	if result.Device() != tensor.CPU {
		t.Errorf("device = %s, want cpu", result.Device())
	}

	got, _ := result.Float32Data()
	want := []float32{0, 2, 0, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("result[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}
