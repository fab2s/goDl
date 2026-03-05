package libtorch_test

import (
	"math"
	"testing"

	"github.com/fab2s/goDl/internal/libtorch"
)

// --- Functional tests ---
//
// These tests verify the full stack: Go → CGo → C shim → libtorch → result.
// No mocks. Each test creates real tensors, performs real operations, and
// checks real results.

func TestZerosAndOnes(t *testing.T) {
	z, err := libtorch.Zeros([]int64{2, 3}, libtorch.Float32, libtorch.CPU)
	if err != nil {
		t.Fatalf("Zeros: %v", err)
	}
	defer z.Free()

	o, err := libtorch.Ones([]int64{2, 3}, libtorch.Float32, libtorch.CPU)
	if err != nil {
		t.Fatalf("Ones: %v", err)
	}
	defer o.Free()

	// Check shapes
	if got := z.Shapes(); len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Errorf("Zeros shape = %v, want [2 3]", got)
	}
	if got := o.Numel(); got != 6 {
		t.Errorf("Ones numel = %d, want 6", got)
	}

	// Check values
	zdata, err := z.Float32Data()
	if err != nil {
		t.Fatalf("Float32Data: %v", err)
	}
	for i, v := range zdata {
		if v != 0 {
			t.Errorf("Zeros[%d] = %f, want 0", i, v)
		}
	}

	odata, err := o.Float32Data()
	if err != nil {
		t.Fatalf("Float32Data: %v", err)
	}
	for i, v := range odata {
		if v != 1 {
			t.Errorf("Ones[%d] = %f, want 1", i, v)
		}
	}
}

func TestFromFloat32(t *testing.T) {
	data := []float32{1, 2, 3, 4, 5, 6}
	tensor, err := libtorch.FromFloat32(data, []int64{2, 3}, libtorch.CPU)
	if err != nil {
		t.Fatalf("FromFloat32: %v", err)
	}
	defer tensor.Free()

	got, err := tensor.Float32Data()
	if err != nil {
		t.Fatalf("Float32Data: %v", err)
	}
	for i := range data {
		if got[i] != data[i] {
			t.Errorf("data[%d] = %f, want %f", i, got[i], data[i])
		}
	}

	// Verify the tensor is a copy — modifying the Go slice shouldn't affect it.
	data[0] = 999
	got2, _ := tensor.Float32Data()
	if got2[0] == 999 {
		t.Error("FromFloat32 did not copy data — tensor shares Go memory")
	}
}

func TestAdd(t *testing.T) {
	a, _ := libtorch.FromFloat32([]float32{1, 2, 3}, []int64{3}, libtorch.CPU)
	defer a.Free()
	b, _ := libtorch.FromFloat32([]float32{10, 20, 30}, []int64{3}, libtorch.CPU)
	defer b.Free()

	c, err := libtorch.Add(a, b)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	defer c.Free()

	got, _ := c.Float32Data()
	want := []float32{11, 22, 33}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Add[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestMul(t *testing.T) {
	a, _ := libtorch.FromFloat32([]float32{2, 3, 4}, []int64{3}, libtorch.CPU)
	defer a.Free()
	b, _ := libtorch.FromFloat32([]float32{5, 6, 7}, []int64{3}, libtorch.CPU)
	defer b.Free()

	c, err := libtorch.Mul(a, b)
	if err != nil {
		t.Fatalf("Mul: %v", err)
	}
	defer c.Free()

	got, _ := c.Float32Data()
	want := []float32{10, 18, 28}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Mul[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestMatmul(t *testing.T) {
	// [2x3] @ [3x2] = [2x2]
	a, _ := libtorch.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3}, libtorch.CPU)
	defer a.Free()
	b, _ := libtorch.FromFloat32([]float32{7, 8, 9, 10, 11, 12}, []int64{3, 2}, libtorch.CPU)
	defer b.Free()

	c, err := libtorch.Matmul(a, b)
	if err != nil {
		t.Fatalf("Matmul: %v", err)
	}
	defer c.Free()

	if got := c.Shapes(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Fatalf("Matmul shape = %v, want [2 2]", got)
	}

	got, _ := c.Float32Data()
	// [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
	// [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
	want := []float32{58, 64, 139, 154}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Matmul[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestActivations(t *testing.T) {
	data := []float32{-2, -1, 0, 1, 2}
	a, _ := libtorch.FromFloat32(data, []int64{5}, libtorch.CPU)
	defer a.Free()

	// ReLU: max(0, x)
	r, err := libtorch.ReLU(a)
	if err != nil {
		t.Fatalf("ReLU: %v", err)
	}
	defer r.Free()
	rdata, _ := r.Float32Data()
	reluWant := []float32{0, 0, 0, 1, 2}
	for i := range reluWant {
		if rdata[i] != reluWant[i] {
			t.Errorf("ReLU[%d] = %f, want %f", i, rdata[i], reluWant[i])
		}
	}

	// Sigmoid: should be in (0, 1), symmetric around 0.5
	s, err := libtorch.Sigmoid(a)
	if err != nil {
		t.Fatalf("Sigmoid: %v", err)
	}
	defer s.Free()
	sdata, _ := s.Float32Data()
	if sdata[2] < 0.499 || sdata[2] > 0.501 {
		t.Errorf("Sigmoid(0) = %f, want ~0.5", sdata[2])
	}
	// sigmoid(-x) + sigmoid(x) should equal 1
	for i := 0; i < 2; i++ {
		sum := sdata[i] + sdata[4-i]
		if sum < 0.999 || sum > 1.001 {
			t.Errorf("Sigmoid symmetry: sigmoid(%f) + sigmoid(%f) = %f, want ~1",
				data[i], data[4-i], sum)
		}
	}

	// Tanh: should be in (-1, 1), odd function
	th, err := libtorch.Tanh(a)
	if err != nil {
		t.Fatalf("Tanh: %v", err)
	}
	defer th.Free()
	tdata, _ := th.Float32Data()
	if math.Abs(float64(tdata[2])) > 0.001 {
		t.Errorf("Tanh(0) = %f, want ~0", tdata[2])
	}
	// tanh(-x) = -tanh(x)
	for i := 0; i < 2; i++ {
		sum := tdata[i] + tdata[4-i]
		if math.Abs(float64(sum)) > 0.001 {
			t.Errorf("Tanh symmetry: tanh(%f) + tanh(%f) = %f, want ~0",
				data[i], data[4-i], sum)
		}
	}
}

func TestDType(t *testing.T) {
	f32, _ := libtorch.Zeros([]int64{2}, libtorch.Float32, libtorch.CPU)
	defer f32.Free()
	f64, _ := libtorch.Zeros([]int64{2}, libtorch.Float64, libtorch.CPU)
	defer f64.Free()
	i32, _ := libtorch.Zeros([]int64{2}, libtorch.Int32, libtorch.CPU)
	defer i32.Free()
	i64, _ := libtorch.Zeros([]int64{2}, libtorch.Int64, libtorch.CPU)
	defer i64.Free()

	if f32.DType() != libtorch.Float32 {
		t.Errorf("Float32 tensor dtype = %d, want %d", f32.DType(), libtorch.Float32)
	}
	if f64.DType() != libtorch.Float64 {
		t.Errorf("Float64 tensor dtype = %d, want %d", f64.DType(), libtorch.Float64)
	}
	if i32.DType() != libtorch.Int32 {
		t.Errorf("Int32 tensor dtype = %d, want %d", i32.DType(), libtorch.Int32)
	}
	if i64.DType() != libtorch.Int64 {
		t.Errorf("Int64 tensor dtype = %d, want %d", i64.DType(), libtorch.Int64)
	}
}

func TestFloat64Roundtrip(t *testing.T) {
	data := []float64{1.1, 2.2, 3.3, 4.4}
	tensor, err := libtorch.FromFloat64(data, []int64{2, 2}, libtorch.CPU)
	if err != nil {
		t.Fatalf("FromFloat64: %v", err)
	}
	defer tensor.Free()

	got, err := tensor.Float64Data()
	if err != nil {
		t.Fatalf("Float64Data: %v", err)
	}
	for i := range data {
		if math.Abs(got[i]-data[i]) > 1e-10 {
			t.Errorf("data[%d] = %f, want %f", i, got[i], data[i])
		}
	}
}

// --- CUDA tests ---
// These run only when CUDA is available. They are skipped otherwise.

func TestCUDAAvailability(t *testing.T) {
	available := libtorch.CUDAAvailable()
	count := libtorch.CUDADeviceCount()
	t.Logf("CUDA available: %v, device count: %d", available, count)
	if available && count == 0 {
		t.Error("CUDA is available but device count is 0")
	}
}

func TestCUDAMatmul(t *testing.T) {
	if !libtorch.CUDAAvailable() {
		t.Skip("CUDA not available")
	}

	// Create tensors directly on CUDA
	a, err := libtorch.FromFloat32(
		[]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3}, libtorch.CUDA)
	if err != nil {
		t.Fatalf("FromFloat32 CUDA: %v", err)
	}
	defer a.Free()

	b, err := libtorch.FromFloat32(
		[]float32{7, 8, 9, 10, 11, 12}, []int64{3, 2}, libtorch.CUDA)
	if err != nil {
		t.Fatalf("FromFloat32 CUDA: %v", err)
	}
	defer b.Free()

	// Verify they're on CUDA
	if a.Device() != libtorch.CUDA {
		t.Fatalf("tensor a on %d, want CUDA (%d)", a.Device(), libtorch.CUDA)
	}

	// Matmul on CUDA — may fail on older GPUs (Pascal/SM 6.x) with newer
	// cuBLAS versions that dropped support for their compute capability.
	c, err := libtorch.Matmul(a, b)
	if err != nil {
		t.Skipf("Matmul CUDA not supported on this GPU (likely arch mismatch): %v", err)
	}
	defer c.Free()

	// Read result back to CPU (Float32Data handles this automatically)
	got, err := c.Float32Data()
	if err != nil {
		t.Fatalf("Float32Data: %v", err)
	}

	want := []float32{58, 64, 139, 154}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("CUDA Matmul[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestCPUToCUDARoundtrip(t *testing.T) {
	if !libtorch.CUDAAvailable() {
		t.Skip("CUDA not available")
	}

	// Create on CPU
	data := []float32{1, 2, 3, 4}
	cpu, err := libtorch.FromFloat32(data, []int64{4}, libtorch.CPU)
	if err != nil {
		t.Fatalf("FromFloat32: %v", err)
	}
	defer cpu.Free()

	// Move to CUDA
	gpu, err := cpu.ToDevice(libtorch.CUDA)
	if err != nil {
		t.Fatalf("ToDevice CUDA: %v", err)
	}
	defer gpu.Free()

	if gpu.Device() != libtorch.CUDA {
		t.Errorf("device = %d, want CUDA", gpu.Device())
	}

	// Move back to CPU
	cpu2, err := gpu.ToDevice(libtorch.CPU)
	if err != nil {
		t.Fatalf("ToDevice CPU: %v", err)
	}
	defer cpu2.Free()

	// Verify data survived the roundtrip
	got, _ := cpu2.Float32Data()
	for i := range data {
		if got[i] != data[i] {
			t.Errorf("roundtrip[%d] = %f, want %f", i, got[i], data[i])
		}
	}
}
