package nn_test

import (
	"testing"

	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

func TestCastParametersToFloat16(t *testing.T) {
	data, err := tensor.FromFloat32([]float32{1.0, 2.0, 3.0}, []int64{3})
	if err != nil {
		t.Fatal(err)
	}
	p := nn.NewParameter(data, "w")

	nn.CastParameters([]*nn.Parameter{p}, tensor.Float16)
	if p.Data().DType() != tensor.Float16 {
		t.Errorf("dtype = %s, want float16", p.Data().DType())
	}

	// Data should survive round-trip through float32.
	f32, err := p.Data().Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	if f32[0] != 1.0 || f32[1] != 2.0 || f32[2] != 3.0 {
		t.Errorf("data = %v, want [1 2 3]", f32)
	}
}

func TestCastParametersToBFloat16(t *testing.T) {
	data, err := tensor.FromFloat32([]float32{4.0, 5.0}, []int64{2})
	if err != nil {
		t.Fatal(err)
	}
	p := nn.NewParameter(data, "w")

	nn.CastParameters([]*nn.Parameter{p}, tensor.BFloat16)
	if p.Data().DType() != tensor.BFloat16 {
		t.Errorf("dtype = %s, want bfloat16", p.Data().DType())
	}

	f32, err := p.Data().Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	if f32[0] != 4.0 || f32[1] != 5.0 {
		t.Errorf("data = %v, want [4 5]", f32)
	}
}

func TestCastParametersRoundTrip(t *testing.T) {
	data, err := tensor.FromFloat32([]float32{1.5, -2.5}, []int64{2})
	if err != nil {
		t.Fatal(err)
	}
	p := nn.NewParameter(data, "w")

	// float32 → float16 → float32
	nn.CastParameters([]*nn.Parameter{p}, tensor.Float16)
	nn.CastParameters([]*nn.Parameter{p}, tensor.Float32)
	if p.Data().DType() != tensor.Float32 {
		t.Errorf("dtype = %s, want float32", p.Data().DType())
	}
	f32, err := p.Data().Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	if f32[0] != 1.5 || f32[1] != -2.5 {
		t.Errorf("data = %v, want [1.5 -2.5]", f32)
	}
}

func TestGradScalerFiniteGrads(t *testing.T) {
	scaler := nn.NewGradScaler()
	if scaler.ScaleFactor() != 65536.0 {
		t.Fatalf("initial scale = %f, want 65536", scaler.ScaleFactor())
	}

	// Create a simple optimizer.
	data, _ := tensor.FromFloat32([]float32{1.0}, []int64{1})
	p := nn.NewParameter(data, "w")
	opt := nn.NewAdam([]*nn.Parameter{p}, 0.001)

	// Simulate a gradient.
	grad, _ := tensor.FromFloat32([]float32{0.5}, []int64{1})
	p.SetGrad(grad.MulScalar(scaler.ScaleFactor())) // pre-scaled gradient

	stepped := scaler.Step(opt)
	if !stepped {
		t.Error("Step returned false, expected true (finite grads)")
	}

	scaler.Update()
	// Scale should stay the same (growth interval not reached).
	if scaler.ScaleFactor() != 65536.0 {
		t.Errorf("scale = %f, want 65536 (no growth yet)", scaler.ScaleFactor())
	}
}

func TestGradScalerInfGrads(t *testing.T) {
	scaler := nn.NewGradScaler()
	initialScale := scaler.ScaleFactor()

	data, _ := tensor.FromFloat32([]float32{1.0}, []int64{1})
	p := nn.NewParameter(data, "w")
	opt := nn.NewAdam([]*nn.Parameter{p}, 0.001)

	// Set an infinite gradient.
	inf, _ := tensor.FromFloat32([]float32{1.0}, []int64{1})
	// Create inf by dividing by zero: 1/0 = inf
	zero, _ := tensor.FromFloat32([]float32{0.0}, []int64{1})
	infGrad := inf.Div(zero)
	p.SetGrad(infGrad)

	stepped := scaler.Step(opt)
	if stepped {
		t.Error("Step returned true, expected false (inf grads)")
	}

	scaler.Update()
	expected := initialScale * 0.5
	if scaler.ScaleFactor() != expected {
		t.Errorf("scale = %f, want %f (backoff after inf)", scaler.ScaleFactor(), expected)
	}
}

func TestTensorHalfFloat(t *testing.T) {
	x, err := tensor.FromFloat32([]float32{1.0, 2.0, 3.0}, []int64{3})
	if err != nil {
		t.Fatal(err)
	}

	h := x.Half()
	if h.DType() != tensor.Float16 {
		t.Errorf("Half dtype = %s, want float16", h.DType())
	}

	f := h.Float()
	if f.DType() != tensor.Float32 {
		t.Errorf("Float dtype = %s, want float32", f.DType())
	}

	data, err := f.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	if data[0] != 1.0 || data[1] != 2.0 || data[2] != 3.0 {
		t.Errorf("round-trip data = %v, want [1 2 3]", data)
	}
}

func TestTensorAllFinite(t *testing.T) {
	x, _ := tensor.FromFloat32([]float32{1.0, 2.0, 3.0}, []int64{3})
	if !x.AllFinite() {
		t.Error("expected all finite")
	}

	// Create inf.
	one, _ := tensor.FromFloat32([]float32{1.0}, []int64{1})
	zero, _ := tensor.FromFloat32([]float32{0.0}, []int64{1})
	inf := one.Div(zero)
	if inf.AllFinite() {
		t.Error("expected not all finite (has inf)")
	}
}

func TestTensorToDType(t *testing.T) {
	x, _ := tensor.FromFloat32([]float32{1.0, 2.0}, []int64{2})

	// No-op when already the right dtype.
	same := x.ToDType(tensor.Float32)
	if same != x {
		t.Error("ToDType same type should return same tensor")
	}

	// Cast to float64 and back.
	f64 := x.ToDType(tensor.Float64)
	if f64.DType() != tensor.Float64 {
		t.Errorf("dtype = %s, want float64", f64.DType())
	}
	f32 := f64.ToDType(tensor.Float32)
	if f32.DType() != tensor.Float32 {
		t.Errorf("dtype = %s, want float32", f32.DType())
	}
	data, _ := f32.Float32Data()
	if data[0] != 1.0 || data[1] != 2.0 {
		t.Errorf("data = %v, want [1 2]", data)
	}
}

func TestFloat32DataFromFloat16(t *testing.T) {
	// Create float32, cast to float16, then read as float32.
	x, _ := tensor.FromFloat32([]float32{1.5, -2.0, 3.0}, []int64{3})
	h := x.Half()
	if h.DType() != tensor.Float16 {
		t.Fatalf("dtype = %s, want float16", h.DType())
	}
	data, err := h.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	if data[0] != 1.5 || data[1] != -2.0 || data[2] != 3.0 {
		t.Errorf("Float32Data from fp16 = %v, want [1.5 -2 3]", data)
	}
}
