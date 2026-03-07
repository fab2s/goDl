package nn_test

import (
	"math"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// makeParam creates a parameter with known data and gradient.
func makeParam(t *testing.T, data, grad []float32, shape []int64) *nn.Parameter {
	t.Helper()
	dt, err := tensor.FromFloat32(data, shape)
	if err != nil {
		t.Fatal(err)
	}
	p := nn.NewParameter(dt, "test")
	gt, err := tensor.FromFloat32(grad, shape)
	if err != nil {
		t.Fatal(err)
	}
	p.SetGrad(gt)
	return p
}

func assertParamClose(t *testing.T, name string, p *nn.Parameter, want []float32) {
	t.Helper()
	got := mustData(t, p.Data())
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Errorf("%s[%d] = %.8f, want %.8f (diff=%.2e)", name, i, got[i], want[i], math.Abs(float64(got[i]-want[i])))
		}
	}
}

// --- SGD without momentum ---

func TestSGDVanillaExact(t *testing.T) {
	// param = [1.0, 2.0, 3.0], grad = [0.1, 0.2, 0.3], lr = 0.5
	// new_param = param - lr * grad = [1.0-0.05, 2.0-0.1, 3.0-0.15] = [0.95, 1.9, 2.85]
	p := makeParam(t, []float32{1.0, 2.0, 3.0}, []float32{0.1, 0.2, 0.3}, []int64{3})
	opt := nn.NewSGD([]*nn.Parameter{p}, 0.5, 0)
	opt.Step()
	assertParamClose(t, "SGD_vanilla", p, []float32{0.95, 1.9, 2.85})
}

// --- SGD with momentum ---

func TestSGDMomentumExact(t *testing.T) {
	// param = [1.0, 2.0], grad = [0.5, -0.5], lr = 0.1, momentum = 0.9
	//
	// Step 1: v_0 = grad = [0.5, -0.5] (first step clones grad as velocity)
	//         param -= lr * v = [1.0, 2.0] - 0.1*[0.5, -0.5] = [0.95, 2.05]
	p := makeParam(t, []float32{1.0, 2.0}, []float32{0.5, -0.5}, []int64{2})
	opt := nn.NewSGD([]*nn.Parameter{p}, 0.1, 0.9)

	opt.Step()
	assertParamClose(t, "SGD_mom_step1", p, []float32{0.95, 2.05})

	// Step 2: new grad = [0.3, -0.3]
	g2, _ := tensor.FromFloat32([]float32{0.3, -0.3}, []int64{2})
	p.SetGrad(g2)
	// v = 0.9 * [0.5, -0.5] + [0.3, -0.3] = [0.75, -0.75]
	// param -= 0.1 * [0.75, -0.75] = [0.95, 2.05] - [0.075, -0.075] = [0.875, 2.125]
	opt.Step()
	assertParamClose(t, "SGD_mom_step2", p, []float32{0.875, 2.125})
}

// --- SGD ZeroGrad ---

func TestSGDZeroGrad(t *testing.T) {
	p := makeParam(t, []float32{1.0}, []float32{0.5}, []int64{1})
	opt := nn.NewSGD([]*nn.Parameter{p}, 0.1, 0)
	opt.ZeroGrad()
	if p.Grad() != nil {
		t.Error("grad should be nil after ZeroGrad")
	}
	// Step with nil grad should be a no-op.
	opt.Step()
	assertParamClose(t, "SGD_noop", p, []float32{1.0})
}

// --- Adam single step ---

func TestAdamExact(t *testing.T) {
	// param = [1.0, 2.0], grad = [0.5, -0.5]
	// lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8
	//
	// t=1:
	// m = 0 * 0.9 + 0.1 * [0.5, -0.5] = [0.05, -0.05]
	// v = 0 * 0.999 + 0.001 * [0.25, 0.25] = [0.00025, 0.00025]
	// m_hat = [0.05, -0.05] / (1 - 0.9^1) = [0.05, -0.05] / 0.1 = [0.5, -0.5]
	// v_hat = [0.00025, 0.00025] / (1 - 0.999^1) = [0.00025, 0.00025] / 0.001 = [0.25, 0.25]
	// update = lr * m_hat / (sqrt(v_hat) + eps) = 0.001 * [0.5, -0.5] / (sqrt(0.25) + 1e-8)
	//        = 0.001 * [0.5, -0.5] / 0.5 = 0.001 * [1.0, -1.0] = [0.001, -0.001]
	// new_param = [1.0 - 0.001, 2.0 + 0.001] = [0.999, 2.001]
	p := makeParam(t, []float32{1.0, 2.0}, []float32{0.5, -0.5}, []int64{2})
	opt := nn.NewAdam([]*nn.Parameter{p}, 0.001)
	opt.Step()
	assertParamClose(t, "Adam_step1", p, []float32{0.999, 2.001})
}

// --- Adam two steps ---

func TestAdamTwoStepsExact(t *testing.T) {
	// Verify bias correction changes between steps.
	p := makeParam(t, []float32{1.0}, []float32{0.4}, []int64{1})
	opt := nn.NewAdam([]*nn.Parameter{p}, 0.001)

	// Step 1: grad=0.4
	// m = 0.1 * 0.4 = 0.04
	// v = 0.001 * 0.16 = 0.00016
	// m_hat = 0.04 / 0.1 = 0.4
	// v_hat = 0.00016 / 0.001 = 0.16
	// update = 0.001 * 0.4 / (sqrt(0.16) + 1e-8) = 0.001 * 0.4 / 0.4 = 0.001
	// param = 1.0 - 0.001 = 0.999
	opt.Step()
	assertParamClose(t, "Adam_2step_s1", p, []float32{0.999})

	// Step 2: grad=0.2
	g2, _ := tensor.FromFloat32([]float32{0.2}, []int64{1})
	p.SetGrad(g2)
	// m = 0.9 * 0.04 + 0.1 * 0.2 = 0.036 + 0.02 = 0.056
	// v = 0.999 * 0.00016 + 0.001 * 0.04 = 0.00015984 + 0.00004 = 0.00019984
	// m_hat = 0.056 / (1 - 0.9^2) = 0.056 / 0.19 = 0.29473684...
	// v_hat = 0.00019984 / (1 - 0.999^2) = 0.00019984 / 0.001999 = 0.09997...
	// update = 0.001 * 0.29473684 / (sqrt(0.09997) + 1e-8)
	//        = 0.001 * 0.29473684 / 0.31618...
	//        = 0.001 * 0.93214... = 0.00093214...
	// param = 0.999 - 0.00093214 = 0.99806786
	beta1, beta2 := 0.9, 0.999
	m := 0.9*0.04 + 0.1*0.2
	v := 0.999*0.00016 + 0.001*0.04
	mHat := m / (1 - beta1*beta1)
	vHat := v / (1 - beta2*beta2)
	update := 0.001 * mHat / (math.Sqrt(vHat) + 1e-8)
	expected := float32(0.999 - update)

	opt.Step()
	assertParamClose(t, "Adam_2step_s2", p, []float32{expected})
}

// --- Adam ZeroGrad ---

func TestAdamZeroGrad(t *testing.T) {
	p := makeParam(t, []float32{1.0}, []float32{0.5}, []int64{1})
	opt := nn.NewAdam([]*nn.Parameter{p}, 0.001)
	opt.ZeroGrad()
	if p.Grad() != nil {
		t.Error("grad should be nil after ZeroGrad")
	}
}

// --- AdamW single step ---

func TestAdamWExact(t *testing.T) {
	// param = [1.0, 2.0], grad = [0.5, -0.5]
	// lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01
	//
	// Same Adam update as TestAdamStep, plus weight decay:
	// adam_update produces param_after_adam = [0.999, 2.001]
	// Then: param -= lr * wd * param_before
	//     = [0.999, 2.001] - 0.001 * 0.01 * [1.0, 2.0]
	//     = [0.999, 2.001] - [0.00001, 0.00002]
	//     = [0.99899, 2.00098]
	p := makeParam(t, []float32{1.0, 2.0}, []float32{0.5, -0.5}, []int64{2})
	opt := nn.NewAdamW([]*nn.Parameter{p}, 0.001, 0.01)
	opt.Step()
	assertParamClose(t, "AdamW_step1", p, []float32{0.99899, 2.00098})
}

// --- AdamW two steps: verify decay accumulates correctly ---

func TestAdamWTwoStepsExact(t *testing.T) {
	// Verify AdamW over two steps with exact hand-computed values.
	p := makeParam(t, []float32{5.0}, []float32{0.5}, []int64{1})
	lr, wd := 0.001, 0.01
	beta1, beta2, epsV := 0.9, 0.999, 1e-8
	opt := nn.NewAdamW([]*nn.Parameter{p}, lr, wd)

	// Step 1: grad=0.5, param=5.0
	// m = 0.1*0.5 = 0.05; v = 0.001*0.25 = 0.00025
	// m_hat = 0.05/0.1 = 0.5; v_hat = 0.00025/0.001 = 0.25
	// adam_update = 0.001 * 0.5 / (sqrt(0.25)+1e-8) = 0.001 * 0.5/0.5 = 0.001
	// after adam: 5.0 - 0.001 = 4.999
	// decay: 4.999 - 0.001*0.01*5.0 = 4.999 - 0.00005 = 4.99895
	opt.Step()
	s1 := 5.0 - lr*0.5/((math.Sqrt(0.25))+epsV) - lr*wd*5.0
	assertParamClose(t, "AdamW_2s_s1", p, []float32{float32(s1)})

	// Step 2: grad=0.3
	g2, _ := tensor.FromFloat32([]float32{0.3}, []int64{1})
	p.SetGrad(g2)
	m := beta1*0.05 + (1-beta1)*0.3
	v := beta2*0.00025 + (1-beta2)*0.09
	mHat := m / (1 - beta1*beta1)
	vHat := v / (1 - beta2*beta2)
	adamUpd := lr * mHat / (math.Sqrt(vHat) + epsV)
	s2 := s1 - adamUpd - lr*wd*s1

	opt.Step()
	assertParamClose(t, "AdamW_2s_s2", p, []float32{float32(s2)})
}

// --- Multiple params ---

func TestAdamMultipleParamsExact(t *testing.T) {
	// Verify optimizer handles multiple parameters independently.
	p1 := makeParam(t, []float32{1.0, 2.0}, []float32{0.5, -0.5}, []int64{2})
	p2 := makeParam(t, []float32{3.0}, []float32{0.1}, []int64{1})
	opt := nn.NewAdam([]*nn.Parameter{p1, p2}, 0.001)
	opt.Step()

	// p1: same as TestAdamStep
	assertParamClose(t, "Adam_multi_p1", p1, []float32{0.999, 2.001})

	// p2: grad=0.1
	// m = 0.1 * 0.1 = 0.01
	// v = 0.001 * 0.01 = 0.00001
	// m_hat = 0.01 / 0.1 = 0.1
	// v_hat = 0.00001 / 0.001 = 0.01
	// update = 0.001 * 0.1 / (sqrt(0.01) + 1e-8) = 0.001 * 0.1 / 0.1 = 0.001
	// new = 3.0 - 0.001 = 2.999
	assertParamClose(t, "Adam_multi_p2", p2, []float32{2.999})
}

// --- Nil grad skipped ---

func TestAdamNilGradSkip(t *testing.T) {
	p1 := makeParam(t, []float32{1.0}, []float32{0.5}, []int64{1})
	p2 := nn.NewParameter(func() *tensor.Tensor { t2, _ := tensor.FromFloat32([]float32{2.0}, []int64{1}); return t2 }(), "no_grad")
	// p2 has no gradient set.
	opt := nn.NewAdam([]*nn.Parameter{p1, p2}, 0.001)
	opt.Step()

	// p1 should be updated.
	got1 := mustData(t, p1.Data())
	if got1[0] == 1.0 {
		t.Error("p1 should have been updated")
	}
	// p2 should be unchanged.
	got2 := mustData(t, p2.Data())
	if got2[0] != 2.0 {
		t.Errorf("p2 should be unchanged, got %f", got2[0])
	}
}

// --- End-to-end: optimizer on a real forward/backward ---

func TestSGDEndToEndExact(t *testing.T) {
	// y = x * w, loss = (y - target)^2
	// x=2, w=1, target=4 → y=2, loss=(2-4)^2=4
	// dy/dw = x = 2, dloss/dw = 2*(y-target)*x = 2*(-2)*2 = -8
	// w_new = w - lr * grad = 1 - 0.1 * (-8) = 1.8
	wt, _ := tensor.FromFloat32([]float32{1.0}, []int64{1})
	w := nn.NewParameter(wt, "w")
	opt := nn.NewSGD([]*nn.Parameter{w}, 0.1, 0)

	xt, _ := tensor.FromFloat32([]float32{2.0}, []int64{1})
	x := autograd.NewVariable(xt, false)
	tt, _ := tensor.FromFloat32([]float32{4.0}, []int64{1})
	target := autograd.NewVariable(tt, false)

	y := x.Mul(w.Variable)
	loss := nn.MSELoss(y, target)

	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	// Verify gradient: dloss/dw = 2*(y-target)*x / N = 2*(2-4)*2 / 1 = -8
	gradData := mustData(t, w.Grad())
	if math.Abs(float64(gradData[0])-(-8.0)) > 1e-4 {
		t.Errorf("grad = %f, want -8.0", gradData[0])
	}

	opt.Step()
	assertParamClose(t, "SGD_e2e", w, []float32{1.8})
}
