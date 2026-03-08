package nn_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// --- Helper ---

// stepOptimizer runs a forward+backward to set gradients, then steps.
func stepOptimizer(t *testing.T, params []*nn.Parameter, opt nn.Optimizer) {
	t.Helper()
	for _, p := range params {
		grad, err := tensor.Ones(p.Data().Shape())
		if err != nil {
			t.Fatal(err)
		}
		p.SetGrad(grad)
	}
	opt.Step()
	opt.ZeroGrad()
}

func tensorData(t *testing.T, ts *tensor.Tensor) []float32 {
	t.Helper()
	data, err := ts.Float32Data()
	if err != nil {
		t.Fatalf("Float32Data: %v", err)
	}
	return data
}

func assertClose(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Errorf("%s[%d] = %g, want %g", label, i, got[i], want[i])
		}
	}
}

func makeParams(t *testing.T) []*nn.Parameter {
	t.Helper()
	w, err := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	if err != nil {
		t.Fatal(err)
	}
	b, err := tensor.FromFloat32([]float32{0.1, 0.2}, []int64{2})
	if err != nil {
		t.Fatal(err)
	}
	return []*nn.Parameter{
		nn.NewParameter(w, "weight"),
		nn.NewParameter(b, "bias"),
	}
}

// cloneParamData copies parameter data from src to dst so they have
// identical values (needed when testing optimizer state restore —
// the optimizer state alone isn't enough, the params must match too).
func cloneParamData(t *testing.T, dst, src []*nn.Parameter) {
	t.Helper()
	for i := range dst {
		data := tensorData(t, src[i].Data())
		clone, err := tensor.FromFloat32(data, src[i].Data().Shape())
		if err != nil {
			t.Fatal(err)
		}
		dst[i].SetData(clone)
	}
}

// --- SGD state ---

func TestSGDSaveLoadState(t *testing.T) {
	params := makeParams(t)
	opt := nn.NewSGD(params, 0.01, 0.9)

	// Step a few times to build up momentum.
	for range 3 {
		stepOptimizer(t, params, opt)
	}

	// Save state.
	var buf bytes.Buffer
	if err := opt.SaveState(&buf); err != nil {
		t.Fatal(err)
	}

	// Create a fresh optimizer, copy param data, and load optimizer state.
	params2 := makeParams(t)
	cloneParamData(t, params2, params)
	opt2 := nn.NewSGD(params2, 0.05, 0.9) // different lr to verify it's overwritten
	if err := opt2.LoadState(&buf); err != nil {
		t.Fatal(err)
	}

	// Verify LR was restored.
	if opt2.LR() != 0.01 {
		t.Errorf("restored LR = %g, want 0.01", opt2.LR())
	}

	// Step both and verify they produce the same result.
	stepOptimizer(t, params, opt)
	stepOptimizer(t, params2, opt2)

	for i := range params {
		got := tensorData(t, params2[i].Data())
		want := tensorData(t, params[i].Data())
		assertClose(t, params[i].Name, got, want)
	}
}

func TestSGDSaveLoadStateNoMomentum(t *testing.T) {
	params := makeParams(t)
	opt := nn.NewSGD(params, 0.01, 0) // no momentum

	stepOptimizer(t, params, opt)

	var buf bytes.Buffer
	if err := opt.SaveState(&buf); err != nil {
		t.Fatal(err)
	}

	params2 := makeParams(t)
	cloneParamData(t, params2, params)
	opt2 := nn.NewSGD(params2, 0.01, 0)
	if err := opt2.LoadState(&buf); err != nil {
		t.Fatal(err)
	}

	stepOptimizer(t, params, opt)
	stepOptimizer(t, params2, opt2)

	for i := range params {
		assertClose(t, params[i].Name, tensorData(t, params2[i].Data()), tensorData(t, params[i].Data()))
	}
}

// --- Adam state ---

func TestAdamSaveLoadState(t *testing.T) {
	params := makeParams(t)
	opt := nn.NewAdam(params, 0.001)

	for range 5 {
		stepOptimizer(t, params, opt)
	}

	var buf bytes.Buffer
	if err := opt.SaveState(&buf); err != nil {
		t.Fatal(err)
	}

	params2 := makeParams(t)
	cloneParamData(t, params2, params)
	opt2 := nn.NewAdam(params2, 0.1) // different lr
	if err := opt2.LoadState(&buf); err != nil {
		t.Fatal(err)
	}

	if opt2.LR() != 0.001 {
		t.Errorf("restored LR = %g, want 0.001", opt2.LR())
	}

	// Step both and compare — the bias correction depends on t being
	// identical, so this is a strong test.
	stepOptimizer(t, params, opt)
	stepOptimizer(t, params2, opt2)

	for i := range params {
		assertClose(t, params[i].Name, tensorData(t, params2[i].Data()), tensorData(t, params[i].Data()))
	}
}

func TestAdamSaveLoadStateBeforeFirstStep(t *testing.T) {
	params := makeParams(t)
	opt := nn.NewAdam(params, 0.001)

	// Save before any step — m and v are nil.
	var buf bytes.Buffer
	if err := opt.SaveState(&buf); err != nil {
		t.Fatal(err)
	}

	params2 := makeParams(t)
	opt2 := nn.NewAdam(params2, 0.001)
	if err := opt2.LoadState(&buf); err != nil {
		t.Fatal(err)
	}

	// Both optimizers should behave identically from step 1.
	stepOptimizer(t, params, opt)
	stepOptimizer(t, params2, opt2)

	for i := range params {
		assertClose(t, params[i].Name, tensorData(t, params2[i].Data()), tensorData(t, params[i].Data()))
	}
}

// --- AdamW state ---

func TestAdamWSaveLoadState(t *testing.T) {
	params := makeParams(t)
	opt := nn.NewAdamW(params, 0.001, 0.01)

	for range 5 {
		stepOptimizer(t, params, opt)
	}

	var buf bytes.Buffer
	if err := opt.SaveState(&buf); err != nil {
		t.Fatal(err)
	}

	params2 := makeParams(t)
	cloneParamData(t, params2, params)
	opt2 := nn.NewAdamW(params2, 0.1, 0.01) // different lr
	if err := opt2.LoadState(&buf); err != nil {
		t.Fatal(err)
	}

	if opt2.LR() != 0.001 {
		t.Errorf("restored LR = %g, want 0.001", opt2.LR())
	}

	stepOptimizer(t, params, opt)
	stepOptimizer(t, params2, opt2)

	for i := range params {
		assertClose(t, params[i].Name, tensorData(t, params2[i].Data()), tensorData(t, params[i].Data()))
	}
}

// --- Scheduler state ---

func TestStepDecaySchedulerSaveLoadState(t *testing.T) {
	opt := nn.NewAdam(makeParams(t), 0.1)
	sched := nn.NewStepDecayScheduler(opt, 10, 0.5)

	// Advance 25 ticks (should have decayed twice: 0.1 * 0.5^2 = 0.025).
	for range 25 {
		sched.Step()
	}
	lrBefore := sched.LR()

	var buf bytes.Buffer
	if err := sched.SaveState(&buf); err != nil {
		t.Fatal(err)
	}

	opt2 := nn.NewAdam(makeParams(t), 0.1)
	sched2 := nn.NewStepDecayScheduler(opt2, 10, 0.5)
	if err := sched2.LoadState(&buf); err != nil {
		t.Fatal(err)
	}

	// Step both once more and compare.
	sched.Step()
	sched2.Step()

	if math.Abs(sched.LR()-sched2.LR()) > 1e-12 {
		t.Errorf("LR mismatch: original=%g restored=%g", sched.LR(), sched2.LR())
	}
	if math.Abs(lrBefore-0.025) > 1e-12 {
		t.Errorf("LR before save = %g, want 0.025", lrBefore)
	}
}

func TestCosineSchedulerSaveLoadState(t *testing.T) {
	opt := nn.NewAdam(makeParams(t), 0.1)
	sched := nn.NewCosineScheduler(opt, 0.1, 0.001, 100)

	for range 42 {
		sched.Step()
	}

	var buf bytes.Buffer
	if err := sched.SaveState(&buf); err != nil {
		t.Fatal(err)
	}

	opt2 := nn.NewAdam(makeParams(t), 0.1)
	sched2 := nn.NewCosineScheduler(opt2, 0.1, 0.001, 100)
	if err := sched2.LoadState(&buf); err != nil {
		t.Fatal(err)
	}

	sched.Step()
	sched2.Step()

	if math.Abs(sched.LR()-sched2.LR()) > 1e-12 {
		t.Errorf("LR mismatch: original=%g restored=%g", sched.LR(), sched2.LR())
	}
}

func TestWarmupSchedulerSaveLoadState(t *testing.T) {
	opt := nn.NewAdam(makeParams(t), 0.1)
	inner := nn.NewCosineScheduler(opt, 0.1, 0.001, 100)
	sched := nn.NewWarmupScheduler(opt, inner, 0.1, 10)

	// Advance past warmup into inner scheduler territory.
	for range 30 {
		sched.Step()
	}

	var buf bytes.Buffer
	if err := sched.SaveState(&buf); err != nil {
		t.Fatal(err)
	}

	opt2 := nn.NewAdam(makeParams(t), 0.1)
	inner2 := nn.NewCosineScheduler(opt2, 0.1, 0.001, 100)
	sched2 := nn.NewWarmupScheduler(opt2, inner2, 0.1, 10)
	if err := sched2.LoadState(&buf); err != nil {
		t.Fatal(err)
	}

	// Step both 10 more times and compare.
	for range 10 {
		sched.Step()
		sched2.Step()
	}

	if math.Abs(sched.LR()-sched2.LR()) > 1e-12 {
		t.Errorf("LR mismatch: original=%g restored=%g", sched.LR(), sched2.LR())
	}
}

func TestPlateauSchedulerSaveLoadState(t *testing.T) {
	opt := nn.NewAdam(makeParams(t), 0.1)
	sched := nn.NewPlateauScheduler(opt, 3, 0.5, 1e-6)

	// Feed observations to build up state.
	sched.Observe(1.0) // best=1.0
	sched.Observe(0.9) // best=0.9
	sched.Observe(1.1) // wait=1
	sched.Observe(1.2) // wait=2

	var buf bytes.Buffer
	if err := sched.SaveState(&buf); err != nil {
		t.Fatal(err)
	}

	opt2 := nn.NewAdam(makeParams(t), 0.1)
	sched2 := nn.NewPlateauScheduler(opt2, 3, 0.5, 1e-6)
	if err := sched2.LoadState(&buf); err != nil {
		t.Fatal(err)
	}

	// One more non-improving observation should trigger decay on both.
	sched.Observe(1.3)  // wait=3 → decay
	sched2.Observe(1.3) // same

	if math.Abs(sched.LR()-sched2.LR()) > 1e-12 {
		t.Errorf("LR mismatch: original=%g restored=%g", sched.LR(), sched2.LR())
	}
	expected := 0.1 * 0.5 // one decay from 0.1
	if math.Abs(sched.LR()-expected) > 1e-12 {
		t.Errorf("LR = %g, want %g", sched.LR(), expected)
	}
}

// --- GradScaler state ---

func TestGradScalerSaveLoadState(t *testing.T) {
	scaler := nn.NewGradScaler()

	// Simulate some updates to change state.
	for range 50 {
		scaler.Update()
	}
	scaleBefore := scaler.ScaleFactor()

	var buf bytes.Buffer
	if err := scaler.SaveState(&buf); err != nil {
		t.Fatal(err)
	}

	scaler2 := nn.NewGradScaler()
	if err := scaler2.LoadState(&buf); err != nil {
		t.Fatal(err)
	}

	if scaler2.ScaleFactor() != scaleBefore {
		t.Errorf("scale = %g, want %g", scaler2.ScaleFactor(), scaleBefore)
	}

	// Step both and verify they stay in sync.
	scaler.Update()
	scaler2.Update()
	if scaler.ScaleFactor() != scaler2.ScaleFactor() {
		t.Errorf("after update: original=%g restored=%g", scaler.ScaleFactor(), scaler2.ScaleFactor())
	}
}

// --- Param count mismatch ---

func TestAdamLoadStateMismatch(t *testing.T) {
	params := makeParams(t)
	opt := nn.NewAdam(params, 0.001)
	stepOptimizer(t, params, opt)

	var buf bytes.Buffer
	if err := opt.SaveState(&buf); err != nil {
		t.Fatal(err)
	}

	// Try to load into optimizer with different param count.
	w, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	singleParam := []*nn.Parameter{nn.NewParameter(w, "single")}
	opt2 := nn.NewAdam(singleParam, 0.001)

	if err := opt2.LoadState(&buf); err == nil {
		t.Fatal("expected error on param count mismatch")
	}
}

// --- Checkpoint integration ---

func TestCheckpointSaveLoad(t *testing.T) {
	dir := t.TempDir()

	// Build a model.
	linear, err := nn.NewLinear(3, 2)
	if err != nil {
		t.Fatal(err)
	}

	opt := nn.NewAdam(linear.Parameters(), 0.001)
	sched := nn.NewCosineScheduler(opt, 0.001, 0, 100)

	// Train a few steps.
	for range 10 {
		xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{1, 3})
		yt, _ := tensor.FromFloat32([]float32{1, 0}, []int64{1, 2})
		pred := linear.Forward(autograd.NewVariable(xt, false))
		loss := nn.MSELoss(pred, autograd.NewVariable(yt, false))
		if err := loss.Backward(); err != nil {
			t.Fatal(err)
		}
		opt.Step()
		opt.ZeroGrad()
		sched.Step()
	}

	// Snapshot parameter values.
	origW := tensorData(t, linear.Parameters()[0].Data())
	origLR := opt.LR()

	// Save checkpoint.
	ckpt := nn.NewCheckpoint(dir + "/run").
		Model(linear).
		Add("optimizer", opt).
		Add("scheduler", sched)

	if err := ckpt.Save(42); err != nil {
		t.Fatal(err)
	}

	// Create fresh model + optimizer + scheduler.
	linear2, err := nn.NewLinear(3, 2)
	if err != nil {
		t.Fatal(err)
	}
	opt2 := nn.NewAdam(linear2.Parameters(), 0.1)
	sched2 := nn.NewCosineScheduler(opt2, 0.001, 0, 100)

	ckpt2 := nn.NewCheckpoint(dir + "/run").
		Model(linear2).
		Add("optimizer", opt2).
		Add("scheduler", sched2)

	epoch, err := ckpt2.Load()
	if err != nil {
		t.Fatal(err)
	}

	if epoch != 42 {
		t.Errorf("epoch = %d, want 42", epoch)
	}

	// Verify parameters restored.
	restoredW := tensorData(t, linear2.Parameters()[0].Data())
	assertClose(t, "weight", restoredW, origW)

	// Verify optimizer LR restored.
	if math.Abs(opt2.LR()-origLR) > 1e-12 {
		t.Errorf("LR = %g, want %g", opt2.LR(), origLR)
	}

	// Step both and compare — proves optimizer state (m, v, t) was restored.
	xt, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{1, 3})
	yt, _ := tensor.FromFloat32([]float32{1, 0}, []int64{1, 2})

	pred := linear.Forward(autograd.NewVariable(xt, false))
	loss := nn.MSELoss(pred, autograd.NewVariable(yt, false))
	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}
	opt.Step()

	xt2, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{1, 3})
	yt2, _ := tensor.FromFloat32([]float32{1, 0}, []int64{1, 2})
	pred2 := linear2.Forward(autograd.NewVariable(xt2, false))
	loss2 := nn.MSELoss(pred2, autograd.NewVariable(yt2, false))
	if err := loss2.Backward(); err != nil {
		t.Fatal(err)
	}
	opt2.Step()

	for i, p := range linear.Parameters() {
		got := tensorData(t, linear2.Parameters()[i].Data())
		want := tensorData(t, p.Data())
		assertClose(t, p.Name, got, want)
	}
}

func TestCheckpointLoadEpoch(t *testing.T) {
	dir := t.TempDir()

	linear, err := nn.NewLinear(2, 1)
	if err != nil {
		t.Fatal(err)
	}

	ckpt := nn.NewCheckpoint(dir + "/model").Model(linear)

	// Save multiple epochs.
	for _, epoch := range []int{10, 20, 30} {
		if err := ckpt.Save(epoch); err != nil {
			t.Fatal(err)
		}
	}

	// Load latest should give 30.
	linear2, _ := nn.NewLinear(2, 1)
	ckpt2 := nn.NewCheckpoint(dir + "/model").Model(linear2)
	epoch, err := ckpt2.Load()
	if err != nil {
		t.Fatal(err)
	}
	if epoch != 30 {
		t.Errorf("latest epoch = %d, want 30", epoch)
	}

	// Load specific epoch.
	linear3, _ := nn.NewLinear(2, 1)
	ckpt3 := nn.NewCheckpoint(dir + "/model").Model(linear3)
	if err := ckpt3.LoadEpoch(20); err != nil {
		t.Fatal(err)
	}
}

func TestCheckpointNoModel(t *testing.T) {
	dir := t.TempDir()

	opt := nn.NewAdam(makeParams(t), 0.001)
	for range 5 {
		stepOptimizer(t, makeParams(t), opt)
	}

	ckpt := nn.NewCheckpoint(dir + "/state").Add("optimizer", opt)
	if err := ckpt.Save(7); err != nil {
		t.Fatal(err)
	}

	opt2 := nn.NewAdam(makeParams(t), 0.1)
	ckpt2 := nn.NewCheckpoint(dir + "/state").Add("optimizer", opt2)
	epoch, err := ckpt2.Load()
	if err != nil {
		t.Fatal(err)
	}
	if epoch != 7 {
		t.Errorf("epoch = %d, want 7", epoch)
	}
	if opt2.LR() != 0.001 {
		t.Errorf("LR = %g, want 0.001", opt2.LR())
	}
}

func TestCheckpointStateMismatch(t *testing.T) {
	dir := t.TempDir()

	opt := nn.NewAdam(makeParams(t), 0.001)
	ckpt := nn.NewCheckpoint(dir + "/state").
		Add("optimizer", opt).
		Add("scheduler", nn.NewCosineScheduler(opt, 0.001, 0, 100))
	if err := ckpt.Save(1); err != nil {
		t.Fatal(err)
	}

	// Load with different component count.
	opt2 := nn.NewAdam(makeParams(t), 0.001)
	ckpt2 := nn.NewCheckpoint(dir + "/state").Add("optimizer", opt2)
	if _, err := ckpt2.Load(); err == nil {
		t.Fatal("expected error on state count mismatch")
	}
}

func TestCheckpointNameMismatch(t *testing.T) {
	dir := t.TempDir()

	opt := nn.NewAdam(makeParams(t), 0.001)
	ckpt := nn.NewCheckpoint(dir + "/state").Add("optimizer", opt)
	if err := ckpt.Save(1); err != nil {
		t.Fatal(err)
	}

	// Load with different name.
	opt2 := nn.NewAdam(makeParams(t), 0.001)
	ckpt2 := nn.NewCheckpoint(dir + "/state").Add("wrong_name", opt2)
	if _, err := ckpt2.Load(); err == nil {
		t.Fatal("expected error on name mismatch")
	}
}

func TestCheckpointNotFound(t *testing.T) {
	dir := t.TempDir()
	ckpt := nn.NewCheckpoint(dir + "/nonexistent")
	if _, err := ckpt.Load(); err == nil {
		t.Fatal("expected error when no checkpoints exist")
	}
}
