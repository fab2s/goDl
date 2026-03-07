package nn_test

import (
	"math"
	"testing"

	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// makeDummyOpt creates an Adam optimizer with a single parameter for scheduler tests.
func makeDummyOpt(t *testing.T, lr float64) *nn.Adam {
	t.Helper()
	data, err := tensor.FromFloat32([]float32{1.0}, []int64{1})
	if err != nil {
		t.Fatal(err)
	}
	p := nn.NewParameter(data, "dummy")
	return nn.NewAdam([]*nn.Parameter{p}, lr)
}

func assertLRClose(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-8 {
		t.Errorf("%s: lr=%.10f, want %.10f (diff=%.2e)", name, got, want, math.Abs(got-want))
	}
}

// --- LRAdjustable ---

func TestSGDLRAdjustable(t *testing.T) {
	data, _ := tensor.FromFloat32([]float32{1.0}, []int64{1})
	p := nn.NewParameter(data, "w")
	opt := nn.NewSGD([]*nn.Parameter{p}, 0.01, 0)
	assertLRClose(t, "initial", opt.LR(), 0.01)
	opt.SetLR(0.005)
	assertLRClose(t, "after set", opt.LR(), 0.005)
}

func TestAdamLRAdjustable(t *testing.T) {
	opt := makeDummyOpt(t, 0.001)
	assertLRClose(t, "initial", opt.LR(), 0.001)
	opt.SetLR(0.0005)
	assertLRClose(t, "after set", opt.LR(), 0.0005)
}

func TestAdamWLRAdjustable(t *testing.T) {
	data, _ := tensor.FromFloat32([]float32{1.0}, []int64{1})
	p := nn.NewParameter(data, "w")
	opt := nn.NewAdamW([]*nn.Parameter{p}, 0.001, 0.01)
	assertLRClose(t, "initial", opt.LR(), 0.001)
	opt.SetLR(0.0005)
	assertLRClose(t, "after set", opt.LR(), 0.0005)
}

// --- Step Decay ---

func TestStepDecayScheduler(t *testing.T) {
	opt := makeDummyOpt(t, 0.1)
	sched := nn.NewStepDecayScheduler(opt, 3, 0.5)

	// Steps 1-3: lr = 0.1 * 0.5^0 = 0.1 (n=0 for tick 1,2; n=1 at tick 3)
	sched.Step() // tick=1, n=0
	assertLRClose(t, "tick1", sched.LR(), 0.1)
	sched.Step() // tick=2, n=0
	assertLRClose(t, "tick2", sched.LR(), 0.1)
	sched.Step() // tick=3, n=1 → 0.1 * 0.5 = 0.05
	assertLRClose(t, "tick3", sched.LR(), 0.05)

	// Steps 4-6
	sched.Step() // tick=4, n=1
	assertLRClose(t, "tick4", sched.LR(), 0.05)
	sched.Step() // tick=5, n=1
	assertLRClose(t, "tick5", sched.LR(), 0.05)
	sched.Step() // tick=6, n=2 → 0.1 * 0.25 = 0.025
	assertLRClose(t, "tick6", sched.LR(), 0.025)
}

// --- Cosine Annealing ---

func TestCosineScheduler(t *testing.T) {
	opt := makeDummyOpt(t, 0.1)
	baseLR := 0.1
	minLR := 0.0
	total := 100
	sched := nn.NewCosineScheduler(opt, baseLR, minLR, total)

	assertLRClose(t, "initial", sched.LR(), baseLR)

	// Step to midpoint: lr should be ~0.05 (cos(pi/2) = 0)
	for i := 0; i < 50; i++ {
		sched.Step()
	}
	midLR := minLR + 0.5*(baseLR-minLR)*(1+math.Cos(0.5*math.Pi))
	assertLRClose(t, "midpoint", sched.LR(), midLR)

	// Step to end: lr should be minLR
	for i := 50; i < 100; i++ {
		sched.Step()
	}
	assertLRClose(t, "end", sched.LR(), minLR)

	// Beyond total: stays at minLR
	sched.Step()
	assertLRClose(t, "beyond", sched.LR(), minLR)
}

func TestCosineSchedulerWithMinLR(t *testing.T) {
	opt := makeDummyOpt(t, 0.1)
	sched := nn.NewCosineScheduler(opt, 0.1, 0.01, 100)

	for i := 0; i < 100; i++ {
		sched.Step()
	}
	assertLRClose(t, "end", sched.LR(), 0.01)
}

// --- Linear Warmup ---

func TestWarmupScheduler(t *testing.T) {
	opt := makeDummyOpt(t, 0.1)
	targetLR := 0.1
	warmup := 10
	total := 100

	inner := nn.NewCosineScheduler(opt, targetLR, 0.0, total-warmup)
	sched := nn.NewWarmupScheduler(opt, inner, targetLR, warmup)

	// LR starts at 0.
	assertLRClose(t, "start", sched.LR(), 0)

	// Linear ramp: step 5 → 0.05
	for i := 0; i < 5; i++ {
		sched.Step()
	}
	assertLRClose(t, "mid_warmup", sched.LR(), 0.05)

	// End of warmup: step 10 → targetLR
	for i := 5; i < 10; i++ {
		sched.Step()
	}
	assertLRClose(t, "end_warmup", sched.LR(), targetLR)

	// After warmup, inner scheduler takes over (cosine from targetLR to 0).
	sched.Step() // tick 11 → inner.Step() called once
	if sched.LR() >= targetLR {
		t.Errorf("expected lr < %f after warmup, got %f", targetLR, sched.LR())
	}
}

// --- Reduce on Plateau ---

func TestPlateauScheduler(t *testing.T) {
	opt := makeDummyOpt(t, 0.1)
	sched := nn.NewPlateauScheduler(opt, 3, 0.5, 1e-6)

	// First observation sets the baseline.
	sched.Observe(1.0)
	assertLRClose(t, "baseline", sched.LR(), 0.1)

	// Improving metric resets patience.
	sched.Observe(0.9)
	assertLRClose(t, "improved", sched.LR(), 0.1)

	// No improvement for 3 observations.
	sched.Observe(0.95) // wait=1
	sched.Observe(0.95) // wait=2
	sched.Observe(0.95) // wait=3 → reduce
	assertLRClose(t, "reduced", sched.LR(), 0.05)

	// Improve again, then stagnate.
	sched.Observe(0.8) // new best, reset wait
	sched.Observe(0.9) // wait=1
	sched.Observe(0.9) // wait=2
	sched.Observe(0.9) // wait=3 → reduce
	assertLRClose(t, "reduced_again", sched.LR(), 0.025)
}

func TestPlateauSchedulerMinLR(t *testing.T) {
	opt := makeDummyOpt(t, 0.001)
	sched := nn.NewPlateauScheduler(opt, 1, 0.1, 0.0005)

	sched.Observe(1.0) // baseline
	sched.Observe(1.0) // wait=1 → reduce: 0.001 * 0.1 = 0.0001 → clamped to 0.0005
	assertLRClose(t, "clamped", sched.LR(), 0.0005)

	// Already at min, should not go lower.
	sched.Observe(1.0) // wait=1 → reduce: 0.0005 * 0.1 = 0.00005 → clamped to 0.0005
	assertLRClose(t, "still_min", sched.LR(), 0.0005)
}
