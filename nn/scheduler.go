package nn

import "math"

// Scheduler adjusts the learning rate of an optimizer over training.
// Call Step() once per step (or once per epoch, depending on the schedule).
//
//	scheduler := nn.NewCosineScheduler(optimizer, 0.001, 1000)
//	for step := 0; step < 1000; step++ {
//	    // ... forward, backward ...
//	    optimizer.Step()
//	    scheduler.Step()
//	}
type Scheduler interface {
	Step()       // advance the schedule by one tick
	LR() float64 // current learning rate
}

// --- Step Decay ---

// StepDecayScheduler multiplies the learning rate by gamma every
// stepSize ticks. Classic staircase decay.
//
//	// Decay by 0.1x every 30 epochs.
//	scheduler := nn.NewStepDecayScheduler(opt, 30, 0.1)
type StepDecayScheduler struct {
	opt      LRAdjustable
	baseLR   float64
	stepSize int
	gamma    float64
	tick     int
}

// NewStepDecayScheduler creates a scheduler that multiplies lr by gamma
// every stepSize ticks.
func NewStepDecayScheduler(opt LRAdjustable, stepSize int, gamma float64) *StepDecayScheduler {
	return &StepDecayScheduler{
		opt:      opt,
		baseLR:   opt.LR(),
		stepSize: stepSize,
		gamma:    gamma,
	}
}

func (s *StepDecayScheduler) Step() {
	s.tick++
	n := s.tick / s.stepSize
	lr := s.baseLR * math.Pow(s.gamma, float64(n))
	s.opt.SetLR(lr)
}

func (s *StepDecayScheduler) LR() float64 { return s.opt.LR() }

// --- Cosine Annealing ---

// CosineScheduler anneals the learning rate from baseLR to minLR
// following a cosine curve over totalSteps ticks. After totalSteps,
// the rate stays at minLR.
//
//	scheduler := nn.NewCosineScheduler(opt, 0.001, 0, 10000)
type CosineScheduler struct {
	opt        LRAdjustable
	baseLR     float64
	minLR      float64
	totalSteps int
	tick       int
}

// NewCosineScheduler creates a cosine annealing scheduler.
func NewCosineScheduler(opt LRAdjustable, baseLR, minLR float64, totalSteps int) *CosineScheduler {
	opt.SetLR(baseLR)
	return &CosineScheduler{
		opt:        opt,
		baseLR:     baseLR,
		minLR:      minLR,
		totalSteps: totalSteps,
	}
}

func (s *CosineScheduler) Step() {
	s.tick++
	if s.tick >= s.totalSteps {
		s.opt.SetLR(s.minLR)
		return
	}
	progress := float64(s.tick) / float64(s.totalSteps)
	lr := s.minLR + 0.5*(s.baseLR-s.minLR)*(1+math.Cos(progress*math.Pi))
	s.opt.SetLR(lr)
}

func (s *CosineScheduler) LR() float64 { return s.opt.LR() }

// --- Linear Warmup ---

// WarmupScheduler linearly increases the learning rate from 0 to
// targetLR over warmupSteps ticks, then delegates to an inner
// scheduler for the remaining training.
//
//	inner := nn.NewCosineScheduler(opt, 0.001, 0, 10000)
//	scheduler := nn.NewWarmupScheduler(opt, inner, 0.001, 500)
type WarmupScheduler struct {
	opt         LRAdjustable
	inner       Scheduler
	targetLR    float64
	warmupSteps int
	tick        int
}

// NewWarmupScheduler creates a linear warmup followed by an inner schedule.
// During warmup, lr ramps linearly from 0 to targetLR. After warmup, the
// inner scheduler takes over (its Step is called for every tick after warmup).
func NewWarmupScheduler(opt LRAdjustable, inner Scheduler, targetLR float64, warmupSteps int) *WarmupScheduler {
	opt.SetLR(0)
	return &WarmupScheduler{
		opt:         opt,
		inner:       inner,
		targetLR:    targetLR,
		warmupSteps: warmupSteps,
	}
}

func (s *WarmupScheduler) Step() {
	s.tick++
	if s.tick <= s.warmupSteps {
		lr := s.targetLR * float64(s.tick) / float64(s.warmupSteps)
		s.opt.SetLR(lr)
		return
	}
	s.inner.Step()
}

func (s *WarmupScheduler) LR() float64 { return s.opt.LR() }

// --- Reduce on Plateau ---

// PlateauScheduler reduces the learning rate when a metric stops
// improving. Call Observe(metric) instead of Step() — typically
// once per epoch with the validation loss.
//
//	scheduler := nn.NewPlateauScheduler(opt, 10, 0.1, 1e-6)
//	for epoch := range epochs {
//	    valLoss := evaluate()
//	    scheduler.Observe(valLoss)
//	}
type PlateauScheduler struct {
	opt      LRAdjustable
	patience int
	factor   float64
	minLR    float64
	best     float64
	wait     int
	started  bool
}

// NewPlateauScheduler creates a reduce-on-plateau scheduler.
// When the metric does not improve for patience observations, the
// learning rate is multiplied by factor. LR never goes below minLR.
func NewPlateauScheduler(opt LRAdjustable, patience int, factor, minLR float64) *PlateauScheduler {
	return &PlateauScheduler{
		opt:      opt,
		patience: patience,
		factor:   factor,
		minLR:    minLR,
	}
}

// Observe reports a metric value (lower is better, e.g. validation loss).
// If the metric has not improved for patience calls, the LR is reduced.
func (s *PlateauScheduler) Observe(metric float64) {
	if !s.started || metric < s.best {
		s.best = metric
		s.wait = 0
		s.started = true
		return
	}
	s.wait++
	if s.wait >= s.patience {
		lr := s.opt.LR() * s.factor
		if lr < s.minLR {
			lr = s.minLR
		}
		s.opt.SetLR(lr)
		s.wait = 0
	}
}

// Step is a no-op for PlateauScheduler. Use Observe(metric) instead.
func (s *PlateauScheduler) Step() {}

func (s *PlateauScheduler) LR() float64 { return s.opt.LR() }
