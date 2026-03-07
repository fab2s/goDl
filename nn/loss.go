package nn

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// MSELoss computes mean squared error: mean((pred - target)²).
// Both inputs must have the same shape. Returns a scalar variable.
func MSELoss(pred, target *autograd.Variable) *autograd.Variable {
	diff := pred.Sub(target)
	sq := diff.Mul(diff)
	// Mean = sum / numel
	n := float64(pred.Data().Numel())
	return sq.Sum().MulScalar(1.0 / n)
}

// CrossEntropyLoss computes cross-entropy loss from raw logits.
//
// pred: [batch, classes] — raw scores (logits), not probabilities.
//
// target accepts two formats:
//   - [batch] int64 class indices (like PyTorch) — preferred
//   - [batch, classes] one-hot encoded float targets
//
// Computes: -mean(sum(target * log_softmax(pred), dim=1))
//
// Uses the log-sum-exp trick for numerical stability.
func CrossEntropyLoss(pred, target *autograd.Variable) *autograd.Variable {
	targetData := target.Data()
	targetDims := targetData.Ndim()

	// Auto-detect: 1D target = class indices, 2D = one-hot.
	var oneHotTarget *autograd.Variable
	if targetDims == 1 {
		// Convert integer class indices [B] to one-hot [B, C].
		predShape := pred.Data().Shape()
		B, C := predShape[0], predShape[1]
		oneHotTarget = autograd.NewVariable(tensor.OneHot(targetData, C, B), false)
	} else {
		oneHotTarget = target
	}

	// Numerically stable log-softmax.
	maxVal := autograd.NewVariable(pred.Data().MaxDim(1, true), false)
	shifted := pred.Sub(maxVal)

	expShifted := shifted.Exp()
	sumExp := expShifted.SumDim(1, true)
	logSumExp := sumExp.Log()
	logSoftmax := shifted.Sub(logSumExp)

	// Cross-entropy: -mean(sum(target * logSoftmax, dim=1))
	perSample := oneHotTarget.Mul(logSoftmax).SumDim(1, false)
	batchSize := float64(pred.Data().Shape()[0])
	return perSample.Sum().MulScalar(-1.0 / batchSize)
}

// BCEWithLogitsLoss computes binary cross-entropy from raw logits (not probabilities).
//
// pred and target must have the same shape. Target values should be 0 or 1.
//
// Uses the numerically stable formula:
//
//	loss = mean(max(x, 0) - x*t + log(1 + exp(-|x|)))
func BCEWithLogitsLoss(pred, target *autograd.Variable) *autograd.Variable {
	relu := pred.ReLU()
	xt := pred.Mul(target)
	absX := pred.Abs()
	logTerm := absX.Neg().Exp().AddScalar(1.0).Log()
	n := float64(pred.Data().Numel())
	return relu.Sub(xt).Add(logTerm).Sum().MulScalar(1.0 / n)
}

// L1Loss computes mean absolute error: mean(|pred - target|).
func L1Loss(pred, target *autograd.Variable) *autograd.Variable {
	diff := pred.Sub(target)
	n := float64(pred.Data().Numel())
	return diff.Abs().Sum().MulScalar(1.0 / n)
}

// SmoothL1Loss computes the Huber loss with transition point beta.
//
//	loss = mean(smooth_l1(pred - target))
//
// where smooth_l1(x) = 0.5 * x² / beta if |x| < beta, else |x| - 0.5 * beta.
func SmoothL1Loss(pred, target *autograd.Variable, beta float64) *autograd.Variable {
	diff := pred.Sub(target)
	absDiff := diff.Abs()

	// Quadratic region: 0.5 * x^2 / beta
	quadratic := diff.Mul(diff).MulScalar(0.5 / beta)
	// Linear region: |x| - 0.5 * beta
	linear := absDiff.AddScalar(-0.5 * beta)

	// Mask: 1.0 where |diff| < beta (quadratic), 0.0 elsewhere (linear).
	mask := autograd.NewVariable(absDiff.Data().LTScalar(beta), false)
	notMask := autograd.NewVariable(absDiff.Data().GEScalar(beta), false)

	perElement := mask.Mul(quadratic).Add(notMask.Mul(linear))
	n := float64(pred.Data().Numel())
	return perElement.Sum().MulScalar(1.0 / n)
}

// KLDivLoss computes the Kullback-Leibler divergence: mean(target * (log(target) - input)).
//
// input should be log-probabilities, target should be probabilities.
// This matches PyTorch's KLDivLoss with reduction='batchmean'.
func KLDivLoss(input, target *autograd.Variable) *autograd.Variable {
	// KL(target || exp(input)) = sum(target * (log(target) - input))
	// log(target) is computed from the target (no gradient needed through it).
	logTarget := autograd.NewVariable(target.Data().Clamp(1e-8, 1e30).Log(), false)
	perElement := target.Mul(logTarget.Sub(input))
	batchSize := float64(input.Data().Shape()[0])
	return perElement.Sum().MulScalar(1.0 / batchSize)
}
