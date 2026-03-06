package nn

import "github.com/fab2s/goDl/autograd"

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
// pred: [batch, classes] — raw scores (logits), not probabilities
// target: [batch, classes] — one-hot encoded targets
//
// Computes: -mean(sum(target * log_softmax(pred), dim=1))
//
// Uses the log-sum-exp trick for numerical stability:
//
//	log_softmax(x) = x - log(sum(exp(x - max(x))))
func CrossEntropyLoss(pred, target *autograd.Variable) *autograd.Variable {
	// Numerically stable log-softmax:
	// 1. max_val = max(pred, dim=1, keepdim=true)
	// 2. shifted = pred - max_val
	// 3. log_softmax = shifted - log(sum(exp(shifted), dim=1, keepdim=true))

	// We need MaxDim on Variable. For now, use the tensor directly
	// and wrap as a detached constant (max doesn't need gradients through it).
	maxVal := autograd.NewVariable(pred.Data().MaxDim(1, true), false)
	shifted := pred.Sub(maxVal)

	expShifted := shifted.Exp()
	sumExp := expShifted.SumDim(1, true)
	logSumExp := sumExp.Log()
	logSoftmax := shifted.Sub(logSumExp)

	// Cross-entropy: -mean(sum(target * logSoftmax, dim=1))
	perSample := target.Mul(logSoftmax).SumDim(1, false)
	batchSize := float64(pred.Data().Shape()[0])
	return perSample.Sum().MulScalar(-1.0 / batchSize)
}
