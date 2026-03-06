package nn

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// BatchNorm applies batch normalization over the feature dimension.
// Input shape: [batch, features].
//
// During training, normalizes using batch statistics and updates
// exponential moving averages of mean and variance. During inference,
// uses the accumulated running statistics.
//
// Computes: output = gamma * (x - mean) / sqrt(var + eps) + beta
//
//	bn, _ := nn.NewBatchNorm(128)
//	output := bn.Forward(input)
type BatchNorm struct {
	Weight      *Parameter
	Bias        *Parameter
	RunningMean *tensor.Tensor
	RunningVar  *tensor.Tensor
	numFeatures int64
	eps         float64
	momentum    float64
	training    bool
}

// NewBatchNorm creates a BatchNorm module for the given feature size.
// Weight (gamma) is initialized to ones, bias (beta) to zeros.
// Running statistics start at zero mean and unit variance.
func NewBatchNorm(numFeatures int64, opts ...tensor.Option) (*BatchNorm, error) {
	weight, err := tensor.Ones([]int64{numFeatures}, opts...)
	if err != nil {
		return nil, err
	}
	bias, err := tensor.Zeros([]int64{numFeatures}, opts...)
	if err != nil {
		weight.Release()
		return nil, err
	}
	runningMean, err := tensor.Zeros([]int64{numFeatures}, opts...)
	if err != nil {
		weight.Release()
		bias.Release()
		return nil, err
	}
	runningVar, err := tensor.Ones([]int64{numFeatures}, opts...)
	if err != nil {
		weight.Release()
		bias.Release()
		runningMean.Release()
		return nil, err
	}

	return &BatchNorm{
		Weight:      NewParameter(weight, "weight"),
		Bias:        NewParameter(bias, "bias"),
		RunningMean: runningMean,
		RunningVar:  runningVar,
		numFeatures: numFeatures,
		eps:         1e-5,
		momentum:    0.1,
		training:    true,
	}, nil
}

// SetTraining enables or disables training mode.
// In training mode, batch statistics are used and running stats are updated.
// In eval mode, running statistics are used for normalization.
func (bn *BatchNorm) SetTraining(training bool) {
	bn.training = training
}

// Forward applies batch normalization.
// Input shape: [batch, features]. Output shape: same as input.
func (bn *BatchNorm) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	x := inputs[0]

	if !bn.training {
		// Eval: normalize using running statistics.
		mean := autograd.NewVariable(bn.RunningMean, false)
		std := autograd.NewVariable(bn.RunningVar, false).AddScalar(bn.eps).Sqrt()
		normalized := x.Sub(mean).Div(std)
		return normalized.Mul(bn.Weight.Variable).Add(bn.Bias.Variable)
	}

	// Training: normalize using batch statistics.
	batchMean := x.MeanDim(0, false) // [features]
	centered := x.Sub(batchMean)
	batchVar := centered.Mul(centered).MeanDim(0, false) // [features]

	normalized := centered.Div(batchVar.AddScalar(bn.eps).Sqrt())
	output := normalized.Mul(bn.Weight.Variable).Add(bn.Bias.Variable)

	// Update running stats (tensor-level, no autograd tracking).
	if err := output.Err(); err == nil {
		bn.updateRunningStats(batchMean.Data(), batchVar.Data(), x.Data().Shape()[0])
	}

	return output
}

// updateRunningStats applies exponential moving average to running statistics.
// Uses Bessel's correction for variance (unbiased estimate).
func (bn *BatchNorm) updateRunningStats(batchMean, batchVar *tensor.Tensor, batchSize int64) {
	m := bn.momentum

	// running_mean = (1 - momentum) * running_mean + momentum * batch_mean
	scaledRM := bn.RunningMean.MulScalar(1 - m)
	scaledBM := batchMean.MulScalar(m)
	newRM := scaledRM.Add(scaledBM)
	scaledRM.Release()
	scaledBM.Release()

	// Bessel's correction: unbiased_var = biased_var * N/(N-1)
	correction := 1.0
	if batchSize > 1 {
		n := float64(batchSize)
		correction = n / (n - 1)
	}
	correctedVar := batchVar.MulScalar(correction)

	// running_var = (1 - momentum) * running_var + momentum * corrected_var
	scaledRV := bn.RunningVar.MulScalar(1 - m)
	scaledBV := correctedVar.MulScalar(m)
	newRV := scaledRV.Add(scaledBV)
	scaledRV.Release()
	scaledBV.Release()
	correctedVar.Release()

	if newRM.Err() != nil || newRV.Err() != nil {
		newRM.Release()
		newRV.Release()
		return
	}

	bn.RunningMean.Release()
	bn.RunningMean = newRM
	bn.RunningVar.Release()
	bn.RunningVar = newRV
}

// Parameters returns weight (gamma) and bias (beta).
func (bn *BatchNorm) Parameters() []*Parameter {
	return []*Parameter{bn.Weight, bn.Bias}
}
