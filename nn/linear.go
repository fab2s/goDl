package nn

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// Linear implements a fully connected layer: y = x @ W^T + b.
//
// Weight shape: [outFeatures, inFeatures]
// Bias shape: [outFeatures] (optional)
type Linear struct {
	Weight *Parameter
	Bias   *Parameter
	inF    int64
	outF   int64
}

// NewLinear creates a Linear layer with Kaiming initialization.
func NewLinear(inFeatures, outFeatures int64, opts ...tensor.Option) (*Linear, error) {
	wData, err := KaimingUniform([]int64{outFeatures, inFeatures}, inFeatures, opts...)
	if err != nil {
		return nil, err
	}

	bound := 1.0 / float64(inFeatures)
	bRand, err := tensor.Rand([]int64{outFeatures}, opts...)
	if err != nil {
		wData.Release()
		return nil, err
	}
	bData := bRand.MulScalar(2 * bound).AddScalar(-bound)
	if err := bData.Err(); err != nil {
		wData.Release()
		bRand.Release()
		return nil, err
	}
	bRand.Release()

	return &Linear{
		Weight: NewParameter(wData, "weight"),
		Bias:   NewParameter(bData, "bias"),
		inF:    inFeatures,
		outF:   outFeatures,
	}, nil
}

// Forward computes x @ W^T + b.
// Input shape: [batch, inFeatures] or [inFeatures]
// Output shape: [batch, outFeatures] or [outFeatures]
func (l *Linear) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	x := inputs[0]
	// W is [out, in], transpose to [in, out] for x @ W^T
	wT := l.Weight.Transpose(0, 1)
	out := x.Matmul(wT)
	if l.Bias != nil {
		out = out.Add(l.Bias.Variable)
	}
	return out
}

// Parameters returns weight and bias.
func (l *Linear) Parameters() []*Parameter {
	if l.Bias != nil {
		return []*Parameter{l.Weight, l.Bias}
	}
	return []*Parameter{l.Weight}
}
