package nn

import (
	"fmt"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// LayerNorm applies layer normalization over the last dimension.
//
//	ln, _ := nn.NewLayerNorm(512)
//	output := ln.Forward(input)  // normalizes the last dim
//
// Computes: output = gamma * (x - mean) / sqrt(var + eps) + beta
// where mean and var are computed over the last dimension.
type LayerNorm struct {
	Weight *Parameter // gamma [size]
	Bias   *Parameter // beta [size]
	size   int64
	eps    float64
}

// NewLayerNorm creates a LayerNorm module for the given feature size.
// Weight (gamma) is initialized to ones, bias (beta) to zeros.
func NewLayerNorm(size int64, opts ...tensor.Option) (*LayerNorm, error) {
	wData, err := tensor.Ones([]int64{size}, opts...)
	if err != nil {
		return nil, err
	}
	bData, err := tensor.Zeros([]int64{size}, opts...)
	if err != nil {
		wData.Release()
		return nil, err
	}

	return &LayerNorm{
		Weight: NewParameter(wData, "weight"),
		Bias:   NewParameter(bData, "bias"),
		size:   size,
		eps:    1e-5,
	}, nil
}

// MustLayerNorm creates a LayerNorm module, panicking on error.
// Use in graph construction where dimensions are known constants.
func MustLayerNorm(size int64, opts ...tensor.Option) *LayerNorm {
	ln, err := NewLayerNorm(size, opts...)
	if err != nil {
		panic(fmt.Sprintf("nn.MustLayerNorm(%d): %v", size, err))
	}
	return ln
}

// Forward normalizes the input over the last dimension.
// Input shape: [*, size] where * is any number of leading dimensions.
// Output shape: same as input.
func (ln *LayerNorm) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	x := inputs[0]
	ndim := x.Data().Ndim()
	dim := ndim - 1

	// mean and variance over last dimension
	mean := x.MeanDim(dim, true)
	centered := x.Sub(mean)
	variance := centered.Mul(centered).MeanDim(dim, true)

	// normalize: (x - mean) / sqrt(var + eps)
	normalized := centered.Div(variance.AddScalar(ln.eps).Sqrt())

	// scale and shift: gamma * normalized + beta
	return normalized.Mul(ln.Weight.Variable).Add(ln.Bias.Variable)
}

// Parameters returns weight (gamma) and bias (beta).
func (ln *LayerNorm) Parameters() []*Parameter {
	return []*Parameter{ln.Weight, ln.Bias}
}
