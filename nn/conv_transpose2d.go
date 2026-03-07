package nn

import (
	"fmt"
	"math"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// ConvTranspose2d applies a 2D transposed convolution (deconvolution).
//
// Input shape:  [batch, inChannels, height, width]
// Output shape: [batch, outChannels, outH, outW]
//
//	deconv, _ := nn.NewConvTranspose2d(128, 64, 3)
//	deconv.Stride = [2]int64{2, 2}
//	deconv.Padding = [2]int64{1, 1}
//	deconv.OutputPadding = [2]int64{1, 1}
//	output := deconv.Forward(input)
type ConvTranspose2d struct {
	Weight        *Parameter
	Bias          *Parameter
	Stride        [2]int64
	Padding       [2]int64
	OutputPadding [2]int64
	Dilation      [2]int64
	Groups        int64
}

// NewConvTranspose2d creates a ConvTranspose2d with a square kernel.
// Weight shape: [inChannels, outChannels, kernelSize, kernelSize].
func NewConvTranspose2d(inChannels, outChannels, kernelSize int64, opts ...tensor.Option) (*ConvTranspose2d, error) {
	fanIn := inChannels * kernelSize * kernelSize
	wData, err := KaimingUniform(
		[]int64{inChannels, outChannels, kernelSize, kernelSize},
		fanIn, opts...,
	)
	if err != nil {
		return nil, err
	}

	bound := 1.0 / math.Sqrt(float64(fanIn))
	bRand, err := tensor.Rand([]int64{outChannels}, opts...)
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

	return &ConvTranspose2d{
		Weight:        NewParameter(wData, "weight"),
		Bias:          NewParameter(bData, "bias"),
		Stride:        [2]int64{1, 1},
		Padding:       [2]int64{0, 0},
		OutputPadding: [2]int64{0, 0},
		Dilation:      [2]int64{1, 1},
		Groups:        1,
	}, nil
}

// MustConvTranspose2d creates a ConvTranspose2d, panicking on error.
func MustConvTranspose2d(inChannels, outChannels, kernelSize int64, opts ...tensor.Option) *ConvTranspose2d {
	c, err := NewConvTranspose2d(inChannels, outChannels, kernelSize, opts...)
	if err != nil {
		panic(fmt.Sprintf("nn.MustConvTranspose2d(%d, %d, %d): %v", inChannels, outChannels, kernelSize, err))
	}
	return c
}

// Forward applies the transposed convolution.
func (c *ConvTranspose2d) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	x := inputs[0]
	var bias *autograd.Variable
	if c.Bias != nil {
		bias = c.Bias.Variable
	}
	return x.ConvTranspose2d(
		c.Weight.Variable, bias,
		c.Stride[:], c.Padding[:], c.OutputPadding[:], c.Dilation[:], c.Groups,
	)
}

// Parameters returns weight and bias (if present).
func (c *ConvTranspose2d) Parameters() []*Parameter {
	if c.Bias != nil {
		return []*Parameter{c.Weight, c.Bias}
	}
	return []*Parameter{c.Weight}
}
