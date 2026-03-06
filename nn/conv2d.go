package nn

import (
	"math"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// Conv2d applies a 2D convolution over an input signal.
//
// Input shape:  [batch, inChannels, height, width]
// Output shape: [batch, outChannels, outH, outW]
//
// Weight shape: [outChannels, inChannels, kernelH, kernelW]
// Bias shape:   [outChannels] (nil if NoBias)
//
//	conv, _ := nn.NewConv2d(3, 64, 3)                  // 3x3 kernel
//	conv.Padding = [2]int64{1, 1}                       // same-padding
//	output := conv.Forward(input)                       // [N, 64, H, W]
type Conv2d struct {
	Weight   *Parameter
	Bias     *Parameter
	Stride   [2]int64
	Padding  [2]int64
	Dilation [2]int64
	Groups   int64
}

// NewConv2d creates a Conv2d with a square kernel. Default: stride=1, padding=0,
// dilation=1, groups=1, with bias. Weight is Kaiming-initialized.
func NewConv2d(inChannels, outChannels, kernelSize int64, opts ...tensor.Option) (*Conv2d, error) {
	fanIn := inChannels * kernelSize * kernelSize
	wData, err := KaimingUniform(
		[]int64{outChannels, inChannels, kernelSize, kernelSize},
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

	return &Conv2d{
		Weight:   NewParameter(wData, "weight"),
		Bias:     NewParameter(bData, "bias"),
		Stride:   [2]int64{1, 1},
		Padding:  [2]int64{0, 0},
		Dilation: [2]int64{1, 1},
		Groups:   1,
	}, nil
}

// Forward applies the convolution.
// Input shape: [batch, inChannels, height, width].
func (c *Conv2d) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	x := inputs[0]
	var bias *autograd.Variable
	if c.Bias != nil {
		bias = c.Bias.Variable
	}
	return x.Conv2d(
		c.Weight.Variable, bias,
		c.Stride[:], c.Padding[:], c.Dilation[:], c.Groups,
	)
}

// Parameters returns weight and bias (if present).
func (c *Conv2d) Parameters() []*Parameter {
	if c.Bias != nil {
		return []*Parameter{c.Weight, c.Bias}
	}
	return []*Parameter{c.Weight}
}
