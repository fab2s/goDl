package nn

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// GRUCell implements a single step of a Gated Recurrent Unit.
//
//	gru, _ := nn.NewGRUCell(128, 256)
//	h := gru.Forward(x, hPrev)  // x: [batch, 128], h: [batch, 256]
//
// Computes:
//
//	r = sigmoid(Wr_x @ x + Wr_h @ h)         reset gate
//	z = sigmoid(Wz_x @ x + Wz_h @ h)         update gate
//	n = tanh(Wn_x @ x + r * (Wn_h @ h))      new gate
//	h' = (1 - z) * n + z * h                  new hidden state
//
// With forward references, h is nil on the first pass (initialized to zeros).
type GRUCell struct {
	// Input projections (one per gate)
	XR *Linear // reset gate
	XZ *Linear // update gate
	XN *Linear // new gate

	// Hidden projections (one per gate)
	HR *Linear // reset gate
	HZ *Linear // update gate
	HN *Linear // new gate

	InputSize  int64
	HiddenSize int64
}

// NewGRUCell creates a GRU cell with the given input and hidden sizes.
func NewGRUCell(inputSize, hiddenSize int64, opts ...tensor.Option) (*GRUCell, error) {
	xr, err := NewLinear(inputSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	xz, err := NewLinear(inputSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	xn, err := NewLinear(inputSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	hr, err := NewLinear(hiddenSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	hz, err := NewLinear(hiddenSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	hn, err := NewLinear(hiddenSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}

	return &GRUCell{
		XR: xr, XZ: xz, XN: xn,
		HR: hr, HZ: hz, HN: hn,
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
	}, nil
}

// Forward computes one GRU step.
// inputs[0]: x — current input [batch, inputSize]
// inputs[1]: h — previous hidden state [batch, hiddenSize] (nil on first call)
func (g *GRUCell) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	x := inputs[0]

	// Handle nil hidden state (first call via forward ref)
	var h *autograd.Variable
	if len(inputs) > 1 && inputs[1] != nil {
		h = inputs[1]
	} else {
		batch := x.Data().Shape()[0]
		ht, err := tensor.Zeros([]int64{batch, g.HiddenSize})
		if err != nil {
			return autograd.ErrVariable(err)
		}
		h = autograd.NewVariable(ht, false)
	}

	// r = sigmoid(Wr_x @ x + Wr_h @ h)
	r := g.XR.Forward(x).Add(g.HR.Forward(h)).Sigmoid()

	// z = sigmoid(Wz_x @ x + Wz_h @ h)
	z := g.XZ.Forward(x).Add(g.HZ.Forward(h)).Sigmoid()

	// n = tanh(Wn_x @ x + r * (Wn_h @ h))
	n := g.XN.Forward(x).Add(r.Mul(g.HN.Forward(h))).Tanh()

	// h' = (1 - z) * n + z * h
	oneMinusZ := z.MulScalar(-1).AddScalar(1)
	return oneMinusZ.Mul(n).Add(z.Mul(h))
}

// Parameters returns all gate weights and biases (12 parameters).
func (g *GRUCell) Parameters() []*Parameter {
	var params []*Parameter
	params = append(params, g.XR.Parameters()...)
	params = append(params, g.XZ.Parameters()...)
	params = append(params, g.XN.Parameters()...)
	params = append(params, g.HR.Parameters()...)
	params = append(params, g.HZ.Parameters()...)
	params = append(params, g.HN.Parameters()...)
	return params
}
