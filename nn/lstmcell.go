package nn

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// LSTMCell implements a single step of a Long Short-Term Memory unit.
//
//	lstm, _ := nn.NewLSTMCell(128, 256)
//	state := lstm.Forward(x, prevState)  // state = cat(h', c') along last dim
//
// The state is a concatenation of hidden and cell states along the last
// dimension: [batch, 2*hiddenSize]. This allows LSTMCell to work with
// the graph builder's forward reference mechanism (which passes a single
// Variable between calls).
//
// Computes:
//
//	i = sigmoid(Wi @ x + Whi @ h)   input gate
//	f = sigmoid(Wf @ x + Whf @ h)   forget gate
//	g = tanh(Wg @ x + Whg @ h)      cell gate
//	o = sigmoid(Wo @ x + Who @ h)   output gate
//	c' = f * c + i * g              new cell state
//	h' = o * tanh(c')               new hidden state
type LSTMCell struct {
	// Input projections (one per gate)
	XI *Linear // input gate
	XF *Linear // forget gate
	XG *Linear // cell gate
	XO *Linear // output gate

	// Hidden projections (one per gate)
	HI *Linear // input gate
	HF *Linear // forget gate
	HG *Linear // cell gate
	HO *Linear // output gate

	InputSize  int64
	HiddenSize int64
}

// NewLSTMCell creates an LSTM cell with the given input and hidden sizes.
func NewLSTMCell(inputSize, hiddenSize int64, opts ...tensor.Option) (*LSTMCell, error) {
	xi, err := NewLinear(inputSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	xf, err := NewLinear(inputSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	xg, err := NewLinear(inputSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	xo, err := NewLinear(inputSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	hi, err := NewLinear(hiddenSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	hf, err := NewLinear(hiddenSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	hg, err := NewLinear(hiddenSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}
	ho, err := NewLinear(hiddenSize, hiddenSize, opts...)
	if err != nil {
		return nil, err
	}

	return &LSTMCell{
		XI: xi, XF: xf, XG: xg, XO: xo,
		HI: hi, HF: hf, HG: hg, HO: ho,
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
	}, nil
}

// Forward computes one LSTM step.
// inputs[0]: x — current input [batch, inputSize]
// inputs[1]: state — cat(h, c) [batch, 2*hiddenSize] (nil on first call)
// Returns: cat(h', c') [batch, 2*hiddenSize]
func (l *LSTMCell) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	x := inputs[0]
	batch := x.Data().Shape()[0]
	hs := l.HiddenSize

	// Split state into h and c (or zero-init on first call)
	var h, c *autograd.Variable
	if len(inputs) > 1 && inputs[1] != nil {
		state := inputs[1]
		h = state.Narrow(1, 0, hs)
		c = state.Narrow(1, hs, hs)
	} else {
		ht, err := tensor.Zeros([]int64{batch, hs})
		if err != nil {
			return autograd.ErrVariable(err)
		}
		ct, err := tensor.Zeros([]int64{batch, hs})
		if err != nil {
			return autograd.ErrVariable(err)
		}
		h = autograd.NewVariable(ht, false)
		c = autograd.NewVariable(ct, false)
	}

	// Gates
	i := l.XI.Forward(x).Add(l.HI.Forward(h)).Sigmoid()
	f := l.XF.Forward(x).Add(l.HF.Forward(h)).Sigmoid()
	g := l.XG.Forward(x).Add(l.HG.Forward(h)).Tanh()
	o := l.XO.Forward(x).Add(l.HO.Forward(h)).Sigmoid()

	// New cell and hidden states
	cNew := f.Mul(c).Add(i.Mul(g))
	hNew := o.Mul(cNew.Tanh())

	// Pack h' and c' into a single tensor
	return hNew.Cat(cNew, 1)
}

// Parameters returns all gate weights and biases (16 parameters).
func (l *LSTMCell) Parameters() []*Parameter {
	var params []*Parameter
	params = append(params, l.XI.Parameters()...)
	params = append(params, l.XF.Parameters()...)
	params = append(params, l.XG.Parameters()...)
	params = append(params, l.XO.Parameters()...)
	params = append(params, l.HI.Parameters()...)
	params = append(params, l.HF.Parameters()...)
	params = append(params, l.HG.Parameters()...)
	params = append(params, l.HO.Parameters()...)
	return params
}
