package nn

import (
	"github.com/fab2s/goDl/tensor"
)

// Optimizer updates parameters based on their gradients.
type Optimizer interface {
	Step()     // apply one update
	ZeroGrad() // reset all gradients
}

// LRAdjustable is implemented by optimizers whose learning rate can be
// changed at runtime. All built-in optimizers implement this interface.
// LR schedulers use it to adjust the rate between steps.
type LRAdjustable interface {
	Optimizer
	LR() float64
	SetLR(lr float64)
}

// --- SGD ---

// SGD implements stochastic gradient descent with optional momentum.
//
//	optimizer := nn.NewSGD(model.Parameters(), 0.01, 0.9)
//	loss.Backward()
//	optimizer.Step()
//	optimizer.ZeroGrad()
type SGD struct {
	params   []*Parameter
	lr       float64
	momentum float64
	velocity []*tensor.Tensor // momentum buffers, nil until first step
}

// NewSGD creates an SGD optimizer.
// momentum=0 gives vanilla SGD. momentum=0.9 is typical.
func NewSGD(params []*Parameter, lr, momentum float64) *SGD {
	return &SGD{
		params:   params,
		lr:       lr,
		momentum: momentum,
		velocity: make([]*tensor.Tensor, len(params)),
	}
}

// Step applies one SGD update to all parameters.
// With momentum: v = momentum * v + grad; param -= lr * v
// Without momentum: param -= lr * grad
func (o *SGD) Step() {
	for i, p := range o.params {
		grad := p.Grad()
		if grad == nil {
			continue
		}

		if o.momentum != 0 {
			if o.velocity[i] == nil {
				// First step: initialize velocity from gradient
				o.velocity[i] = grad.MulScalar(1) // clone
			} else {
				// v = momentum * v + grad
				o.velocity[i] = o.velocity[i].MulScalar(o.momentum).Add(grad)
			}
			// param -= lr * v
			update := o.velocity[i].MulScalar(o.lr)
			newData := p.Data().Sub(update)
			p.SetData(newData)
		} else {
			// param -= lr * grad
			update := grad.MulScalar(o.lr)
			newData := p.Data().Sub(update)
			p.SetData(newData)
		}
	}
}

// LR returns the current learning rate.
func (o *SGD) LR() float64 { return o.lr }

// SetLR changes the learning rate.
func (o *SGD) SetLR(lr float64) { o.lr = lr }

// ZeroGrad resets all parameter gradients.
func (o *SGD) ZeroGrad() {
	for _, p := range o.params {
		p.ZeroGrad()
	}
}

// --- Adam ---

// Adam implements the Adam optimizer (Kingma & Ba, 2014).
//
//	optimizer := nn.NewAdam(model.Parameters(), 0.001)
//	loss.Backward()
//	optimizer.Step()
//	optimizer.ZeroGrad()
type Adam struct {
	params []*Parameter
	lr     float64
	beta1  float64
	beta2  float64
	eps    float64
	m      []*tensor.Tensor // first moment (mean of gradients)
	v      []*tensor.Tensor // second moment (mean of squared gradients)
	t      int              // step counter
}

// NewAdam creates an Adam optimizer with default betas (0.9, 0.999) and eps (1e-8).
func NewAdam(params []*Parameter, lr float64) *Adam {
	return &Adam{
		params: params,
		lr:     lr,
		beta1:  0.9,
		beta2:  0.999,
		eps:    1e-8,
		m:      make([]*tensor.Tensor, len(params)),
		v:      make([]*tensor.Tensor, len(params)),
	}
}

// Step applies one Adam update.
// m = β1*m + (1-β1)*grad
// v = β2*v + (1-β2)*grad²
// m_hat = m / (1-β1^t)
// v_hat = v / (1-β2^t)
// param -= lr * m_hat / (sqrt(v_hat) + eps)
func (a *Adam) Step() {
	a.t++
	for i, p := range a.params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		a.adamUpdate(i, p, grad, 0)
	}
}

// adamUpdate applies the Adam update for a single parameter.
// weightDecay > 0 applies decoupled weight decay (AdamW).
func (a *Adam) adamUpdate(i int, p *Parameter, grad *tensor.Tensor, weightDecay float64) {
	if a.m[i] == nil {
		a.m[i] = grad.ZerosLike()
		a.v[i] = grad.ZerosLike()
	}

	// m = β1*m + (1-β1)*grad
	a.m[i] = a.m[i].MulScalar(a.beta1).Add(grad.MulScalar(1 - a.beta1))

	// v = β2*v + (1-β2)*grad²
	a.v[i] = a.v[i].MulScalar(a.beta2).Add(grad.Mul(grad).MulScalar(1 - a.beta2))

	// Bias correction
	mHat := a.m[i].MulScalar(1.0 / (1.0 - pow(a.beta1, a.t)))
	vHat := a.v[i].MulScalar(1.0 / (1.0 - pow(a.beta2, a.t)))

	// param -= lr * mHat / (sqrt(vHat) + eps)
	update := mHat.Div(vHat.Sqrt().AddScalar(a.eps)).MulScalar(a.lr)
	newData := p.Data().Sub(update)

	// Decoupled weight decay: param -= lr * wd * param
	if weightDecay > 0 {
		newData = newData.Sub(p.Data().MulScalar(a.lr * weightDecay))
	}

	p.SetData(newData)
}

// LR returns the current learning rate.
func (a *Adam) LR() float64 { return a.lr }

// SetLR changes the learning rate.
func (a *Adam) SetLR(lr float64) { a.lr = lr }

// ZeroGrad resets all parameter gradients.
func (a *Adam) ZeroGrad() {
	for _, p := range a.params {
		p.ZeroGrad()
	}
}

// --- AdamW ---

// AdamW implements Adam with decoupled weight decay (Loshchilov & Hutter, 2017).
// Unlike L2 regularization, decoupled weight decay is applied directly to the
// parameters, not to the gradient. This distinction matters for adaptive
// optimizers and generally improves generalization.
//
//	optimizer := nn.NewAdamW(model.Parameters(), 0.001, 0.01)
type AdamW struct {
	adam        Adam
	weightDecay float64
}

// NewAdamW creates an AdamW optimizer. weightDecay is typically 0.01.
func NewAdamW(params []*Parameter, lr, weightDecay float64) *AdamW {
	return &AdamW{
		adam: Adam{
			params: params,
			lr:     lr,
			beta1:  0.9,
			beta2:  0.999,
			eps:    1e-8,
			m:      make([]*tensor.Tensor, len(params)),
			v:      make([]*tensor.Tensor, len(params)),
		},
		weightDecay: weightDecay,
	}
}

// Step applies one AdamW update.
func (w *AdamW) Step() {
	w.adam.t++
	for i, p := range w.adam.params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		w.adam.adamUpdate(i, p, grad, w.weightDecay)
	}
}

// LR returns the current learning rate.
func (w *AdamW) LR() float64 { return w.adam.lr }

// SetLR changes the learning rate.
func (w *AdamW) SetLR(lr float64) { w.adam.lr = lr }

// ZeroGrad resets all parameter gradients.
func (w *AdamW) ZeroGrad() {
	w.adam.ZeroGrad()
}

// pow computes base^exp for small integer exponents.
func pow(base float64, exp int) float64 {
	result := 1.0
	for range exp {
		result *= base
	}
	return result
}
