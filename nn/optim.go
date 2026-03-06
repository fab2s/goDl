package nn

import (
	"github.com/fab2s/goDl/tensor"
)

// Optimizer updates parameters based on their gradients.
type Optimizer interface {
	Step()     // apply one update
	ZeroGrad() // reset all gradients
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
type Adam struct {
	params []*Parameter
	lr     float64
	beta1  float64
	beta2  float64
	eps    float64
	m      []*tensor.Tensor // first moment (mean)
	v      []*tensor.Tensor // second moment (variance)
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
//
// Since we don't have element-wise sqrt or division yet, we implement
// the update using available ops: mul_scalar, add, sub, mul.
// sqrt(v_hat) + eps ≈ (v_hat + eps²)^0.5 — we approximate using
// exp(0.5 * log(v_hat + eps)) for numerical correctness.
func (a *Adam) Step() {
	a.t++
	for i, p := range a.params {
		grad := p.Grad()
		if grad == nil {
			continue
		}

		// Initialize moments to zeros on first step
		if a.m[i] == nil {
			zeros, err := tensor.Zeros(grad.Shape(), tensor.WithDType(grad.DType()))
			if err != nil {
				continue
			}
			a.m[i] = zeros
			zeros2, err := tensor.Zeros(grad.Shape(), tensor.WithDType(grad.DType()))
			if err != nil {
				continue
			}
			a.v[i] = zeros2
		}

		// m = β1*m + (1-β1)*grad
		a.m[i] = a.m[i].MulScalar(a.beta1).Add(grad.MulScalar(1 - a.beta1))

		// v = β2*v + (1-β2)*grad²
		gradSq := grad.Mul(grad)
		a.v[i] = a.v[i].MulScalar(a.beta2).Add(gradSq.MulScalar(1 - a.beta2))

		// Bias correction
		bc1 := 1.0 / (1.0 - pow(a.beta1, a.t))
		bc2 := 1.0 / (1.0 - pow(a.beta2, a.t))
		mHat := a.m[i].MulScalar(bc1)
		vHat := a.v[i].MulScalar(bc2)

		// param -= lr * mHat / (sqrt(vHat) + eps)
		// sqrt(vHat) + eps = exp(0.5 * log(vHat + eps)) approximately
		// For safety: vHat is always >= 0, add eps before log
		sqrtV := vHat.AddScalar(a.eps).Log().MulScalar(0.5).Exp()

		// mHat / sqrtV — we don't have division, so use: mHat * (1/sqrtV)
		// 1/sqrtV = exp(-log(sqrtV)) = exp(-0.5*log(vHat+eps))
		invSqrtV := vHat.AddScalar(a.eps).Log().MulScalar(-0.5).Exp()
		update := mHat.Mul(invSqrtV).MulScalar(a.lr)

		_ = sqrtV // used only for clarity above

		newData := p.Data().Sub(update)
		p.SetData(newData)
	}
}

// ZeroGrad resets all parameter gradients.
func (a *Adam) ZeroGrad() {
	for _, p := range a.params {
		p.ZeroGrad()
	}
}

// pow computes base^exp for small integer exponents.
func pow(base float64, exp int) float64 {
	result := 1.0
	for range exp {
		result *= base
	}
	return result
}
