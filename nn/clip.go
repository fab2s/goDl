package nn

import (
	"math"

	"github.com/fab2s/goDl/tensor"
)

// ClipGradNorm scales parameter gradients so that the total (L2) norm
// does not exceed maxNorm. Returns the original total norm before clipping.
//
// This is the standard gradient clipping technique used to prevent
// exploding gradients, especially important for recurrent structures
// (Loop.While, Loop.Until) and deep graphs.
//
// Call between Backward() and optimizer.Step():
//
//	loss.Backward()
//	nn.ClipGradNorm(model.Parameters(), 1.0)
//	optimizer.Step()
func ClipGradNorm(params []*Parameter, maxNorm float64) float64 {
	totalNormSq := 0.0
	withGrad := make([]*Parameter, 0, len(params))

	for _, p := range params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		withGrad = append(withGrad, p)
		sq := grad.Mul(grad).Sum()
		data, err := sq.Float32Data()
		if err != nil || len(data) == 0 {
			continue
		}
		totalNormSq += float64(data[0])
	}

	totalNorm := math.Sqrt(totalNormSq)
	if totalNorm <= maxNorm || len(withGrad) == 0 {
		return totalNorm
	}

	scale := maxNorm / totalNorm
	for _, p := range withGrad {
		p.SetGrad(p.Grad().MulScalar(scale))
	}

	return totalNorm
}

// ClipGradValue clamps each gradient element to [-maxVal, maxVal].
// Returns the maximum absolute gradient value before clipping.
//
//	loss.Backward()
//	nn.ClipGradValue(model.Parameters(), 0.5)
//	optimizer.Step()
func ClipGradValue(params []*Parameter, maxVal float64) float64 {
	maxAbs := 0.0

	for _, p := range params {
		grad := p.Grad()
		if grad == nil {
			continue
		}
		data, err := grad.Float32Data()
		if err != nil {
			continue
		}

		needsClip := false
		for _, v := range data {
			abs := math.Abs(float64(v))
			if abs > maxAbs {
				maxAbs = abs
			}
			if abs > maxVal {
				needsClip = true
			}
		}

		if needsClip {
			clamped := make([]float32, len(data))
			for i, v := range data {
				clamped[i] = float32(math.Max(-maxVal, math.Min(maxVal, float64(v))))
			}
			newGrad, err := tensor.FromFloat32(clamped, grad.Shape())
			if err != nil {
				continue
			}
			p.SetGrad(newGrad)
		}
	}

	return maxAbs
}
