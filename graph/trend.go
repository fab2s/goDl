package graph

import "math"

// Trend provides statistical queries over a time series of scalar values.
//
// Typically obtained via g.Trend(tag), which returns the epoch-level
// history for a tag (one value per Flush call). Can also be created
// from any []float64 with NewTrend.
//
// All query methods accept a window parameter: positive values limit
// the analysis to the last N data points; zero or negative uses all.
// Methods return safe zero-values for empty or insufficient data.
//
// Common patterns:
//
//	trend := g.Trend("loss")
//	trend.Improving(5)          // is loss decreasing over last 5 epochs?
//	trend.Stalled(10, 1e-4)     // has loss stopped moving?
//	trend.Converged(5, 1e-5)    // has loss stabilized?
//	trend.Slope(0)              // linear trend over all epochs
type Trend struct {
	values []float64
}

// NewTrend creates a Trend from a slice of values.
// The slice is not copied — the Trend is a view over the data.
func NewTrend(values []float64) *Trend {
	return &Trend{values: values}
}

// Len returns the number of data points.
func (t *Trend) Len() int { return len(t.values) }

// Values returns the underlying data.
func (t *Trend) Values() []float64 { return t.values }

// Last returns the last n values. If n > Len(), returns all values.
func (t *Trend) Last(n int) []float64 {
	if n >= len(t.values) {
		return t.values
	}
	return t.values[len(t.values)-n:]
}

// Mean returns the arithmetic mean. Returns 0 for empty series.
func (t *Trend) Mean() float64 {
	if len(t.values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range t.values {
		sum += v
	}
	return sum / float64(len(t.values))
}

// Min returns the minimum value. Returns 0 for empty series.
func (t *Trend) Min() float64 {
	if len(t.values) == 0 {
		return 0
	}
	m := t.values[0]
	for _, v := range t.values[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

// Max returns the maximum value. Returns 0 for empty series.
func (t *Trend) Max() float64 {
	if len(t.values) == 0 {
		return 0
	}
	m := t.values[0]
	for _, v := range t.values[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

// Slope returns the OLS linear regression slope over the last window
// values. A negative slope means the values are decreasing.
// Returns 0 if fewer than 2 values are available.
// If window <= 0, uses all values.
func (t *Trend) Slope(window int) float64 {
	vals := t.tail(window)
	n := len(vals)
	if n < 2 {
		return 0
	}

	// OLS: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
	var sx, sy, sxy, sx2 float64
	for i, y := range vals {
		x := float64(i)
		sx += x
		sy += y
		sxy += x * y
		sx2 += x * x
	}
	fn := float64(n)
	denom := fn*sx2 - sx*sx
	if denom == 0 {
		return 0
	}
	return (fn*sxy - sx*sy) / denom
}

// Stalled returns true if the absolute slope over the last window
// values is below tolerance — the metric isn't moving meaningfully.
// Returns false if fewer than 2 values are available.
func (t *Trend) Stalled(window int, tol float64) bool {
	if len(t.tail(window)) < 2 {
		return false
	}
	return math.Abs(t.Slope(window)) < tol
}

// Improving returns true if the slope over the last window values
// is negative — the metric is decreasing (good for loss).
// Returns false if fewer than 2 values are available.
func (t *Trend) Improving(window int) bool {
	if len(t.tail(window)) < 2 {
		return false
	}
	return t.Slope(window) < 0
}

// Converged returns true if the variance over the last window values
// is below tolerance — the metric has stabilized.
// Returns false if fewer than 2 values are available.
func (t *Trend) Converged(window int, tol float64) bool {
	vals := t.tail(window)
	if len(vals) < 2 {
		return false
	}
	mean := 0.0
	for _, v := range vals {
		mean += v
	}
	mean /= float64(len(vals))

	variance := 0.0
	for _, v := range vals {
		d := v - mean
		variance += d * d
	}
	variance /= float64(len(vals))
	return variance < tol
}

// tail returns the last n values, or all if n <= 0.
func (t *Trend) tail(n int) []float64 {
	if n <= 0 || n >= len(t.values) {
		return t.values
	}
	return t.values[len(t.values)-n:]
}
