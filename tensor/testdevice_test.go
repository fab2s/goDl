//go:build !cuda

package tensor_test

import "github.com/fab2s/goDl/tensor"

var testDevice = tensor.CPU

func deviceUnavailable() bool {
	return false // CPU is always available.
}
