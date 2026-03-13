//go:build cuda

package tensor_test

import "github.com/fab2s/goDl/tensor"

var testDevice = tensor.CUDA

func deviceUnavailable() bool {
	return !tensor.CUDAAvailable()
}
