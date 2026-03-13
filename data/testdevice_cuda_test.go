//go:build cuda

package data_test

import (
	"testing"

	"github.com/fab2s/goDl/tensor"
)

var testDevice = tensor.CUDA

func skipIfDeviceUnavailable(t *testing.T) {
	t.Helper()
	if !tensor.CUDAAvailable() {
		t.Skip("CUDA not available")
	}
}
