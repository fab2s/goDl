//go:build !cuda

package autograd_test

import (
	"testing"

	"github.com/fab2s/goDl/tensor"
)

var testDevice = tensor.CPU

func skipIfDeviceUnavailable(t *testing.T) {
	t.Helper()
}
