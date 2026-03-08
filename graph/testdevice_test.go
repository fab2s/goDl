//go:build !cuda

package graph

import (
	"testing"

	"github.com/fab2s/goDl/tensor"
)

var testDevice = tensor.CPU

func skipIfDeviceUnavailable(t *testing.T) {
	t.Helper()
	// CPU is always available.
}
