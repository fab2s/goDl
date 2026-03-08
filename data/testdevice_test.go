//go:build !cuda

package data_test

import (
	"testing"

	"github.com/fab2s/goDl/tensor"
)

var testDevice = tensor.CPU

func skipIfDeviceUnavailable(t *testing.T) {
	t.Helper()
}
