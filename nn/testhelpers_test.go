package nn_test

import (
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// moduleToDevice moves all parameters of a module to testDevice.
// No-op when testDevice is CPU (parameters are created on CPU by default).
// Also moves non-parameter state (e.g., BatchNorm running stats) via DeviceMover.
func moduleToDevice(m nn.Module) {
	if testDevice == tensor.CPU {
		return
	}
	for _, p := range m.Parameters() {
		moved := p.Data().ToDevice(testDevice)
		p.SetData(moved)
	}
	// Move non-parameter state (e.g., BatchNorm running stats).
	nn.WalkModules(m, make(map[nn.Module]bool), func(mod nn.Module) {
		if dm, ok := mod.(nn.DeviceMover); ok {
			dm.MoveToDevice(testDevice)
		}
	})
}
