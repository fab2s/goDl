package tensor

import "github.com/fab2s/goDl/internal/libtorch"

// Device represents where a tensor's data lives.
type Device int

const (
	CPU  Device = Device(libtorch.CPU)
	CUDA Device = Device(libtorch.CUDA)
)

func (d Device) String() string {
	switch d {
	case CPU:
		return "cpu"
	case CUDA:
		return "cuda"
	default:
		return "unknown"
	}
}

// toLibtorch converts to the internal libtorch Device.
func (d Device) toLibtorch() libtorch.Device {
	return libtorch.Device(d)
}

// CUDAAvailable returns true if CUDA is available on this system.
func CUDAAvailable() bool {
	return libtorch.CUDAAvailable()
}

// CUDADeviceCount returns the number of available CUDA devices.
func CUDADeviceCount() int {
	return libtorch.CUDADeviceCount()
}
