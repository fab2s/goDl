package tensor

// Option configures tensor creation. Use WithDType and WithDevice.
type Option func(*options)

type options struct {
	dtype  DType
	device Device
}

// defaults
func applyOptions(opts []Option) options {
	o := options{
		dtype:  Float32,
		device: CPU,
	}
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// WithDType sets the element type for tensor creation.
func WithDType(dtype DType) Option {
	return func(o *options) {
		o.dtype = dtype
	}
}

// WithDevice sets the device for tensor creation.
func WithDevice(device Device) Option {
	return func(o *options) {
		o.device = device
	}
}
