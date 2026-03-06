package data

import "github.com/fab2s/goDl/tensor"

// Dataset provides random access to individual samples.
type Dataset interface {
	// Len returns the total number of samples.
	Len() int
	// Get returns the input and target tensors for the given index.
	Get(index int) (input, target *tensor.Tensor, err error)
}

// TensorDataset wraps pre-loaded input and target tensors.
// Each sample is a slice along dimension 0.
type TensorDataset struct {
	inputs  *tensor.Tensor
	targets *tensor.Tensor
	len     int
}

// NewTensorDataset creates a dataset from batched tensors.
// Both tensors must have the same size along dimension 0.
func NewTensorDataset(inputs, targets *tensor.Tensor) *TensorDataset {
	return &TensorDataset{
		inputs:  inputs,
		targets: targets,
		len:     int(inputs.Shape()[0]),
	}
}

func (d *TensorDataset) Len() int { return d.len }

func (d *TensorDataset) Get(index int) (input, target *tensor.Tensor, err error) {
	input = d.inputs.Select(0, int64(index))
	if err = input.Err(); err != nil {
		return nil, nil, err
	}
	target = d.targets.Select(0, int64(index))
	if err = target.Err(); err != nil {
		return nil, nil, err
	}
	return input, target, nil
}
