// Package tensor provides a safe, idiomatic Go tensor type built on libtorch.
//
// Tensors are immutable values — every operation returns a new tensor.
// Memory is managed via GC finalizers (safety net) and optional Scopes
// (deterministic bulk cleanup for training loops).
//
// Operations are chainable methods that carry errors:
//
//	result := x.Matmul(w).Add(b).ReLU()
//	if err := result.Err(); err != nil {
//	    log.Fatal(err)
//	}
package tensor

import "github.com/fab2s/goDl/internal/libtorch"

// DType represents a tensor's element type.
type DType int

const (
	Float16  DType = DType(libtorch.Float16)
	BFloat16 DType = DType(libtorch.BFloat16)
	Float32  DType = DType(libtorch.Float32)
	Float64  DType = DType(libtorch.Float64)
	Int32    DType = DType(libtorch.Int32)
	Int64    DType = DType(libtorch.Int64)
)

func (d DType) String() string {
	switch d {
	case Float16:
		return "float16"
	case BFloat16:
		return "bfloat16"
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	default:
		return "unknown"
	}
}

// ElementSize returns the size in bytes of one element of this dtype.
func (d DType) ElementSize() int64 {
	switch d {
	case Float16, BFloat16:
		return 2
	case Float32, Int32:
		return 4
	case Float64, Int64:
		return 8
	default:
		return 0
	}
}

// toLibtorch converts to the internal libtorch DType.
func (d DType) toLibtorch() libtorch.DType {
	return libtorch.DType(d)
}
