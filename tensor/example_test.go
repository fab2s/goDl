package tensor_test

import (
	"fmt"
	"log"

	"github.com/fab2s/goDl/tensor"
)

func Example_creation() {
	// Create tensors with default options (float32, CPU)
	z, err := tensor.Zeros([]int64{2, 3})
	if err != nil {
		log.Fatal(err)
	}
	defer z.Release()
	fmt.Println(z)

	// Create with options
	o, err := tensor.Ones([]int64{4}, tensor.WithDType(tensor.Float64))
	if err != nil {
		log.Fatal(err)
	}
	defer o.Release()
	fmt.Println(o)

	// Create from Go slice
	x, err := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{2, 3})
	if err != nil {
		log.Fatal(err)
	}
	defer x.Release()
	fmt.Println(x)

	// Output:
	// Tensor(shape=[2 3], dtype=float32, device=cpu)
	// Tensor(shape=[4], dtype=float64, device=cpu)
	// Tensor(shape=[2 3], dtype=float32, device=cpu)
}

func Example_chaining() {
	// Operations chain naturally. Errors propagate — check once at the end.
	x, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{1, 3})
	defer x.Release()
	w, _ := tensor.FromFloat32([]float32{1, 0, 0, 1, 0, 0}, []int64{3, 2})
	defer w.Release()
	b, _ := tensor.FromFloat32([]float32{-5, 10}, []int64{1, 2})
	defer b.Release()

	// Linear layer: y = ReLU(x @ W + b)
	y := x.Matmul(w).Add(b).ReLU()
	if err := y.Err(); err != nil {
		log.Fatal(err)
	}
	defer y.Release()

	data, _ := y.Float32Data()
	fmt.Println(data)

	// Output:
	// [0 12]
}

func Example_errorPropagation() {
	a, _ := tensor.FromFloat32([]float32{1, 2, 3}, []int64{3})
	defer a.Release()

	// Release b to simulate a bad tensor
	b, _ := tensor.FromFloat32([]float32{4, 5, 6}, []int64{3})
	b.Release()

	// The error propagates through the entire chain
	result := a.Add(b).ReLU().Sigmoid()
	fmt.Println(result.Err())

	// Output:
	// tensor: use after release
}

func Example_scope() {
	// Scopes provide deterministic cleanup for groups of tensors.
	// All tracked tensors are freed when the scope closes.
	result, err := tensor.WithScope(func(s *tensor.Scope) *tensor.Tensor {
		a, _ := tensor.Zeros([]int64{3})
		s.Track(a)
		b, _ := tensor.Ones([]int64{3})
		s.Track(b)

		// The returned tensor survives; a and b are freed.
		return a.Add(b)
	})
	if err != nil {
		log.Fatal(err)
	}
	defer result.Release()

	data, _ := result.Float32Data()
	fmt.Println(data)

	// Output:
	// [1 1 1]
}

func Example_scopeManual() {
	// Scopes can also be managed manually for more control.
	scope := tensor.NewScope()

	a, _ := tensor.Zeros([]int64{100, 100})
	scope.Track(a)
	b, _ := tensor.Ones([]int64{100, 100})
	scope.Track(b)
	c := scope.Track(a.Add(b))

	fmt.Println(c)

	// Free all tracked tensors at once
	scope.Close()
	fmt.Println(c.Err())

	// Output:
	// Tensor(shape=[100 100], dtype=float32, device=cpu)
	// tensor: use after release
}

func Example_dataAccess() {
	x, _ := tensor.FromFloat32([]float32{1.5, 2.5, 3.5}, []int64{3})
	defer x.Release()

	// Read data back to Go
	data, err := x.Float32Data()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(data)

	// Metadata
	fmt.Println("shape:", x.Shape())
	fmt.Println("ndim:", x.Ndim())
	fmt.Println("numel:", x.Numel())
	fmt.Println("dtype:", x.DType())
	fmt.Println("device:", x.Device())

	// Output:
	// [1.5 2.5 3.5]
	// shape: [3]
	// ndim: 1
	// numel: 3
	// dtype: float32
	// device: cpu
}

func Example_activations() {
	x, _ := tensor.FromFloat32([]float32{-2, -1, 0, 1, 2}, []int64{5})
	defer x.Release()

	// ReLU: max(0, x)
	r := x.ReLU()
	defer r.Release()
	rdata, _ := r.Float32Data()
	fmt.Println("relu:", rdata)

	// Sigmoid: 1 / (1 + exp(-x)), output in (0, 1)
	s := x.Sigmoid()
	defer s.Release()
	sdata, _ := s.Float32Data()
	fmt.Printf("sigmoid(0): %.1f\n", sdata[2])

	// Tanh: output in (-1, 1)
	t := x.Tanh()
	defer t.Release()
	tdata, _ := t.Float32Data()
	fmt.Printf("tanh(0): %.1f\n", tdata[2])

	// Output:
	// relu: [0 0 0 1 2]
	// sigmoid(0): 0.5
	// tanh(0): 0.0
}
