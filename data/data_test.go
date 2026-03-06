package data_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/fab2s/goDl/data"
	"github.com/fab2s/goDl/tensor"
)

// --- synthetic dataset for testing ---

// rangeDataset returns samples where input=[i] and target=[i*10].
type rangeDataset struct {
	n int
}

func (d *rangeDataset) Len() int { return d.n }

func (d *rangeDataset) Get(index int) (input, target *tensor.Tensor, err error) {
	inp, err := tensor.FromFloat32([]float32{float32(index)}, []int64{1})
	if err != nil {
		return nil, nil, err
	}
	tgt, err := tensor.FromFloat32([]float32{float32(index * 10)}, []int64{1})
	if err != nil {
		return nil, nil, err
	}
	return inp, tgt, nil
}

// --- tensor.Stack ---

func TestStack(t *testing.T) {
	a, _ := tensor.FromFloat32([]float32{1, 2}, []int64{2})
	defer a.Release()
	b, _ := tensor.FromFloat32([]float32{3, 4}, []int64{2})
	defer b.Release()
	c, _ := tensor.FromFloat32([]float32{5, 6}, []int64{2})
	defer c.Release()

	s := tensor.Stack([]*tensor.Tensor{a, b, c}, 0)
	if err := s.Err(); err != nil {
		t.Fatalf("Stack: %v", err)
	}
	defer s.Release()

	if got := s.Shape(); len(got) != 2 || got[0] != 3 || got[1] != 2 {
		t.Fatalf("shape = %v, want [3 2]", got)
	}
	d, _ := s.Float32Data()
	want := []float32{1, 2, 3, 4, 5, 6}
	for i := range want {
		if d[i] != want[i] {
			t.Errorf("data[%d] = %f, want %f", i, d[i], want[i])
		}
	}
}

func TestStackDim1(t *testing.T) {
	a, _ := tensor.FromFloat32([]float32{1, 2}, []int64{2})
	defer a.Release()
	b, _ := tensor.FromFloat32([]float32{3, 4}, []int64{2})
	defer b.Release()

	s := tensor.Stack([]*tensor.Tensor{a, b}, 1)
	if err := s.Err(); err != nil {
		t.Fatalf("Stack: %v", err)
	}
	defer s.Release()

	// [2] stacked along dim 1 → [2, 2]
	if got := s.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 2 {
		t.Fatalf("shape = %v, want [2 2]", got)
	}
	d, _ := s.Float32Data()
	// [[1,3],[2,4]]
	want := []float32{1, 3, 2, 4}
	for i := range want {
		if d[i] != want[i] {
			t.Errorf("data[%d] = %f, want %f", i, d[i], want[i])
		}
	}
}

func TestStackEmpty(t *testing.T) {
	s := tensor.Stack(nil, 0)
	if s.Err() == nil {
		t.Fatal("expected error from empty Stack")
	}
}

// --- TensorDataset ---

func TestTensorDataset(t *testing.T) {
	inputs, _ := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6}, []int64{3, 2})
	defer inputs.Release()
	targets, _ := tensor.FromFloat32([]float32{10, 20, 30}, []int64{3})
	defer targets.Release()

	ds := data.NewTensorDataset(inputs, targets)
	if ds.Len() != 3 {
		t.Fatalf("Len = %d, want 3", ds.Len())
	}

	inp, tgt, err := ds.Get(1)
	if err != nil {
		t.Fatalf("Get(1): %v", err)
	}
	defer inp.Release()
	defer tgt.Release()

	d, _ := inp.Float32Data()
	if d[0] != 3 || d[1] != 4 {
		t.Errorf("input = %v, want [3 4]", d)
	}
	td, _ := tgt.Float32Data()
	if td[0] != 20 {
		t.Errorf("target = %v, want [20]", td)
	}
}

// --- Loader sequential ---

func TestLoaderBasic(t *testing.T) {
	ds := &rangeDataset{n: 10}
	loader := data.NewLoader(ds, data.LoaderConfig{BatchSize: 3})
	defer loader.Close()

	var batches int
	var totalSamples int
	for loader.Next() {
		input, target := loader.Batch()
		shape := input.Shape()
		totalSamples += int(shape[0])
		batches++

		// Verify target = input * 10
		id, _ := input.Float32Data()
		td, _ := target.Float32Data()
		for i := range id {
			if math.Abs(float64(td[i]-id[i]*10)) > 0.01 {
				t.Errorf("batch %d sample %d: target=%f, want input*10=%f",
					batches, i, td[i], id[i]*10)
			}
		}
	}
	if err := loader.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}

	if batches != 4 { // 3+3+3+1
		t.Errorf("batches = %d, want 4", batches)
	}
	if totalSamples != 10 {
		t.Errorf("total samples = %d, want 10", totalSamples)
	}
}

func TestLoaderDropLast(t *testing.T) {
	ds := &rangeDataset{n: 10}
	loader := data.NewLoader(ds, data.LoaderConfig{BatchSize: 3, DropLast: true})
	defer loader.Close()

	var batches int
	for loader.Next() {
		batches++
	}
	if batches != 3 { // drops final batch of 1
		t.Errorf("batches = %d, want 3", batches)
	}
}

func TestLoaderShuffle(t *testing.T) {
	ds := &rangeDataset{n: 100}
	loader := data.NewLoader(ds, data.LoaderConfig{BatchSize: 100, Shuffle: true})
	defer loader.Close()

	if !loader.Next() {
		t.Fatal("expected a batch")
	}
	input, _ := loader.Batch()
	d, _ := input.Float32Data()

	// Check that at least some elements are not in sequential order.
	inOrder := 0
	for i, v := range d {
		if int(v) == i {
			inOrder++
		}
	}
	// With 100 elements, shuffled should have very few in-place.
	if inOrder > 50 {
		t.Errorf("shuffle: %d/100 elements in original position, expected much fewer", inOrder)
	}
}

func TestLoaderReset(t *testing.T) {
	ds := &rangeDataset{n: 5}
	loader := data.NewLoader(ds, data.LoaderConfig{BatchSize: 5})
	defer loader.Close()

	// First epoch
	if !loader.Next() {
		t.Fatal("expected batch in epoch 1")
	}
	if loader.Next() {
		t.Fatal("expected end of epoch 1")
	}

	// Reset for second epoch
	loader.Reset()
	if !loader.Next() {
		t.Fatal("expected batch in epoch 2")
	}
}

// --- Loader parallel ---

func TestLoaderParallel(t *testing.T) {
	ds := &rangeDataset{n: 20}
	loader := data.NewLoader(ds, data.LoaderConfig{
		BatchSize:  4,
		NumWorkers: 3,
		PrefetchN:  2,
	})
	defer loader.Close()

	var batches int
	var totalSamples int
	seen := make(map[int]bool)

	for loader.Next() {
		input, target := loader.Batch()
		bs := int(input.Shape()[0])
		totalSamples += bs
		batches++

		id, _ := input.Float32Data()
		td, _ := target.Float32Data()
		for i := range id {
			idx := int(id[i])
			if seen[idx] {
				t.Errorf("duplicate sample index %d", idx)
			}
			seen[idx] = true
			if math.Abs(float64(td[i]-float32(idx*10))) > 0.01 {
				t.Errorf("target mismatch for index %d: got %f", idx, td[i])
			}
		}
	}
	if err := loader.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if totalSamples != 20 {
		t.Errorf("total samples = %d, want 20", totalSamples)
	}
	if batches != 5 {
		t.Errorf("batches = %d, want 5", batches)
	}
}

func TestLoaderParallelShuffle(t *testing.T) {
	ds := &rangeDataset{n: 50}
	loader := data.NewLoader(ds, data.LoaderConfig{
		BatchSize:  50,
		Shuffle:    true,
		NumWorkers: 2,
		PrefetchN:  1,
	})
	defer loader.Close()

	if !loader.Next() {
		t.Fatal("expected a batch")
	}
	input, _ := loader.Batch()
	d, _ := input.Float32Data()

	seen := make(map[int]bool)
	for _, v := range d {
		idx := int(v)
		if seen[idx] {
			t.Errorf("duplicate index %d", idx)
		}
		seen[idx] = true
	}
	if len(seen) != 50 {
		t.Errorf("unique samples = %d, want 50", len(seen))
	}
}

func TestLoaderParallelReset(t *testing.T) {
	ds := &rangeDataset{n: 8}
	loader := data.NewLoader(ds, data.LoaderConfig{
		BatchSize:  4,
		NumWorkers: 2,
	})
	defer loader.Close()

	// Consume epoch 1
	count := 0
	for loader.Next() {
		count++
	}
	if count != 2 {
		t.Errorf("epoch 1: batches = %d, want 2", count)
	}

	// Reset and consume epoch 2
	loader.Reset()
	count = 0
	for loader.Next() {
		count++
	}
	if count != 2 {
		t.Errorf("epoch 2: batches = %d, want 2", count)
	}
}

// --- error propagation ---

type errorDataset struct {
	n      int
	failAt int
}

func (d *errorDataset) Len() int { return d.n }

func (d *errorDataset) Get(index int) (input, target *tensor.Tensor, err error) {
	if index == d.failAt {
		return nil, nil, fmt.Errorf("dataset: synthetic error at index %d", index)
	}
	inp, err := tensor.FromFloat32([]float32{float32(index)}, []int64{1})
	if err != nil {
		return nil, nil, err
	}
	tgt, err := tensor.FromFloat32([]float32{0}, []int64{1})
	if err != nil {
		return nil, nil, err
	}
	return inp, tgt, nil
}

func TestLoaderErrorPropagation(t *testing.T) {
	ds := &errorDataset{n: 10, failAt: 5}
	loader := data.NewLoader(ds, data.LoaderConfig{BatchSize: 3})
	defer loader.Close()

	var batches int
	for loader.Next() {
		batches++
	}
	if loader.Err() == nil {
		t.Fatal("expected error from failing dataset")
	}
	// Should have gotten at most 1 successful batch (indices 0-2),
	// then fail on batch containing index 5.
	if batches > 2 {
		t.Errorf("batches = %d, expected at most 2 before error", batches)
	}
}
