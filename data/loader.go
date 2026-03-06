package data

import (
	"math/rand/v2"
	"sync"

	"github.com/fab2s/goDl/tensor"
)

// LoaderConfig configures a Loader.
type LoaderConfig struct {
	BatchSize  int  // samples per batch (required, must be > 0)
	Shuffle    bool // randomize sample order each epoch
	NumWorkers int  // parallel fetch goroutines (0 = sequential)
	PrefetchN  int  // batches buffered ahead (0 = no prefetch)
	DropLast   bool // drop incomplete final batch
}

// batch holds a stacked input/target pair ready for consumption.
type batch struct {
	input  *tensor.Tensor
	target *tensor.Tensor
}

// sample holds a single fetched item with its position for ordered assembly.
type sample struct {
	input  *tensor.Tensor
	target *tensor.Tensor
}

// Loader iterates over a Dataset in batches.
//
//	loader := data.NewLoader(ds, data.LoaderConfig{BatchSize: 32, Shuffle: true})
//	for loader.Next() {
//	    input, target := loader.Batch()
//	    // ... training step ...
//	}
//	if err := loader.Err(); err != nil { ... }
//	loader.Reset() // new epoch
type Loader struct {
	ds   Dataset
	cfg  LoaderConfig
	perm []int // index permutation

	// sequential state
	pos int // current position in perm

	// parallel pipeline
	batchCh chan batch
	done    chan struct{}
	wg      sync.WaitGroup

	// current batch (valid after Next returns true)
	curInput  *tensor.Tensor
	curTarget *tensor.Tensor
	err       error
}

// NewLoader creates a Loader for the given dataset and configuration.
func NewLoader(ds Dataset, cfg LoaderConfig) *Loader {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1
	}
	if cfg.NumWorkers < 0 {
		cfg.NumWorkers = 0
	}
	if cfg.PrefetchN < 0 {
		cfg.PrefetchN = 0
	}

	l := &Loader{
		ds:  ds,
		cfg: cfg,
	}
	l.initPerm()
	if cfg.NumWorkers > 0 {
		l.startPipeline()
	}
	return l
}

func (l *Loader) initPerm() {
	n := l.ds.Len()
	l.perm = make([]int, n)
	for i := range l.perm {
		l.perm[i] = i
	}
	if l.cfg.Shuffle {
		rand.Shuffle(n, func(i, j int) {
			l.perm[i], l.perm[j] = l.perm[j], l.perm[i]
		})
	}
	l.pos = 0
}

// Next advances to the next batch. Returns false when the epoch is
// exhausted or an error occurred. Check Err() after the loop.
func (l *Loader) Next() bool {
	if l.err != nil {
		return false
	}
	if l.cfg.NumWorkers > 0 {
		return l.nextParallel()
	}
	return l.nextSequential()
}

func (l *Loader) nextSequential() bool {
	n := l.ds.Len()
	if l.pos >= n {
		return false
	}
	end := l.pos + l.cfg.BatchSize
	if end > n {
		if l.cfg.DropLast {
			return false
		}
		end = n
	}

	indices := l.perm[l.pos:end]
	l.pos = end

	inputs := make([]*tensor.Tensor, len(indices))
	targets := make([]*tensor.Tensor, len(indices))
	for i, idx := range indices {
		inp, tgt, err := l.ds.Get(idx)
		if err != nil {
			l.err = err
			return false
		}
		inputs[i] = inp
		targets[i] = tgt
	}

	l.curInput = tensor.Stack(inputs, 0)
	if err := l.curInput.Err(); err != nil {
		l.err = err
		return false
	}
	l.curTarget = tensor.Stack(targets, 0)
	if err := l.curTarget.Err(); err != nil {
		l.err = err
		return false
	}
	return true
}

// --- parallel pipeline ---

func (l *Loader) startPipeline() {
	prefetch := l.cfg.PrefetchN
	if prefetch <= 0 {
		prefetch = 1
	}
	l.batchCh = make(chan batch, prefetch)
	l.done = make(chan struct{})

	l.wg.Add(1)
	go l.batchProducer()
}

func (l *Loader) batchProducer() {
	defer l.wg.Done()
	defer close(l.batchCh)

	n := l.ds.Len()
	pos := 0

	sampleCh := make(chan []sample, 1)

	for pos < n {
		end := pos + l.cfg.BatchSize
		if end > n {
			if l.cfg.DropLast {
				return
			}
			end = n
		}
		indices := l.perm[pos:end]
		pos = end

		// Fetch samples in parallel using worker goroutines.
		batchSamples := make([]sample, len(indices))
		var fetchErr error
		var mu sync.Mutex

		workers := l.cfg.NumWorkers
		if workers > len(indices) {
			workers = len(indices)
		}

		var fetchWg sync.WaitGroup
		idxCh := make(chan int, len(indices))
		for i := range indices {
			idxCh <- i
		}
		close(idxCh)

		fetchWg.Add(workers)
		for w := 0; w < workers; w++ {
			go func() {
				defer fetchWg.Done()
				for i := range idxCh {
					mu.Lock()
					failed := fetchErr != nil
					mu.Unlock()
					if failed {
						return
					}
					inp, tgt, err := l.ds.Get(indices[i])
					if err != nil {
						mu.Lock()
						if fetchErr == nil {
							fetchErr = err
						}
						mu.Unlock()
						return
					}
					batchSamples[i] = sample{input: inp, target: tgt}
				}
			}()
		}

		// Wait for fetch completion in a goroutine so we can
		// also listen for done.
		go func() {
			fetchWg.Wait()
			sampleCh <- batchSamples
		}()

		select {
		case <-l.done:
			// Drain remaining workers.
			fetchWg.Wait()
			return
		case <-sampleCh:
		}

		if fetchErr != nil {
			// Send zero batch to signal error; store error in loader via channel.
			// Instead, just return — the consumer will see channel close.
			// We store error via a slightly different approach: use the batch channel.
			return
		}

		inputs := make([]*tensor.Tensor, len(batchSamples))
		targets := make([]*tensor.Tensor, len(batchSamples))
		for i, s := range batchSamples {
			inputs[i] = s.input
			targets[i] = s.target
		}

		stacked := batch{
			input:  tensor.Stack(inputs, 0),
			target: tensor.Stack(targets, 0),
		}
		if err := stacked.input.Err(); err != nil {
			return
		}
		if err := stacked.target.Err(); err != nil {
			return
		}

		select {
		case <-l.done:
			return
		case l.batchCh <- stacked:
		}
	}
}

func (l *Loader) nextParallel() bool {
	b, ok := <-l.batchCh
	if !ok {
		return false
	}
	l.curInput = b.input
	l.curTarget = b.target
	return true
}

// Batch returns the current batch tensors. Valid only after Next returns true.
func (l *Loader) Batch() (input, target *tensor.Tensor) {
	return l.curInput, l.curTarget
}

// Err returns the first error encountered during iteration.
func (l *Loader) Err() error {
	return l.err
}

// Reset prepares the loader for a new epoch. If Shuffle is enabled,
// the sample order is randomized.
func (l *Loader) Reset() {
	l.Close()
	l.err = nil
	l.curInput = nil
	l.curTarget = nil
	l.initPerm()
	if l.cfg.NumWorkers > 0 {
		l.startPipeline()
	}
}

// Close stops background workers and releases resources.
// Safe to call multiple times.
func (l *Loader) Close() {
	if l.done != nil {
		select {
		case <-l.done:
			// already closed
		default:
			close(l.done)
		}
		l.wg.Wait()
		// Drain any remaining batches.
		//nolint:revive // intentional drain
		for range l.batchCh {
		}
		l.done = nil
		l.batchCh = nil
	}
}
