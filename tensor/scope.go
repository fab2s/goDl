package tensor

import "sync"

// Scope tracks tensors for bulk cleanup. All tensors created through a
// Scope are released when the Scope closes, except those explicitly
// kept via Keep().
//
// Scopes are useful in training loops where each iteration creates many
// intermediate tensors that should all be freed together:
//
//	for batch := range dataloader {
//	    scope := tensor.NewScope()
//	    x := scope.Track(loadBatch(batch))
//	    out := scope.Track(x.Matmul(w).Add(b).ReLU())
//	    // ... use out ...
//	    scope.Close() // frees x, out, and all intermediates
//	}
//
// Or with the functional style:
//
//	result, err := tensor.WithScope(func(s *tensor.Scope) *tensor.Tensor {
//	    x := s.Track(loadBatch(batch))
//	    return x.Matmul(w).Add(b).ReLU()
//	})
//	// result is kept alive; everything else is freed
type Scope struct {
	mu      sync.Mutex
	tensors []*Tensor
	closed  bool
}

// NewScope creates a new scope for tracking tensors.
func NewScope() *Scope {
	return &Scope{}
}

// Track registers a tensor with this scope. The tensor will be released
// when the scope closes. Returns the same tensor for chaining:
//
//	out := scope.Track(x.Matmul(w))
func (s *Scope) Track(t *Tensor) *Tensor {
	if t == nil || !t.valid() {
		return t
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return t
	}
	s.tensors = append(s.tensors, t)
	return t
}

// Close releases all tracked tensors. Safe to call multiple times.
func (s *Scope) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	for _, t := range s.tensors {
		t.Release()
	}
	s.tensors = nil
}

// WithScope runs fn with a new scope and returns the result.
// All tensors tracked by the scope are released after fn returns,
// EXCEPT the returned tensor (which is automatically untracked).
//
//	result, err := tensor.WithScope(func(s *tensor.Scope) *tensor.Tensor {
//	    a := s.Track(someOp())
//	    b := s.Track(someOtherOp(a))
//	    return b  // b survives, a is freed
//	})
func WithScope(fn func(s *Scope) *Tensor) (*Tensor, error) {
	s := NewScope()
	result := fn(s)

	// Untrack the result so it survives scope closure
	if result != nil && result.valid() {
		s.mu.Lock()
		for i, t := range s.tensors {
			if t == result {
				s.tensors[i] = s.tensors[len(s.tensors)-1]
				s.tensors = s.tensors[:len(s.tensors)-1]
				break
			}
		}
		s.mu.Unlock()
	}

	s.Close()

	if result == nil {
		return nil, nil
	}
	return result, result.Err()
}
