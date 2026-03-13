package autograd

import (
	"sync"
	"sync/atomic"
)

// Scope tracks intermediate Variables created by autograd ops for
// deterministic batch-level cleanup. When closed, it releases the
// underlying C++ tensors immediately — no runtime.GC() needed.
//
// Only op results (created via newVar) are tracked. User-created
// Variables (via NewVariable) are NOT tracked — their lifecycle is
// managed by the caller or the GC. This means leaf parameters,
// detached state, and batch input tensors are never freed by Close.
//
// Usage in a training loop:
//
//	for loader.Next() {
//	    scope := autograd.NewScope()
//	    // ... forward, backward, step, read metrics ...
//	    scope.Close() // frees all intermediate tensors instantly
//	}
type Scope struct {
	mu   sync.Mutex
	vars []*Variable
}

var activeScope atomic.Pointer[Scope]

// NewScope starts tracking intermediate Variables created by autograd
// ops. Only one scope at a time — not designed for nesting.
func NewScope() *Scope {
	s := &Scope{
		vars: make([]*Variable, 0, 512),
	}
	activeScope.Store(s)
	return s
}

// Close releases the underlying tensor of every intermediate Variable
// created by autograd ops since NewScope. After Close, the tracked
// Variables' Data() returns nil — callers must read any needed values
// (Item, Float32Data, etc.) BEFORE calling Close.
func (s *Scope) Close() {
	activeScope.CompareAndSwap(s, nil)
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, v := range s.vars {
		if v.data != nil {
			v.data.Release()
			v.data = nil
		}
	}
	s.vars = s.vars[:0]
}

// track registers a Variable with the active scope. Called from
// newVar() for op results. No-op if no scope is active.
func track(v *Variable) {
	if s := activeScope.Load(); s != nil {
		s.mu.Lock()
		s.vars = append(s.vars, v)
		s.mu.Unlock()
	}
}
