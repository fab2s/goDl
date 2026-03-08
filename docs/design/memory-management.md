# Memory Management: The CGo/VRAM Challenge

goDl wraps libtorch tensors via CGo. A Go `*tensor.Tensor` struct is
~24 bytes, but the underlying C++ tensor can hold megabytes of GPU
memory. Go's garbage collector sees the 24-byte struct; it does not
see the VRAM. This mismatch is the central challenge for GPU training.

---

## The problem

During a training step, memory flows through three phases:

```
Forward:   input → [op → intermediate tensor → op → ...] → output
                    each op also saves tensors for backward ──┐
                                                              │
Backward:  loss.Backward() walks the DAG in reverse       ◄──┘
           uses saved tensors to compute gradients
           after processing a node, its saved tensors are dead

Cleanup:   optimizer.Step() creates new parameter tensors
           old parameter data + all intermediates are dead
```

**What stays alive and shouldn't:**

After `Backward()` returns, every intermediate Variable in the
computation graph is still reachable. The user holds `loss`, which
holds a `gradFn` chain linking back through every operation to the
leaf parameters. Each `gradFn` closure captures the `*tensor.Tensor`
pointers it needed for backward — but backward is already finished.

The entire graph — every intermediate forward result, every
saved-for-backward tensor — stays in memory until Go's GC collects
the chain starting from `loss`. And Go's GC has no urgency: it sees
24 bytes becoming unreachable, not 200MB of VRAM.

**The numbers:**

For a model with N ops in the forward pass:
- N intermediate result tensors (forward outputs)
- ~N saved tensor references in gradFn closures
- N gradient tensors created during backward

Peak memory is during backward (all three categories alive). After
backward, categories 1 and 2 are dead but not freed. Category 3
(gradients) is needed by the optimizer and correctly freed by
`ZeroGrad()`.

On an 8GB GPU, this means training crashes after a few hundred steps
because "dead" VRAM accumulates faster than the GC reclaims it.

---

## Current state

**Tensor lifecycle tools that exist:**

| Mechanism | How it works | Limitation |
|-----------|-------------|------------|
| `runtime.SetFinalizer` | GC eventually calls `Release()` | GC doesn't see VRAM pressure — too slow |
| `tensor.Release()` | Explicit immediate free | User must know which tensors to free |
| `tensor.Scope` / `WithScope` | Bulk cleanup of tracked tensors | Not integrated with autograd |
| `runtime.GC()` in training loop | Force collection after backward | Blunt, ~1-5ms per call, fixes symptom not cause |

**What autograd does today (`engine.go`):**

```go
for i := len(order) - 1; i >= 0; i-- {
    node := order[i]
    inputGrads := node.gradFn.apply(grad)       // use saved tensors
    for j, input := range node.gradFn.inputs {   // distribute gradients
        grads[input] = inputGrads[j]
    }
    // node.gradFn still holds closure + inputs → everything stays alive
}
```

After this loop, the entire backward graph is still intact. Nothing
is released until the GC collects `loss` and follows the chain.

---

## Solution: phased approach

### Phase 1: Release the backward graph during backward (fix)

**Status: implemented**

After backward processes a node, its `gradFn` is never used again.
Nil it out immediately:

```go
for i := len(order) - 1; i >= 0; i-- {
    node := order[i]
    // ... existing backward logic ...

    // Release backward graph as we walk it.
    if !node.isLeaf {
        node.gradFn = nil  // drops closure + all captured tensors
    }
}
```

We nil `gradFn` but **not** `data` — the user may read intermediate
results after backward (e.g., `loss.Item()`). The `data` tensors
still become collectible sooner because the `gradFn` chain linking
all nodes together is now broken: each non-leaf Variable becomes an
isolated struct rather than part of a deep reference chain the GC
must trace from root to leaves.

**Why this is safe:**
- Non-leaf `gradFn` is only used once during backward — we just used it
- Leaf variables keep their `data` (parameters) and `grad` (for optimizer)
- The topological order ensures all consumers are processed before producers

**What it achieves:**
- Saved-for-backward tensors (captured in closures) become GC-eligible
  during backward, not after
- The gradFn chain is broken — GC can collect nodes independently
  instead of tracing the entire graph from the loss variable
- Gradient map entries for intermediates are deleted as we go
- Zero risk — the graph is walked once and discarded

**What it doesn't solve:**
- GC still needs to run to actually free the C++ memory
- For CPU training, this is adequate — Go's GC sees process RSS grow
- For GPU training, the GC is still blind to VRAM

### Phase 2: Native memory pressure tracking

Give Go's GC visibility into C++ allocations so it runs when VRAM is
filling up, not just when Go heap grows.

**Option A: `runtime.SetMemoryLimit` + accounting**

Track total C++ tensor bytes in an atomic counter. Increment on
allocation, decrement on free. When the counter exceeds a threshold,
hint the GC via `runtime.GC()` or adjust `GOGC` dynamically.

```go
var nativeBytes atomic.Int64

func wrap(raw *libtorch.Tensor) *Tensor {
    size := raw.Nbytes()
    nativeBytes.Add(int64(size))
    t := &Tensor{raw: raw}
    runtime.SetFinalizer(t, func(t *Tensor) {
        nativeBytes.Add(-int64(size))
        t.release()
    })
    return t
}
```

Pro: no API changes, works with existing finalizer flow.
Con: still depends on finalizer timing for actual release.

**Option B: Go-side reference counting**

Add an atomic refcount to `Tensor`. The autograd engine calls
`Retain()` when saving for backward, `Release()` when done. At
count 0, the C++ tensor is freed immediately — no GC involvement.

```go
type Tensor struct {
    raw  *libtorch.Tensor
    refs int64  // atomic, starts at 1
}

func (t *Tensor) Retain()  { atomic.AddInt64(&t.refs, 1) }
func (t *Tensor) Release() {
    if atomic.AddInt64(&t.refs, -1) == 0 {
        t.raw.Free()
    }
}
```

Pro: deterministic, immediate VRAM release, ~1-2ns per refcount op.
Con: every op must correctly Retain/Release — error-prone. Requires
refactoring gradFn to store saved tensors explicitly (not in closures)
so they can be Released individually after backward uses them.

The refcounting is needed because the same tensor can be saved by
multiple gradFns (e.g., a weight used in two Matmul operations). You
cannot Release a saved tensor after one gradFn without knowing whether
another gradFn still references it.

### Phase 3: Deterministic lifecycle in autograd

The autograd DAG has well-defined ownership. Rather than general
refcounting, exploit the structure:

1. **Forward:** each op declares which tensors it needs for backward
   by storing them in a `saved []*tensor.Tensor` field on `gradFn`
   (not captured in closures).

2. **Backward:** after `gradFn.apply()` runs, call `Release()` on
   each saved tensor. The refcount (from Phase 2) ensures shared
   tensors survive until all consumers are done.

3. **End of backward:** all saved tensors have been released. The
   only surviving tensors are leaf parameters and their gradients.

```go
type gradFn struct {
    name   string
    inputs []*Variable
    saved  []*tensor.Tensor
    apply  func(saved []*tensor.Tensor, grad *tensor.Tensor) []*tensor.Tensor
}
```

This is the design PyTorch uses internally — `SavedVariable` with
reference counting via `c10::intrusive_ptr`. The difference is that
PyTorch gets refcounting "for free" from CPython's reference counting
GC. In Go, we must add it explicitly.

**What this achieves:**
- Deterministic VRAM release during backward — each tensor is freed
  the instant its last consumer finishes
- Peak memory = forward intermediates + gradients being computed
  (saved tensors are freed as backward progresses)
- No dependence on Go's GC for VRAM lifecycle
- Go finalizers remain as a safety net for leaked tensors

---

## What NOT to do

**Expose libtorch's C++ refcounting through CGo (Level 2):**
Every `retain()`/`release()` is a CGo call (~100ns). With 36 ops and
multiple saved tensors each, this adds measurable overhead to every
forward+backward pass. Go-side atomic ops are ~1-2ns — 50-100x cheaper.

**Add `runtime.GC()` calls throughout the engine:**
This papers over the problem. A single GC call after backward is
acceptable as a temporary measure; scattering them throughout the
engine is not. The goal is deterministic lifecycle, not GC pressure.

**Manual Release() in user training loops:**
The autograd engine knows exactly when tensors become dead. Pushing
this responsibility to users is a design failure.

---

## Implementation priority

| Phase | Effort | Impact | Risk |
|-------|--------|--------|------|
| Phase 1: nil backward graph | ~10 lines in engine.go | High — halves post-backward memory | Zero — graph is write-once |
| Phase 2: refcounting | ~50 lines in tensor + autograd | Medium — enables Phase 3 | Low — additive change |
| Phase 3: deterministic release | ~200 lines across ops.go | Very high — PyTorch-level memory | Medium — touches all 36 ops |

Phase 1 is a fix that should ship immediately. Phases 2+3 are
engineering work that can be done incrementally — start with the
highest-memory ops (Matmul, Conv2d) and expand coverage over time.

---

## Comparison with PyTorch

PyTorch benefits from CPython's reference counting: every Python object
(including tensors) has an immediate refcount decrement when it goes
out of scope. No GC needed for deterministic cleanup. PyTorch's
`SavedVariable` adds a second layer of refcounting for backward-saved
tensors, releasing them as backward progresses.

goDl uses Go's tracing GC, which is fundamentally non-deterministic
for C memory. Phase 1 (nil-out) + Phase 2+3 (Go-side refcounting +
deterministic release) brings goDl to parity with PyTorch's memory
behavior — deterministic VRAM lifecycle independent of GC timing.
