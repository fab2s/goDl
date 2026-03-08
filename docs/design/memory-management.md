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

### Phase 2+3: Deterministic lifecycle via refcounting

**Status: implemented**

Go-side atomic reference counting on `Tensor`, combined with explicit
saved-tensor tracking in `gradFn`. The backward engine releases saved
tensors immediately after computing gradients — C++ memory (including
VRAM) is freed deterministically without waiting for Go's GC.

**Tensor refcounting (`tensor/tensor.go`):**

```go
type Tensor struct {
    raw  *libtorch.Tensor
    refs int64  // atomic; 1 from wrap(), 0 for error tensors
    err  error
}

func (t *Tensor) Retain()  { atomic.AddInt64(&t.refs, 1) }
func (t *Tensor) release() {
    if atomic.AddInt64(&t.refs, -1) == 0 {
        t.raw.Free(); t.raw = nil
        runtime.SetFinalizer(t, nil)
    }
}
```

- `wrap()` initializes refs to 1 and sets a GC finalizer as safety net
- `Retain()` increments the refcount (~1-2ns atomic op)
- `Release()` decrements; at zero, frees C++ memory immediately
- GC finalizer calls `release()` — safety net for leaked tensors

**Saved tensors in autograd (`autograd/variable.go`, `autograd/ops.go`):**

```go
type gradFn struct {
    name   string
    inputs []*Variable
    saved  []*tensor.Tensor  // Retained during forward, Released after backward
    apply  func(gradOutput *tensor.Tensor) []*tensor.Tensor
}
```

During forward, each op calls `saveForBackward()` which Retains each
tensor that the backward closure needs. The closure still captures the
same tensor pointers — the saved field provides a parallel lifecycle
tracker. All 23 ops that save tensors for backward are covered:

| Group | Ops | Saves |
|-------|-----|-------|
| Save input | ReLU, Sum, SumDim, MeanDim, Select, Narrow, IndexSelect, Abs, Pow, Clamp | input data |
| Save output | Sigmoid, Tanh, Exp, Log, Sqrt, Softmax | forward result |
| Save both inputs | Mul, Div, Matmul | both operands |
| Save input+params | Conv2d, ConvTranspose2d, AdaptiveAvgPool2d, GridSample | input + weight/grid |

The remaining 12 ops (Add, Sub, Neg, AddScalar, MulScalar, Reshape,
Transpose, Squeeze, Unsqueeze, Flatten, Permute, Expand, Cat) capture
only Go scalars/slices (shapes, dimensions) — no tensor saves needed.

**Engine release (`autograd/engine.go`):**

The engine performs three types of deterministic release during backward:

1. **Saved tensors** — Released immediately after `apply()`:
```go
inputGrads := node.gradFn.apply(grad)
for _, saved := range node.gradFn.saved {
    saved.Release()
}
```

2. **Old gradient accumulators** — when a variable is used multiple
   times (skip connections, shared weights), gradients accumulate.
   The old accumulator tensor is Released when replaced:
```go
if existing, ok := grads[input]; ok {
    acc := existing.Add(inputGrads[j])
    if existing != grad {  // guard against aliasing in self-ops
        existing.Release()
    }
    grads[input] = acc
}
```

3. **Stale leaf gradients** — `accumulateGrad` Releases the previous
   backward's gradient when accumulating across mini-batches:
```go
old := v.grad
v.grad = old.Add(grad)
old.Release()
```

If a saved tensor is shared (e.g., a weight used in two Matmul ops),
the refcount prevents premature free — each consumer's Release
decrements, and the tensor is freed only when the last consumer is done.

**What this achieves:**
- Saved-for-backward tensors freed during backward, not after GC
- Old gradient accumulators freed immediately when replaced
- ~1-2ns overhead per Retain/Release (Go-side atomic op)
- Same deterministic lifecycle as PyTorch's `SavedVariable` mechanism
- GC finalizers remain as safety net — no correctness risk
- No `runtime.GC()` calls needed in training loops

**Remaining gap (closed by Phase 4):**
Forward intermediate result tensors (Variable.data for non-leaf nodes)
still rely on GC for cleanup. Releasing them during backward is unsafe
because user code may hold references to intermediate Variables (e.g.,
reading `result.Data().Shape()` after backward). Phase 4 addresses this.

### Phase 4: CUDA OOM → GC callback

**Status: implemented**

When the CUDA caching allocator cannot satisfy an allocation from its
free-block pool, it invokes registered `FreeMemoryCallback` instances
before falling back to `cudaMalloc`. We register a callback that
triggers Go's garbage collector, freeing unreachable forward
intermediates on demand.

The `c10::FreeMemoryCallback` API has been stable since PyTorch 1.9
(June 2021) — same namespace, same signature. goDl requires
libtorch ≥ 2.0 (for the CUDA build; CPU builds have no version floor
for this feature since the callback is a no-op).

**C++ side (`shim.cpp`):**

```cpp
static void (*godl_gc_callback)(void) = nullptr;

#ifdef GODL_CUDA
namespace c10 {
class GoDlGCCallback : public FreeMemoryCallback {
public:
    bool Execute() override {
        if (godl_gc_callback) {
            godl_gc_callback();
            return true;  // may have freed memory
        }
        return false;
    }
};
REGISTER_FREE_MEMORY_CALLBACK("goDl", GoDlGCCallback);
} // namespace c10
#endif
```

**Go side (`gc_callback.go`):**

The callback fires while the allocator holds its `recursive_mutex`.
Go's GC finalizers run on a separate goroutine/thread, so they cannot
call `Free()` directly (the mutex is held by a different thread).

Solution: a pending-free queue. When `gcCallbackActive > 0`, the
`Free()` method queues handles instead of freeing directly. The
callback thread drains the queue — recursive mutex allows re-entry
from the same thread.

```
goTriggerGC() [on allocator thread, mutex held]:
  1. gcCallbackActive++
  2. runtime.GC()  →  finalizers queue handles to pendingFreeHandles
  3. sleep(1ms)    →  let finalizer goroutine process
  4. drain queue   →  Free() on this thread, re-enters recursive mutex
  5. repeat once   →  second GC pass for finalizer-freed objects
  6. gcCallbackActive--
```

**What this achieves:**
- Zero cost in happy path — callback only fires when VRAM is tight
- Forward intermediates freed on demand, not at GC's discretion
- No `runtime.GC()` calls in user code or engine
- Pending-free queue prevents deadlock with allocator's mutex
- If GC doesn't free enough, normal OOM error propagates

---

## What NOT to do

**Expose libtorch's C++ refcounting through CGo:**
Every `retain()`/`release()` is a CGo call (~100ns). With 36 ops and
multiple saved tensors each, this adds measurable overhead to every
forward+backward pass. Go-side atomic ops are ~1-2ns — 50-100x cheaper.

**Add `runtime.GC()` calls throughout the engine:**
This papers over the problem. The goal is deterministic lifecycle, not
GC pressure.

**Manual Release() in user training loops:**
The autograd engine knows exactly when tensors become dead. Pushing
this responsibility to users is a design failure.

---

## Implementation summary

| Phase | Status | Lines | Impact |
|-------|--------|-------|--------|
| Phase 1: nil backward graph | Done | ~10 in engine.go | Halves post-backward GC pressure |
| Phase 2: refcounting | Done | ~30 in tensor.go | Enables deterministic free |
| Phase 3: saved tensor tracking | Done | ~35 gradFn changes across ops.go + engine.go | Deterministic VRAM release during backward |
| Phase 4: CUDA OOM → GC callback | Done | ~15 in shim.cpp + ~80 in gc_callback.go | On-demand GC when VRAM is tight |

---

## Comparison with PyTorch

PyTorch benefits from CPython's reference counting: every Python object
(including tensors) has an immediate refcount decrement when it goes
out of scope. No GC needed for deterministic cleanup. PyTorch's
`SavedVariable` adds a second layer of refcounting for backward-saved
tensors, releasing them as backward progresses.

goDl uses Go's tracing GC, which is fundamentally non-deterministic
for C memory. Phase 1 (nil-out) + Phase 2+3 (Go-side refcounting +
deterministic release) brings goDl to near-parity with PyTorch for
saved-for-backward tensors. Phase 4 (CUDA OOM → GC callback) closes
the remaining gap for forward intermediates — the allocator asks Go
for GC exactly when VRAM is tight, with zero cost in the happy path.
