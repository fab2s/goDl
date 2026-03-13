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
`Free()` method queues handles instead of freeing directly.

```
goTriggerGC() [on allocator thread, mutex held]:
  1. gcCallbackActive++
  2. runtime.GC()  →  finalizers queue handles to pendingFreeHandles
  3. sleep(1ms)    →  let finalizer goroutine process
  4. gcCallbackActive--
```

Critically, we do **not** drain the pending-free queue inside the
callback. The queued handles are drained lazily by the next
`Tensor.Free()` call outside the callback context:

```go
func (t *Tensor) Free() {
    if t.handle != nil {
        h := t.handle
        t.handle = nil
        if gcCallbackActive.Load() > 0 {
            queueFreeHandle(unsafe.Pointer(h))
        } else {
            drainPendingFrees()  // free any queued handles first
            C.godl_free_tensor(h)
        }
    }
}
```

This happens naturally during explicit `Release()` calls in the
backward pass or GC finalization between training steps.

**`runtime.KeepAlive` on CGo call sites (`tensor/ops.go`, `tensor/tensor.go`):**

Go's tracing GC can collect a `*tensor.Tensor` wrapper as soon as no
live Go code references it — even while a CGo call using its C++
handle is still running. If the GC callback fires during that CGo
call, the finalizer queues the handle for freeing, and a subsequent
drain frees it out from under the active C++ operation.

The standard Go solution (used by `os.File`, `net.Conn`, etc.) is
`runtime.KeepAlive(t)` after each CGo call. This inserts a reference
that the compiler cannot optimize away, guaranteeing the wrapper
survives until the CGo call returns:

```go
raw, err := libtorch.Matmul(t.raw, other.raw)
runtime.KeepAlive(t)
runtime.KeepAlive(other)
```

Applied to all ~50 CGo call sites in `tensor/ops.go` (arithmetic,
activations, reductions, convolutions, device transfer) and
`tensor/tensor.go` (Shape, DType, Device, data access methods).

**Why draining during the callback caused use-after-free:**

The original implementation drained the pending-free queue inside
`goTriggerGC()`. This was a use-after-free because the callback runs
on the **same thread and call stack** as the CGo operation that
triggered the allocation:

```
Go code: t.Matmul(other)
  → CGo call: godl_matmul(t.handle, other.handle)
    → C++: at::matmul needs temp memory
      → CUDA allocator: no free blocks → FreeMemoryCallback
        → goTriggerGC():
            runtime.GC() → finalizer marks `other` as dead (no more
                            Go references after the CGo boundary)
            drainPendingFrees() → frees other.handle
        ← returns to allocator
      ← allocator retries allocation
    ← at::matmul continues... using freed other.handle → SIGSEGV
```

Symptoms varied: corrupted vtable, "tensor does not have a device",
TLS assertion failures. Only triggered under VRAM pressure with large
tensors. The combination of `runtime.KeepAlive` (prevents premature
GC) and lazy draining (prevents same-stack freeing) eliminates this
class of bug entirely.

**Proactive VRAM-aware GC (`vram_budget.go`):**

Go's GC has zero visibility into C++ memory. A `*Tensor` wrapper is
~50 bytes on the Go heap but can hide megabytes of VRAM. When
unreachable wrappers ("zombies") accumulate, Go sees a nearly empty
heap and doesn't run GC. On modern NVIDIA drivers with unified/managed
memory, the driver silently satisfies allocations by spilling VRAM to
system RAM — system RAM grows unboundedly.

The proactive GC bridges this gap via two components:

**C++ allocation tracking** (`shim.cpp`): An atomic counter tracks
total bytes held by CUDA tensors, piggybacked on existing `wrap()` and
`godl_free_tensor()` calls. Only CUDA tensors are counted — CPU
tensors don't contribute to VRAM pressure. Zero extra CGo roundtrips.

```cpp
static std::atomic<int64_t> godl_cuda_alloc_bytes{0};

static TorchTensor wrap(torch::Tensor t) {
    auto* p = new torch::Tensor(std::move(t));
    if (p->is_cuda()) {
        godl_cuda_alloc_bytes += p->nbytes();  // ~1ns atomic add
    }
    return (TorchTensor)p;
}
```

**Go-side periodic check** (`vram_budget.go`): On init, the VRAM
budget is set to 90% of physical VRAM (via `cudaMemGetInfo`). Every
100 tensor creations, `EnforceVRAMBudget()` reads the C++ counter and
triggers `runtime.GC()` if over budget:

```go
func EnforceVRAMBudget() {
    if vramBudget <= 0 { return }               // CPU-only: no-op
    if wrapCount.Add(1)%100 != 0 { return }     // ~1ns atomic
    if CUDAAllocatedBytes() > vramBudget {       // one CGo call
        runtime.GC()                             // ~2ms, rare
    }
}
```

Cost: one atomic increment per `wrap()` (~1ns). One CGo call every 100
wraps (~100ns amortized to ~1ns). `runtime.GC()` only when over budget
(~2ms, rare after warmup as the working set stabilizes). CPU-only
builds pay nothing — the `vramBudget <= 0` check exits immediately.

Override via `GODL_VRAM_BUDGET` environment variable (e.g. `"0.90"`).

**Hard allocator cap** (`SetMemoryFraction`): Safety net behind the
proactive GC. `setMemoryFraction(0.95)` caps the allocator — above
this, allocations fail and trigger the GC callback. This catches cases
where the proactive GC didn't free enough (the working set genuinely
needs >90% VRAM).

**Three layers of defense:**

| Layer | Trigger | Mechanism | Cost |
|-------|---------|-----------|------|
| Proactive GC | 90% tracked CUDA bytes | atomic counter + `runtime.GC()` | ~1ns/wrap, ~2ms when triggered |
| Allocator cap | 95% VRAM | `setMemoryFraction` → allocator fails → GC callback | zero until triggered |
| OOM callback | true CUDA OOM | `FreeMemoryCallback` → `goTriggerGC` | zero until triggered |

**What this achieves:**
- Zero cost in happy path — proactive GC rarely fires after warmup
- Forward intermediates freed proactively, not at GC's discretion
- No `runtime.GC()` calls in user code or engine
- Pending-free queue prevents deadlock with allocator's mutex
- KeepAlive prevents premature GC of tensor wrappers during CGo calls
- Lazy drain prevents use-after-free on the allocator's call stack
- Three-layer defense prevents silent spill to system RAM
- If all layers fail, normal OOM error propagates

---

## What NOT to do

**Expose libtorch's C++ refcounting through CGo:**
Every `retain()`/`release()` is a CGo call (~100ns). With 36 ops and
multiple saved tensors each, this adds measurable overhead to every
forward+backward pass. Go-side atomic ops are ~1-2ns — 50-100x cheaper.

**Drain pending frees inside the GC callback:**
The callback runs on the same thread/call-stack as the CGo operation
that triggered the allocation. Draining there frees handles that C++
code higher in the stack is still using. Always drain lazily from the
next `Free()` call outside the callback.

**Omit `runtime.KeepAlive` after CGo calls:**
Without KeepAlive, Go's GC can collect a `*tensor.Tensor` wrapper
while its C++ handle is passed to a CGo function. If the GC callback
fires during that call, the finalizer queues the handle, and the next
drain frees it. This is a use-after-free that only manifests under
VRAM pressure — extremely hard to reproduce and debug.

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
| Phase 4: VRAM-aware GC | Done | ~20 in shim.cpp + ~80 in gc_callback.go + ~90 in vram_budget.go + ~50 KeepAlive in tensor/ | Proactive GC at 90% VRAM, hard cap at 95%, OOM callback as last resort |
| Phase 5: autograd.Scope | Done | ~65 in scope.go + ~5 in variable.go | Deterministic batch-level cleanup, no GC needed |

---

### Phase 5: autograd.Scope — deterministic batch-level cleanup

**Status: implemented**

Phases 1-4 make backward deterministic and add VRAM safety nets, but
forward intermediate result tensors (Variable.data for non-leaf nodes)
still rely on `runtime.GC()` for cleanup. On GPUs, GC pauses cause
pipeline stalls — `runtime.GC()` is stop-the-world, scans the entire
heap, and blocks GPU work submission.

`autograd.Scope` eliminates this by tracking intermediate Variables
and freeing their C++ tensors at batch boundaries:

```go
for loader.Next() {
    scope := autograd.NewScope()
    // ... forward, backward, step, read metrics ...
    scope.Close() // frees all intermediate tensors instantly
}
```

**What gets tracked:** Only Variables created by autograd ops (via the
internal `newVar()` helper). Every op in `ops.go` flows through
`newVar()`, which calls `track(v)` to register the Variable with the
active scope.

**What does NOT get tracked:**
- Leaf parameters (created via `NewVariable` in module constructors)
- User inputs (created via `NewVariable` in training loops)
- Detached state (created via `NewVariable` in `DetachState`)
- Error variables (no valid tensor to release)

This distinction is critical. `NewVariable` wraps external tensors
whose lifecycle is managed by the caller. Op results are fresh tensors
created by the autograd system — the scope can safely own them.

**Why not track NewVariable too?**

Three interacting problems:
1. Phase 1 nils `gradFn` on all non-leaf nodes during backward. After
   backward, intermediates have `requiresGrad=true, gradFn=nil` —
   identical to leaf parameters. Any skip condition based on these
   fields would either skip everything or nothing.
2. `DetachState` creates new Variables via `NewVariable` that share
   the same `*tensor.Tensor` as the original intermediate. If both
   are tracked, Close() releases the shared tensor, breaking the
   next batch.
3. User inputs (`NewVariable(batch.Image, false)`) wrap tensors the
   scope doesn't own. Releasing them invalidates the caller's data.

Tracking only op results via `newVar()` sidesteps all three problems.

**Thread safety:** The scope uses a mutex for the tracked list.
Parallel graph execution (goroutines in independent branches) can
call `track()` concurrently. The mutex is uncontended in the common
(sequential) case.

**Cost:** One atomic pointer load per `newVar()` (~1ns) when no scope
is active. One mutex lock+append when active (~10ns uncontended).
`Close()` walks the list calling `Release()` + nil — microseconds for
a typical batch of ~500-1000 intermediates.

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
saved-for-backward tensors. Phase 4 (VRAM-aware GC) adds safety nets
for forward intermediates. Phase 5 (autograd.Scope) closes the
remaining gap — forward intermediates are freed deterministically at
batch boundaries, matching PyTorch's scope-exit behavior without
relying on reference counting.
