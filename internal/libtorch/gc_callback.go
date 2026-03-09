// gc_callback.go — CUDA OOM → Go GC bridge.
//
// When the CUDA caching allocator cannot satisfy an allocation from its
// free-block pool, it invokes registered FreeMemoryCallback instances
// before falling back to cudaMalloc. The C++ side (shim.cpp) registers
// GoDlGCCallback which calls goTriggerGC via a function pointer.
//
// goTriggerGC triggers Go's garbage collector to finalize unreachable
// tensor wrappers. Because the allocator holds a recursive mutex during
// the callback, Go finalizers (which run on a separate goroutine/thread)
// cannot free tensors directly without deadlocking. Instead, Free()
// queues handles into a pending list when gcCallbackActive > 0.
//
// IMPORTANT: We do NOT drain the pending list during the callback.
// The callback runs on the same thread and call stack as the CGo
// operation that triggered the allocation. Draining would free tensor
// handles that C++ code above us is still using — classic
// use-after-free. Instead, the pending list is drained by the next
// Free() call outside the callback context.
//
// The CUDA allocator has fallback mechanisms (cudaMalloc, releasing
// all cached blocks) that handle the case where the callback doesn't
// immediately reclaim memory.
//
// Cost: zero in the happy path — the callback only fires when VRAM is
// genuinely exhausted. When it fires, ~2ms for GC.
package libtorch

/*
#include "shim.h"

// Forward-declare the Go-exported callback.
extern void goTriggerGC();

// C helper to pass the Go function pointer to the registration function.
static inline void _godl_register_gc_cb() {
    godl_register_cuda_gc_callback(goTriggerGC);
}
*/
import "C"

import (
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// gcCallbackActive is incremented while inside the CUDA allocator's
// FreeMemoryCallback. When > 0, Tensor.Free() queues handles instead
// of calling godl_free_tensor directly (which would deadlock on the
// allocator's mutex held by a different thread).
var gcCallbackActive atomic.Int32

var (
	pendingFreeMu      sync.Mutex
	pendingFreeHandles []unsafe.Pointer
)

// queueFreeHandle adds a C tensor handle to the deferred-free list.
// Called from Tensor.Free() when a GC callback is in flight.
func queueFreeHandle(h unsafe.Pointer) {
	pendingFreeMu.Lock()
	pendingFreeHandles = append(pendingFreeHandles, h)
	pendingFreeMu.Unlock()
}

// drainPendingFrees frees all queued tensor handles on the calling
// thread. This thread holds the allocator's recursive mutex, so the
// re-entrant free calls succeed without deadlock.
func drainPendingFrees() {
	pendingFreeMu.Lock()
	items := pendingFreeHandles
	pendingFreeHandles = nil
	pendingFreeMu.Unlock()
	for _, h := range items {
		C.godl_free_tensor(C.TorchTensor(h))
	}
}

// goTriggerGC is called from C++ when the CUDA caching allocator needs
// memory. It triggers Go's GC to finalize unreachable tensor wrappers,
// queuing their handles for deferred freeing.
//
// We intentionally do NOT drain the pending-free queue here. This
// function runs on the same thread and call stack as the CGo operation
// that triggered the CUDA allocation. Draining would call
// godl_free_tensor for handles that C++ code above us (e.g., matmul,
// to_device) is still actively using — a use-after-free that manifests
// as SIGSEGV with varying symptoms (corrupted vtable, "tensor does not
// have a device", TLS assertion failures).
//
// The queued handles are drained by the next Tensor.Free() call that
// occurs outside the callback context, which happens naturally during
// explicit Release() calls in the backward pass or GC finalization
// between training steps.
//
//export goTriggerGC
func goTriggerGC() {
	gcCallbackActive.Add(1)

	// GC identifies unreachable tensors and schedules finalizers.
	// The finalizer goroutine runs release() → sees gcCallbackActive > 0
	// → adds handles to the pending list instead of calling Free directly.
	runtime.GC()
	time.Sleep(time.Millisecond) // let the finalizer goroutine process

	gcCallbackActive.Add(-1)
}

func init() {
	if CUDAAvailable() {
		C._godl_register_gc_cb()
	}
}
