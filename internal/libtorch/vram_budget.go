// vram_budget.go — proactive VRAM-aware GC trigger.
//
// Go's tracing GC has zero visibility into C++ memory. A *Tensor wrapper
// is ~50 bytes on the Go heap but can hide megabytes of VRAM. When
// unreachable wrappers ("zombies") accumulate, Go sees a nearly empty
// heap and doesn't bother running GC. On modern NVIDIA drivers with
// unified/managed memory, the CUDA driver silently satisfies allocations
// by spilling VRAM to system RAM — system RAM grows unboundedly.
//
// This file bridges the gap: the C++ shim tracks allocated CUDA bytes
// via an atomic counter (piggybacked on existing wrap/free calls, zero
// extra CGo roundtrips). Every N tensor creations, Go reads the counter
// and triggers runtime.GC() if allocated bytes exceed a configurable
// fraction of physical VRAM.
//
// Layers of defense:
//  1. Proactive (this file): atomic counter check → GC before VRAM fills
//  2. Allocator cap (SetMemoryFraction): hard 95% cap → allocator fails → GC callback
//  3. Last resort (gc_callback.go): true CUDA OOM → GC callback
package libtorch

/*
#include "shim.h"
*/
import "C"

import (
	"os"
	"runtime"
	"strconv"
	"sync/atomic"
)

// DefaultVRAMBudget is the default fraction of physical VRAM used as
// the proactive GC trigger. When goDl's tracked CUDA allocations exceed
// this fraction, runtime.GC() fires to finalize zombie tensor wrappers.
const DefaultVRAMBudget = 0.90

var (
	wrapCount  atomic.Int64
	vramBudget int64 // bytes; 0 disables (CPU-only builds)
)

const checkEveryN = 100

// CUDAAllocatedBytes returns the total bytes currently held by CUDA
// tensors tracked by goDl. CPU tensors are excluded.
func CUDAAllocatedBytes() int64 {
	return int64(C.godl_cuda_allocated_bytes())
}

// CUDAMemInfo returns the free and total physical VRAM on device 0.
// Returns (0, 0) on CPU-only builds.
func CUDAMemInfo() (free, total int64) {
	var f, t C.int64_t
	C.godl_cuda_mem_info(&f, &t)
	return int64(f), int64(t)
}

// VRAMBudget returns the current VRAM budget in bytes. Returns 0 if
// CUDA is not available or the budget is disabled.
func VRAMBudget() int64 {
	return vramBudget
}

// EnforceVRAMBudget is called from tensor.wrap() on every tensor creation.
// It performs a lightweight periodic check: every checkEveryN wraps, it
// reads the C++ allocation counter (one CGo call) and triggers GC if
// CUDA allocations exceed the budget.
//
// Cost: one atomic increment per call (~1ns). One CGo call every 100
// wraps (~100ns amortized to ~1ns). runtime.GC() only when over budget.
func EnforceVRAMBudget() {
	if vramBudget <= 0 {
		return
	}
	if wrapCount.Add(1)%checkEveryN != 0 {
		return
	}
	if CUDAAllocatedBytes() > vramBudget {
		runtime.GC()
	}
}

// initVRAMBudget sets the VRAM budget based on physical VRAM.
// Called from init() in gc_callback.go when CUDA is available.
func initVRAMBudget() {
	_, total := CUDAMemInfo()
	if total <= 0 {
		return
	}

	fraction := DefaultVRAMBudget
	if env := os.Getenv("GODL_VRAM_BUDGET"); env != "" {
		if v, err := strconv.ParseFloat(env, 64); err == nil && v > 0 && v <= 1 {
			fraction = v
		}
	}
	vramBudget = int64(float64(total) * fraction)
}
