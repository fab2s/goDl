// Build tag: include CUDA libraries when building with the "cuda" tag.
//
// Build with:  go build -tags cuda ./...
// Or set in Dockerfile/environment.

//go:build cuda

package libtorch

// #cgo LDFLAGS: -Wl,--no-as-needed -ltorch_cuda -lc10_cuda
import "C"
