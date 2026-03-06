// shim.h — C interface to libtorch's C++ API.
//
// CGo can only call C functions, not C++. This header declares a minimal
// set of C functions that wrap libtorch operations. The implementations
// live in shim.cpp (compiled as C++) and export C linkage via extern "C".
//
// Design: every function that can fail returns an error string (caller must
// free it). NULL means success. This avoids C++ exceptions crossing the
// CGo boundary, which would crash the process.

#ifndef GODL_SHIM_H
#define GODL_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a torch::Tensor. Go code holds this pointer but
// never dereferences it — all operations go through shim functions.
typedef void* TorchTensor;

// --- Tensor creation ---

// Create a tensor filled with zeros. Shape is an array of dims.
// Returns error string on failure (caller frees), NULL on success.
char* godl_zeros(int64_t* shape, int ndim, int dtype, int device,
                 TorchTensor* result);

// Create a tensor filled with ones.
char* godl_ones(int64_t* shape, int ndim, int dtype, int device,
                TorchTensor* result);

// Create a tensor filled with random values from uniform distribution [0, 1).
char* godl_rand(int64_t* shape, int ndim, int dtype, int device,
                TorchTensor* result);

// Create a tensor from a raw data pointer (copies the data).
char* godl_from_blob(void* data, int64_t* shape, int ndim, int dtype,
                     int device, TorchTensor* result);

// --- Tensor lifecycle ---

// Free a tensor. Must be called exactly once per tensor.
void godl_free_tensor(TorchTensor t);

// --- Tensor metadata ---

int godl_ndim(TorchTensor t);
int64_t godl_shape(TorchTensor t, int dim);
int godl_dtype(TorchTensor t);
int godl_device(TorchTensor t);
int64_t godl_numel(TorchTensor t);

// --- Data access ---

// Copy tensor data to a pre-allocated buffer. Tensor is moved to CPU first
// if necessary. Buffer must be large enough to hold numel * element_size bytes.
char* godl_copy_data(TorchTensor t, void* buffer, int64_t buffer_bytes);

// --- Basic operations ---
// All ops create a new tensor. The caller owns it and must free it.

char* godl_add(TorchTensor a, TorchTensor b, TorchTensor* result);
char* godl_mul(TorchTensor a, TorchTensor b, TorchTensor* result);
char* godl_matmul(TorchTensor a, TorchTensor b, TorchTensor* result);
char* godl_relu(TorchTensor a, TorchTensor* result);
char* godl_sigmoid(TorchTensor a, TorchTensor* result);
char* godl_tanh_op(TorchTensor a, TorchTensor* result);

// --- Additional operations (used by autograd backward) ---

char* godl_sub(TorchTensor a, TorchTensor b, TorchTensor* result);
char* godl_transpose(TorchTensor t, int dim0, int dim1, TorchTensor* result);
char* godl_sum(TorchTensor t, TorchTensor* result);
char* godl_sum_dim(TorchTensor t, int dim, int keepdim, TorchTensor* result);
char* godl_ones_like(TorchTensor t, TorchTensor* result);
char* godl_mul_scalar(TorchTensor t, double scalar, TorchTensor* result);
char* godl_gt_scalar(TorchTensor t, double scalar, TorchTensor* result);
char* godl_reshape(TorchTensor t, int64_t* shape, int ndim, TorchTensor* result);
char* godl_exp(TorchTensor t, TorchTensor* result);
char* godl_log(TorchTensor t, TorchTensor* result);
char* godl_randn(int64_t* shape, int ndim, int dtype, int device,
                  TorchTensor* result);
char* godl_add_scalar(TorchTensor t, double scalar, TorchTensor* result);
char* godl_neg(TorchTensor t, TorchTensor* result);
char* godl_max_dim(TorchTensor t, int dim, int keepdim, TorchTensor* result);
char* godl_softmax(TorchTensor t, int dim, TorchTensor* result);
char* godl_select(TorchTensor t, int dim, int64_t index, TorchTensor* result);
char* godl_zeros_like(TorchTensor t, TorchTensor* result);
char* godl_select_scatter(TorchTensor input, TorchTensor src, int dim,
                          int64_t index, TorchTensor* result);

// --- Reduction ---

char* godl_mean_dim(TorchTensor t, int dim, int keepdim, TorchTensor* result);

// --- Indexing ---

char* godl_index_select(TorchTensor t, int dim, TorchTensor index,
                        TorchTensor* result);
char* godl_index_add(TorchTensor t, int dim, TorchTensor index,
                     TorchTensor src, TorchTensor* result);

// --- Slicing and concatenation ---

char* godl_narrow(TorchTensor t, int dim, int64_t start, int64_t length,
                  TorchTensor* result);
char* godl_narrow_scatter(TorchTensor input, TorchTensor src, int dim,
                          int64_t start, TorchTensor* result);
char* godl_cat2(TorchTensor a, TorchTensor b, int dim, TorchTensor* result);

// --- Element-wise math ---

char* godl_sqrt(TorchTensor t, TorchTensor* result);
char* godl_div(TorchTensor a, TorchTensor b, TorchTensor* result);

// --- Convolution ---

// 2D convolution forward.
// bias may be NULL for no bias. stride/padding/dilation are 2-element arrays.
char* godl_conv2d(TorchTensor input, TorchTensor weight, TorchTensor bias,
                  int64_t* stride, int64_t* padding, int64_t* dilation,
                  int64_t groups, TorchTensor* result);

// 2D convolution backward. Computes gradients for input, weight, and optionally bias.
// Out-pointers for disabled gradients (compute_*=0) are not written.
char* godl_conv2d_backward(TorchTensor grad_output, TorchTensor input,
                           TorchTensor weight,
                           int64_t* stride, int64_t* padding, int64_t* dilation,
                           int64_t groups, int compute_bias,
                           TorchTensor* grad_input, TorchTensor* grad_weight,
                           TorchTensor* grad_bias);

// --- Device operations ---

// Move tensor to a different device. Returns a new tensor.
char* godl_to_device(TorchTensor t, int device, TorchTensor* result);

// --- Utility ---

// Free an error string returned by any shim function.
void godl_free_string(char* s);

// Check if CUDA is available.
int godl_cuda_is_available(void);

// Get number of CUDA devices.
int godl_cuda_device_count(void);

// --- DType constants (matching torch::kFloat32, etc.) ---
#define GODL_FLOAT32 6
#define GODL_FLOAT64 7
#define GODL_INT32   3
#define GODL_INT64   4

// --- Device constants ---
#define GODL_CPU  0
#define GODL_CUDA 1

#ifdef __cplusplus
}
#endif

#endif // GODL_SHIM_H
