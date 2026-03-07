// shim.cpp — C++ implementations of the C shim functions.
//
// Each function wraps a libtorch C++ call and:
// 1. Catches any C++ exception
// 2. Returns the error message as a malloc'd C string
// 3. Returns NULL on success
//
// This prevents C++ exceptions from crossing the CGo boundary (which
// would crash the process with no useful error message).

#include "shim.h"
#include <torch/torch.h>
#include <cstring>
#include <string>

// Helper: convert a C++ exception to a malloc'd C string.
static char* make_error(const std::string& msg) {
    char* err = (char*)malloc(msg.size() + 1);
    if (err) {
        memcpy(err, msg.c_str(), msg.size() + 1);
    }
    return err;
}

// Helper: convert our dtype constant to torch::ScalarType.
static torch::ScalarType to_scalar_type(int dtype) {
    switch (dtype) {
        case GODL_FLOAT16:  return torch::kFloat16;
        case GODL_BFLOAT16: return torch::kBFloat16;
        case GODL_FLOAT32:  return torch::kFloat32;
        case GODL_FLOAT64:  return torch::kFloat64;
        case GODL_INT32:    return torch::kInt32;
        case GODL_INT64:    return torch::kInt64;
        default:            return torch::kFloat32;
    }
}

// Helper: convert our dtype constant back from torch::ScalarType.
static int from_scalar_type(torch::ScalarType st) {
    switch (st) {
        case torch::kFloat16:  return GODL_FLOAT16;
        case torch::kBFloat16: return GODL_BFLOAT16;
        case torch::kFloat32:  return GODL_FLOAT32;
        case torch::kFloat64:  return GODL_FLOAT64;
        case torch::kInt32:    return GODL_INT32;
        case torch::kInt64:    return GODL_INT64;
        default:               return GODL_FLOAT32;
    }
}

// Helper: convert our device constant to torch::Device.
static torch::Device to_device(int device) {
    if (device == GODL_CUDA) {
        return torch::Device(torch::kCUDA, 0);
    }
    return torch::Device(torch::kCPU);
}

// Helper: convert torch::Device back to our constant.
static int from_device(const torch::Device& dev) {
    if (dev.is_cuda()) return GODL_CUDA;
    return GODL_CPU;
}

// Helper: wrap a new torch::Tensor into a heap-allocated pointer.
// The caller (Go) owns this pointer and must call godl_free_tensor.
static TorchTensor wrap(torch::Tensor t) {
    return (TorchTensor)(new torch::Tensor(std::move(t)));
}

// Helper: unwrap a TorchTensor handle back to a reference.
static torch::Tensor& unwrap(TorchTensor t) {
    return *((torch::Tensor*)t);
}

// Helper: build IntArrayRef from C array.
static torch::IntArrayRef make_shape(int64_t* shape, int ndim) {
    return torch::IntArrayRef(shape, ndim);
}

// --- Tensor creation ---

extern "C" char* godl_zeros(int64_t* shape, int ndim, int dtype, int device,
                             TorchTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device));
        *result = wrap(torch::zeros(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_ones(int64_t* shape, int ndim, int dtype, int device,
                            TorchTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device));
        *result = wrap(torch::ones(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_rand(int64_t* shape, int ndim, int dtype, int device,
                            TorchTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device));
        *result = wrap(torch::rand(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_from_blob(void* data, int64_t* shape, int ndim,
                                 int dtype, int device, TorchTensor* result) {
    try {
        auto options = torch::TensorOptions().dtype(to_scalar_type(dtype));
        // from_blob does not take ownership — clone to get an independent copy.
        auto t = torch::from_blob(data, make_shape(shape, ndim), options).clone();
        if (device == GODL_CUDA) {
            t = t.to(torch::kCUDA);
        }
        *result = wrap(std::move(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_linspace(double start, double end, int64_t steps,
                                int dtype, int device, TorchTensor* result) {
    try {
        auto options = torch::TensorOptions().dtype(to_scalar_type(dtype));
        auto t = torch::linspace(start, end, steps, options);
        if (device == GODL_CUDA) {
            t = t.to(torch::kCUDA);
        }
        *result = wrap(std::move(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_expand(TorchTensor t, int64_t* new_shape, int ndim,
                              TorchTensor* result) {
    try {
        // expand returns a view; contiguous() makes an owned copy.
        *result = wrap(unwrap(t).expand(make_shape(new_shape, ndim)).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Tensor lifecycle ---

extern "C" void godl_free_tensor(TorchTensor t) {
    if (t) {
        delete (torch::Tensor*)t;
    }
}

// --- Tensor metadata ---

extern "C" int godl_ndim(TorchTensor t) {
    return (int)unwrap(t).dim();
}

extern "C" int64_t godl_shape(TorchTensor t, int dim) {
    return unwrap(t).size(dim);
}

extern "C" int godl_dtype(TorchTensor t) {
    return from_scalar_type(unwrap(t).scalar_type());
}

extern "C" int godl_device(TorchTensor t) {
    return from_device(unwrap(t).device());
}

extern "C" int64_t godl_numel(TorchTensor t) {
    return unwrap(t).numel();
}

// --- Data access ---

extern "C" char* godl_copy_data(TorchTensor t, void* buffer,
                                 int64_t buffer_bytes) {
    try {
        auto tensor = unwrap(t);
        // Move to CPU if on another device
        if (!tensor.is_cpu()) {
            tensor = tensor.to(torch::kCPU);
        }
        // Ensure contiguous layout
        tensor = tensor.contiguous();
        int64_t data_bytes = tensor.numel() * tensor.element_size();
        if (buffer_bytes < data_bytes) {
            return make_error("buffer too small: need " +
                              std::to_string(data_bytes) + " bytes, got " +
                              std::to_string(buffer_bytes));
        }
        memcpy(buffer, tensor.data_ptr(), data_bytes);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Basic operations ---

extern "C" char* godl_add(TorchTensor a, TorchTensor b, TorchTensor* result) {
    try {
        *result = wrap(unwrap(a) + unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_mul(TorchTensor a, TorchTensor b, TorchTensor* result) {
    try {
        *result = wrap(unwrap(a) * unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_matmul(TorchTensor a, TorchTensor b,
                              TorchTensor* result) {
    try {
        *result = wrap(torch::matmul(unwrap(a), unwrap(b)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_relu(TorchTensor a, TorchTensor* result) {
    try {
        *result = wrap(torch::relu(unwrap(a)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_sigmoid(TorchTensor a, TorchTensor* result) {
    try {
        *result = wrap(torch::sigmoid(unwrap(a)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_tanh_op(TorchTensor a, TorchTensor* result) {
    try {
        *result = wrap(torch::tanh(unwrap(a)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Additional operations (used by autograd backward) ---

extern "C" char* godl_sub(TorchTensor a, TorchTensor b, TorchTensor* result) {
    try {
        *result = wrap(unwrap(a) - unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_transpose(TorchTensor t, int dim0, int dim1,
                                 TorchTensor* result) {
    try {
        *result = wrap(unwrap(t).transpose(dim0, dim1).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_sum(TorchTensor t, TorchTensor* result) {
    try {
        *result = wrap(unwrap(t).sum());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_sum_dim(TorchTensor t, int dim, int keepdim,
                               TorchTensor* result) {
    try {
        *result = wrap(unwrap(t).sum(dim, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_ones_like(TorchTensor t, TorchTensor* result) {
    try {
        *result = wrap(torch::ones_like(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_mul_scalar(TorchTensor t, double scalar,
                                  TorchTensor* result) {
    try {
        *result = wrap(unwrap(t) * scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_gt_scalar(TorchTensor t, double scalar,
                                 TorchTensor* result) {
    try {
        // Returns float mask (0.0 or 1.0) matching the input dtype,
        // suitable for element-wise multiplication in backward passes.
        auto mask = torch::gt(unwrap(t), scalar);
        *result = wrap(mask.to(unwrap(t).scalar_type()));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_reshape(TorchTensor t, int64_t* shape, int ndim,
                               TorchTensor* result) {
    try {
        *result = wrap(unwrap(t).reshape(make_shape(shape, ndim)).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_exp(TorchTensor t, TorchTensor* result) {
    try {
        *result = wrap(torch::exp(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_log(TorchTensor t, TorchTensor* result) {
    try {
        *result = wrap(torch::log(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_randn(int64_t* shape, int ndim, int dtype, int device,
                              TorchTensor* result) {
    try {
        auto options = torch::TensorOptions()
            .dtype(to_scalar_type(dtype))
            .device(to_device(device));
        *result = wrap(torch::randn(make_shape(shape, ndim), options));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_add_scalar(TorchTensor t, double scalar,
                                  TorchTensor* result) {
    try {
        *result = wrap(unwrap(t) + scalar);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_neg(TorchTensor t, TorchTensor* result) {
    try {
        *result = wrap(-unwrap(t));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_max_dim(TorchTensor t, int dim, int keepdim,
                               TorchTensor* result) {
    try {
        // std::get<0> gets the values (not the indices)
        *result = wrap(std::get<0>(unwrap(t).max(dim, keepdim != 0)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_softmax(TorchTensor t, int dim, TorchTensor* result) {
    try {
        *result = wrap(torch::softmax(unwrap(t), dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_select(TorchTensor t, int dim, int64_t index,
                              TorchTensor* result) {
    try {
        *result = wrap(unwrap(t).select(dim, index).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_zeros_like(TorchTensor t, TorchTensor* result) {
    try {
        *result = wrap(torch::zeros_like(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_select_scatter(TorchTensor input, TorchTensor src,
                                      int dim, int64_t index,
                                      TorchTensor* result) {
    try {
        auto out = unwrap(input).clone();
        out.select(dim, index).copy_(unwrap(src));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Slicing and concatenation ---

extern "C" char* godl_narrow(TorchTensor t, int dim, int64_t start,
                              int64_t length, TorchTensor* result) {
    try {
        *result = wrap(unwrap(t).narrow(dim, start, length).contiguous());
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_narrow_scatter(TorchTensor input, TorchTensor src,
                                      int dim, int64_t start,
                                      TorchTensor* result) {
    try {
        auto out = unwrap(input).clone();
        out.narrow(dim, start, unwrap(src).size(dim)).copy_(unwrap(src));
        *result = wrap(out);
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_cat2(TorchTensor a, TorchTensor b, int dim,
                            TorchTensor* result) {
    try {
        *result = wrap(torch::cat({unwrap(a), unwrap(b)}, dim));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Reduction ---

extern "C" char* godl_mean_dim(TorchTensor t, int dim, int keepdim,
                                TorchTensor* result) {
    try {
        *result = wrap(unwrap(t).mean(dim, keepdim != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Indexing ---

extern "C" char* godl_index_select(TorchTensor t, int dim, TorchTensor index,
                                    TorchTensor* result) {
    try {
        *result = wrap(torch::index_select(unwrap(t), dim, unwrap(index)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_index_add(TorchTensor t, int dim, TorchTensor index,
                                 TorchTensor src, TorchTensor* result) {
    try {
        // Out-of-place: returns t with src scattered at index positions.
        *result = wrap(unwrap(t).index_add(dim, unwrap(index), unwrap(src)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Element-wise math ---

extern "C" char* godl_sqrt(TorchTensor t, TorchTensor* result) {
    try {
        *result = wrap(torch::sqrt(unwrap(t)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_div(TorchTensor a, TorchTensor b, TorchTensor* result) {
    try {
        *result = wrap(unwrap(a) / unwrap(b));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Convolution ---

extern "C" char* godl_conv2d(TorchTensor input, TorchTensor weight,
                              TorchTensor bias,
                              int64_t* stride, int64_t* padding,
                              int64_t* dilation,
                              int64_t groups, TorchTensor* result) {
    try {
        auto in = unwrap(input);
        auto w = unwrap(weight);
        c10::optional<torch::Tensor> b;
        if (bias != nullptr) {
            b = unwrap(bias);
        }
        *result = wrap(torch::conv2d(in, w, b,
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(dilation, 2),
            groups));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_conv2d_backward(TorchTensor grad_output, TorchTensor input,
                                       TorchTensor weight,
                                       int64_t* stride, int64_t* padding,
                                       int64_t* dilation,
                                       int64_t groups, int compute_bias,
                                       TorchTensor* grad_input,
                                       TorchTensor* grad_weight,
                                       TorchTensor* grad_bias) {
    try {
        auto go_ = unwrap(grad_output);
        auto in = unwrap(input);
        auto w = unwrap(weight);

        c10::OptionalIntArrayRef bias_sizes = c10::nullopt;
        std::vector<int64_t> bias_sizes_vec;
        if (compute_bias) {
            bias_sizes_vec = {w.size(0)};
            bias_sizes = bias_sizes_vec;
        }

        std::vector<int64_t> output_padding = {0, 0};
        auto result = at::convolution_backward(
            go_, in, w,
            bias_sizes,
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(dilation, 2),
            false, // transposed
            output_padding,
            groups,
            {true, true, compute_bias != 0}
        );

        *grad_input = wrap(std::get<0>(result));
        *grad_weight = wrap(std::get<1>(result));
        if (compute_bias) {
            *grad_bias = wrap(std::get<2>(result));
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Transposed convolution ---

extern "C" char* godl_conv_transpose2d(TorchTensor input, TorchTensor weight,
                                        TorchTensor bias,
                                        int64_t* stride, int64_t* padding,
                                        int64_t* output_padding, int64_t* dilation,
                                        int64_t groups, TorchTensor* result) {
    try {
        auto in = unwrap(input);
        auto w = unwrap(weight);
        c10::optional<torch::Tensor> b;
        if (bias != nullptr) {
            b = unwrap(bias);
        }
        *result = wrap(torch::conv_transpose2d(in, w, b,
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(output_padding, 2),
            groups,
            torch::IntArrayRef(dilation, 2)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_conv_transpose2d_backward(TorchTensor grad_output,
                                                  TorchTensor input,
                                                  TorchTensor weight,
                                                  int64_t* stride, int64_t* padding,
                                                  int64_t* output_padding,
                                                  int64_t* dilation,
                                                  int64_t groups, int compute_bias,
                                                  TorchTensor* grad_input,
                                                  TorchTensor* grad_weight,
                                                  TorchTensor* grad_bias) {
    try {
        auto go_ = unwrap(grad_output);
        auto in = unwrap(input);
        auto w = unwrap(weight);

        c10::OptionalIntArrayRef bias_sizes = c10::nullopt;
        std::vector<int64_t> bias_sizes_vec;
        if (compute_bias) {
            bias_sizes_vec = {w.size(1) * groups};
            bias_sizes = bias_sizes_vec;
        }

        auto result = at::convolution_backward(
            go_, in, w,
            bias_sizes,
            torch::IntArrayRef(stride, 2),
            torch::IntArrayRef(padding, 2),
            torch::IntArrayRef(dilation, 2),
            true, // transposed
            torch::IntArrayRef(output_padding, 2),
            groups,
            {true, true, compute_bias != 0}
        );

        *grad_input = wrap(std::get<0>(result));
        *grad_weight = wrap(std::get<1>(result));
        if (compute_bias) {
            *grad_bias = wrap(std::get<2>(result));
        }
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Adaptive average pooling ---

extern "C" char* godl_adaptive_avg_pool2d(TorchTensor input, int64_t* output_size,
                                           TorchTensor* result) {
    try {
        *result = wrap(at::adaptive_avg_pool2d(
            unwrap(input), torch::IntArrayRef(output_size, 2)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_adaptive_avg_pool2d_backward(TorchTensor grad_output,
                                                     TorchTensor input,
                                                     TorchTensor* grad_input) {
    try {
        *grad_input = wrap(at::_adaptive_avg_pool2d_backward(
            unwrap(grad_output), unwrap(input)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Grid sampling ---

extern "C" char* godl_grid_sample(TorchTensor input, TorchTensor grid,
                                   int mode, int padding_mode,
                                   int align_corners, TorchTensor* result) {
    try {
        *result = wrap(at::grid_sampler(
            unwrap(input), unwrap(grid), mode, padding_mode, align_corners != 0));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_grid_sample_backward(TorchTensor grad_output,
                                            TorchTensor input, TorchTensor grid,
                                            int mode, int padding_mode,
                                            int align_corners,
                                            TorchTensor* grad_input,
                                            TorchTensor* grad_grid) {
    try {
        auto result = at::grid_sampler_2d_backward(
            unwrap(grad_output), unwrap(input), unwrap(grid),
            mode, padding_mode, align_corners != 0,
            {true, true});
        *grad_input = wrap(std::get<0>(result));
        *grad_grid = wrap(std::get<1>(result));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Dtype casting ---

extern "C" char* godl_to_dtype(TorchTensor t, int dtype, TorchTensor* result) {
    try {
        *result = wrap(unwrap(t).to(to_scalar_type(dtype)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

extern "C" char* godl_all_finite(TorchTensor t, int* result) {
    try {
        auto& tensor = unwrap(t);
        *result = torch::isfinite(tensor).all().item<bool>() ? 1 : 0;
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Device operations ---

extern "C" char* godl_to_device(TorchTensor t, int device,
                                 TorchTensor* result) {
    try {
        *result = wrap(unwrap(t).to(to_device(device)));
        return nullptr;
    } catch (const std::exception& e) {
        return make_error(e.what());
    }
}

// --- Utility ---

extern "C" void godl_free_string(char* s) {
    free(s);
}

extern "C" int godl_cuda_is_available(void) {
    return torch::cuda::is_available() ? 1 : 0;
}

extern "C" int godl_cuda_device_count(void) {
    return (int)torch::cuda::device_count();
}
