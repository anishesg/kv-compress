#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <stdexcept>

#include "codebook.cuh"
#include "codebook_train.cuh"
#include "quantize.cuh"
#include "attention_mixed.cuh"

// ---------------------------------------------------------------------------
// Helper: allocate a device Codebook struct and return it as a torch tensor
// of bytes so Python can hold a reference-counted handle to it.
// We expose the raw device pointer as a Python integer for kernel calls.
// ---------------------------------------------------------------------------

// Train a codebook from fp16 calibration vectors.
// vectors: [N, D] float16 CUDA tensor
// codebook_size: number of centroids (<=256)
// num_iters: number of Lloyd iterations
// seed: RNG seed
// Returns: (centroids_tensor [codebook_size, D] float32, codebook_ptr int)
//   centroids_tensor is a CPU copy of the trained centroids for inspection.
//   codebook_ptr is a Python int holding the raw device Codebook* address.
std::tuple<torch::Tensor, int64_t> bind_train_codebook(
    torch::Tensor vectors,
    int           codebook_size,
    int           num_iters,
    unsigned int  seed)
{
    TORCH_CHECK(vectors.is_cuda(),      "vectors must be a CUDA tensor");
    TORCH_CHECK(vectors.is_contiguous(),"vectors must be contiguous");
    TORCH_CHECK(vectors.dtype() == torch::kFloat16, "vectors must be float16");
    TORCH_CHECK(vectors.dim() == 2, "vectors must be 2-D [N, D]");
    TORCH_CHECK(codebook_size >= 1 && codebook_size <= kMaxCodebookSize,
                "codebook_size must be in [1, 256]");
    TORCH_CHECK(vectors.size(1) <= kMaxHeadDim,
                "head_dim exceeds kMaxHeadDim=256");

    at::cuda::CUDAGuard guard(vectors.device());

    int num_vectors = static_cast<int>(vectors.size(0));
    int head_dim    = static_cast<int>(vectors.size(1));

    Codebook* d_cb = nullptr;
    cudaMalloc(&d_cb, sizeof(Codebook));

    train_codebook(
        reinterpret_cast<const __half*>(vectors.data_ptr<at::Half>()),
        num_vectors,
        head_dim,
        codebook_size,
        num_iters,
        d_cb,
        seed);

    cudaDeviceSynchronize();

    // Copy centroids back to host as a float32 tensor for inspection
    Codebook h_cb;
    cudaMemcpy(&h_cb, d_cb, sizeof(Codebook), cudaMemcpyDeviceToHost);

    auto centroids = torch::from_blob(
        h_cb.centroids,
        {codebook_size, head_dim},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();  // clone to own the data

    return {centroids, reinterpret_cast<int64_t>(d_cb)};
}

// Free a device Codebook allocated by bind_train_codebook.
void bind_free_codebook(int64_t ptr)
{
    if (ptr != 0) {
        cudaFree(reinterpret_cast<void*>(ptr));
    }
}

// Retrieve centroids from a device Codebook pointer as a CPU float32 tensor.
torch::Tensor bind_get_centroids(int64_t codebook_ptr, int codebook_size, int head_dim)
{
    TORCH_CHECK(codebook_ptr != 0, "null codebook pointer");
    Codebook h_cb;
    cudaMemcpy(&h_cb, reinterpret_cast<const void*>(codebook_ptr),
               sizeof(Codebook), cudaMemcpyDeviceToHost);
    auto out = torch::from_blob(
        h_cb.centroids,
        {codebook_size, head_dim},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
    return out;
}

// Quantize fp16 vectors to codebook indices.
// vectors:      [N, D] float16 CUDA
// codebook_ptr: device Codebook* as int64
// Returns:      [N] uint8 CUDA tensor of indices
torch::Tensor bind_quantize(torch::Tensor vectors, int64_t codebook_ptr)
{
    TORCH_CHECK(vectors.is_cuda(),       "vectors must be a CUDA tensor");
    TORCH_CHECK(vectors.is_contiguous(), "vectors must be contiguous");
    TORCH_CHECK(vectors.dtype() == torch::kFloat16, "vectors must be float16");
    TORCH_CHECK(vectors.dim() == 2, "vectors must be 2-D [N, D]");
    TORCH_CHECK(codebook_ptr != 0, "null codebook pointer");

    at::cuda::CUDAGuard guard(vectors.device());

    int num_vectors = static_cast<int>(vectors.size(0));

    auto indices = torch::empty(
        {num_vectors},
        torch::TensorOptions().dtype(torch::kUInt8).device(vectors.device()));

    quantize_to_codebook(
        reinterpret_cast<const __half*>(vectors.data_ptr<at::Half>()),
        num_vectors,
        static_cast<int>(vectors.size(1)),
        reinterpret_cast<const Codebook*>(codebook_ptr),
        reinterpret_cast<CodebookIndex*>(indices.data_ptr<uint8_t>()));

    return indices;
}

// Mixed-precision attention.
// Q:           [num_queries, head_dim] float16 CUDA
// comp_k:      [comp_count] uint8 CUDA
// comp_v:      [comp_count] uint8 CUDA
// full_k:      [full_count, head_dim] float16 CUDA (contiguous, oldest first)
// full_v:      [full_count, head_dim] float16 CUDA
// cb_k_ptr:    device Codebook* for K, as int64
// cb_v_ptr:    device Codebook* for V, as int64
// Returns:     [num_queries, head_dim] float16 CUDA
torch::Tensor bind_attention_mixed(
    torch::Tensor Q,
    torch::Tensor comp_k,
    torch::Tensor comp_v,
    torch::Tensor full_k,
    torch::Tensor full_v,
    int64_t       cb_k_ptr,
    int64_t       cb_v_ptr)
{
    TORCH_CHECK(Q.is_cuda() && Q.is_contiguous(), "Q must be contiguous CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(Q.dim() == 2, "Q must be 2-D");

    TORCH_CHECK(comp_k.is_cuda() && comp_k.is_contiguous(), "comp_k must be contiguous CUDA");
    TORCH_CHECK(comp_k.dtype() == torch::kUInt8, "comp_k must be uint8");
    TORCH_CHECK(comp_v.is_cuda() && comp_v.is_contiguous(), "comp_v must be contiguous CUDA");
    TORCH_CHECK(comp_v.dtype() == torch::kUInt8, "comp_v must be uint8");

    TORCH_CHECK(full_k.is_cuda() && full_k.is_contiguous(), "full_k must be contiguous CUDA");
    TORCH_CHECK(full_k.dtype() == torch::kFloat16, "full_k must be float16");
    TORCH_CHECK(full_v.is_cuda() && full_v.is_contiguous(), "full_v must be contiguous CUDA");
    TORCH_CHECK(full_v.dtype() == torch::kFloat16, "full_v must be float16");

    TORCH_CHECK(cb_k_ptr != 0 && cb_v_ptr != 0, "null codebook pointer");

    at::cuda::CUDAGuard guard(Q.device());

    int num_queries = static_cast<int>(Q.size(0));
    int head_dim    = static_cast<int>(Q.size(1));
    int comp_count  = static_cast<int>(comp_k.size(0));
    int full_count  = static_cast<int>(full_k.size(0));

    auto out = torch::empty_like(Q);

    attention_mixed(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const CodebookIndex*>(comp_k.data_ptr<uint8_t>()),
        reinterpret_cast<const CodebookIndex*>(comp_v.data_ptr<uint8_t>()),
        reinterpret_cast<const __half*>(full_k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(full_v.data_ptr<at::Half>()),
        reinterpret_cast<const Codebook*>(cb_k_ptr),
        reinterpret_cast<const Codebook*>(cb_v_ptr),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        num_queries,
        comp_count,
        full_count,
        head_dim);

    return out;
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("train_codebook",    &bind_train_codebook,
          "Train a K-means codebook from fp16 calibration vectors. "
          "Returns (centroids_cpu_f32, device_ptr_int64).");
    m.def("free_codebook",     &bind_free_codebook,
          "Free a device Codebook allocated by train_codebook.");
    m.def("get_centroids",     &bind_get_centroids,
          "Copy centroids from device Codebook to CPU float32 tensor.");
    m.def("quantize",          &bind_quantize,
          "Quantize fp16 vectors to uint8 codebook indices.");
    m.def("attention_mixed",   &bind_attention_mixed,
          "Mixed-precision attention over compressed + full-precision KV cache.");
}
