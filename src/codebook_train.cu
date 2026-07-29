#include "codebook_train.cuh"
#include "codebook.cuh"

#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cstring>
#include <cstdio>

// ---------------------------------------------------------------------------
// Kernel: initialize centroids by picking codebook_size random vectors
// ---------------------------------------------------------------------------
__global__ void init_centroids_kernel(
    const __half* __restrict__ vectors,
    int           num_vectors,
    int           head_dim,
    int           codebook_size,
    float*        centroids,      // [codebook_size x head_dim]
    unsigned int  seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= codebook_size) return;

    curandState rng;
    curand_init(seed, i, 0, &rng);
    int src_row = curand(&rng) % num_vectors;

    const __half* src = vectors + src_row * head_dim;
    float*        dst = centroids + i * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        dst[d] = __half2float(src[d]);
    }
}

// ---------------------------------------------------------------------------
// Kernel: assign each vector to its nearest centroid
// ---------------------------------------------------------------------------
__global__ void assign_kernel(
    const __half* __restrict__ vectors,    // [num_vectors x head_dim]
    const float*  __restrict__ centroids,  // [codebook_size x head_dim]
    int*          __restrict__ assignments, // [num_vectors]
    int           num_vectors,
    int           head_dim,
    int           codebook_size)
{
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= num_vectors) return;

    const __half* vec = vectors + vec_idx * head_dim;
    float best_dist = 1e30f;
    int   best_centroid = 0;

    for (int c = 0; c < codebook_size; ++c) {
        float dist = l2_dist_sq_fp16(vec, centroids + c * head_dim, head_dim);
        if (dist < best_dist) {
            best_dist = dist;
            best_centroid = c;
        }
    }
    assignments[vec_idx] = best_centroid;
}

// ---------------------------------------------------------------------------
// Kernel: accumulate sums per centroid (atomic adds)
// Each thread handles one vector.
// ---------------------------------------------------------------------------
__global__ void accumulate_kernel(
    const __half* __restrict__ vectors,     // [num_vectors x head_dim]
    const int*    __restrict__ assignments, // [num_vectors]
    float*        __restrict__ sums,        // [codebook_size x head_dim] zeroed by caller
    int*          __restrict__ counts,      // [codebook_size] zeroed by caller
    int           num_vectors,
    int           head_dim)
{
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= num_vectors) return;

    int c = assignments[vec_idx];
    atomicAdd(&counts[c], 1);

    const __half* src = vectors + vec_idx * head_dim;
    float*        dst = sums + c * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        atomicAdd(&dst[d], __half2float(src[d]));
    }
}

// ---------------------------------------------------------------------------
// Kernel: update centroids from sums/counts; if a centroid has no assigned
// vectors, reinitialize it to a random input vector to avoid dead centroids.
// ---------------------------------------------------------------------------
__global__ void update_centroids_kernel(
    float*        __restrict__ centroids,   // [codebook_size x head_dim] (in/out)
    const float*  __restrict__ sums,        // [codebook_size x head_dim]
    const int*    __restrict__ counts,      // [codebook_size]
    const __half* __restrict__ vectors,     // [num_vectors x head_dim]
    int           num_vectors,
    int           head_dim,
    int           codebook_size,
    unsigned int  seed,
    int           iter)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= codebook_size) return;

    float* centroid = centroids + c * head_dim;

    if (counts[c] > 0) {
        float inv = 1.0f / static_cast<float>(counts[c]);
        const float* s = sums + c * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            centroid[d] = s[d] * inv;
        }
    } else {
        // Dead centroid: pick a random vector to restart it
        curandState rng;
        curand_init(seed + iter * 1000u, c, 0, &rng);
        int src_row = curand(&rng) % num_vectors;
        const __half* src = vectors + src_row * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            centroid[d] = __half2float(src[d]);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: copy trained float32 centroids into the output Codebook struct
// ---------------------------------------------------------------------------
__global__ void write_codebook_kernel(
    Codebook*    __restrict__ cb_out,
    const float* __restrict__ centroids,
    int          codebook_size,
    int          head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = codebook_size * head_dim;
    if (idx >= total) return;
    cb_out->centroids[idx] = centroids[idx];
    if (idx == 0) {
        cb_out->codebook_size = codebook_size;
        cb_out->head_dim      = head_dim;
    }
}

// ---------------------------------------------------------------------------
// Host function
// ---------------------------------------------------------------------------
void train_codebook(
    const __half* vectors,
    int           num_vectors,
    int           head_dim,
    int           codebook_size,
    int           num_iters,
    Codebook*     cb_out,
    unsigned int  seed)
{
    // Workspace
    float* d_centroids = nullptr;
    float* d_sums      = nullptr;
    int*   d_counts    = nullptr;
    int*   d_assigns   = nullptr;

    cudaMalloc(&d_centroids, codebook_size * head_dim * sizeof(float));
    cudaMalloc(&d_sums,      codebook_size * head_dim * sizeof(float));
    cudaMalloc(&d_counts,    codebook_size * sizeof(int));
    cudaMalloc(&d_assigns,   num_vectors   * sizeof(int));

    // Initialize centroids
    int block = 128;
    init_centroids_kernel<<<(codebook_size + block - 1) / block, block>>>(
        vectors, num_vectors, head_dim, codebook_size, d_centroids, seed);

    for (int iter = 0; iter < num_iters; ++iter) {
        // Assignment step
        assign_kernel<<<(num_vectors + block - 1) / block, block>>>(
            vectors, d_centroids, d_assigns,
            num_vectors, head_dim, codebook_size);

        // Zero accumulators
        cudaMemset(d_sums,   0, codebook_size * head_dim * sizeof(float));
        cudaMemset(d_counts, 0, codebook_size * sizeof(int));

        // Accumulate sums
        accumulate_kernel<<<(num_vectors + block - 1) / block, block>>>(
            vectors, d_assigns, d_sums, d_counts,
            num_vectors, head_dim);

        // Update centroids
        update_centroids_kernel<<<(codebook_size + block - 1) / block, block>>>(
            d_centroids, d_sums, d_counts,
            vectors, num_vectors, head_dim, codebook_size,
            seed, iter);
    }

    // Write result into Codebook struct
    int total = codebook_size * head_dim;
    write_codebook_kernel<<<(total + block - 1) / block, block>>>(
        cb_out, d_centroids, codebook_size, head_dim);

    cudaDeviceSynchronize();

    cudaFree(d_centroids);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_assigns);
}
