#include "quantize.cuh"
#include "codebook.cuh"

#include <cuda_fp16.h>

// Shared memory capacity for codebook centroids.
// At head_dim=256, codebook_size=256: 256*256*4 = 256KB, exceeds 48KB smem.
// We tile: load as many centroids as fit in shared memory per tile.
static constexpr int kSmemCentroidBytes = 32 * 1024;  // 32 KB tile

// Each thread handles one input vector. Codebook centroids are cached in shared
// memory in tiles of size floor(smem_bytes / (head_dim * sizeof(float))).
// For head_dim=128: floor(32768 / 512) = 64 centroids per tile.
// For head_dim=64:  floor(32768 / 256) = 128 centroids per tile.
__global__ void quantize_kernel(
    const __half*   __restrict__ vectors,     // [num_vectors x head_dim]
    const float*    __restrict__ centroids,   // [codebook_size x head_dim]
    CodebookIndex*  __restrict__ indices_out, // [num_vectors]
    int             num_vectors,
    int             head_dim,
    int             codebook_size)
{
    extern __shared__ float smem_centroids[];  // tile_size * head_dim floats

    int tile_size = kSmemCentroidBytes / (head_dim * sizeof(float));
    if (tile_size > codebook_size) tile_size = codebook_size;
    if (tile_size < 1) tile_size = 1;

    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = (vec_idx < num_vectors);

    // Load this thread's vector into registers to avoid repeated global reads.
    float vec_reg[kMaxHeadDim];
    if (active) {
        const __half* src = vectors + vec_idx * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            vec_reg[d] = __half2float(src[d]);
        }
    }

    float best_dist = 1e30f;
    int   best_idx  = 0;

    // Process codebook in tiles loaded into shared memory.
    for (int tile_start = 0; tile_start < codebook_size; tile_start += tile_size) {
        int tile_end = tile_start + tile_size;
        if (tile_end > codebook_size) tile_end = codebook_size;
        int actual_tile = tile_end - tile_start;

        // Cooperatively load centroids for this tile
        int total_floats = actual_tile * head_dim;
        for (int i = threadIdx.x; i < total_floats; i += blockDim.x) {
            smem_centroids[i] = centroids[(tile_start * head_dim) + i];
        }
        __syncthreads();

        if (active) {
            for (int c = 0; c < actual_tile; ++c) {
                const float* cent = smem_centroids + c * head_dim;
                float dist = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    float diff = vec_reg[d] - cent[d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx  = tile_start + c;
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        indices_out[vec_idx] = static_cast<CodebookIndex>(best_idx);
    }
}

void quantize_to_codebook(
    const __half*   vectors,
    int             num_vectors,
    int             head_dim,
    const Codebook* cb,
    CodebookIndex*  indices_out)
{
    // We need the centroids as a flat float array; they live inside the Codebook struct.
    // Extract the pointer to the centroids field on the device.
    // Since Codebook is a POD struct, centroids is at offset 0.
    const float* d_centroids = cb->centroids;

    int tile_size = kSmemCentroidBytes / (head_dim * sizeof(float));
    if (tile_size < 1) tile_size = 1;
    size_t smem_bytes = static_cast<size_t>(tile_size) * head_dim * sizeof(float);

    int block = 128;
    int grid  = (num_vectors + block - 1) / block;

    // Read codebook_size from device: copy just the int field
    int codebook_size_host;
    cudaMemcpy(&codebook_size_host,
               reinterpret_cast<const char*>(cb) + offsetof(Codebook, codebook_size),
               sizeof(int), cudaMemcpyDeviceToHost);

    quantize_kernel<<<grid, block, smem_bytes>>>(
        vectors, d_centroids, indices_out,
        num_vectors, head_dim, codebook_size_host);
}
