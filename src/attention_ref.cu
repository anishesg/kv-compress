#include "attention_ref.cuh"
#include <cuda_fp16.h>
#include <math.h>

// Tile size for K/V loading into shared memory.
// At head_dim=128: one tile row = 256 bytes (fp16). 48KB smem / 256B = 192 rows.
// We use 64 as a conservative tile for portability.
static constexpr int kRefTile = 64;

// One block handles one query. Threads collaborate on loading K/V tiles and
// computing dot products. Online softmax (Dao et al.) accumulates over tiles.
__global__ void attention_ref_kernel(
    const __half* __restrict__ Q,    // [num_queries x head_dim]
    const __half* __restrict__ K,    // [seq_len     x head_dim]
    const __half* __restrict__ V,    // [seq_len     x head_dim]
    __half*       __restrict__ out,  // [num_queries x head_dim]
    int           num_queries,
    int           seq_len,
    int           head_dim)
{
    // Shared memory layout: tile of K rows, then tile of V rows
    extern __shared__ __half smem[];
    __half* sk = smem;                          // [kRefTile x head_dim]
    __half* sv = smem + kRefTile * head_dim;    // [kRefTile x head_dim]

    int q_idx = blockIdx.x;
    if (q_idx >= num_queries) return;

    const __half* q_row = Q + q_idx * head_dim;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Online softmax state
    float m = -1e30f;  // running max
    float d = 0.0f;    // running denominator
    float acc[256];    // running output accumulator; head_dim <= kMaxHeadDim=256
    for (int i = 0; i < head_dim; ++i) acc[i] = 0.0f;

    for (int tile_start = 0; tile_start < seq_len; tile_start += kRefTile) {
        int tile_end = min(tile_start + kRefTile, seq_len);
        int tile_len = tile_end - tile_start;

        // Load K and V tile cooperatively
        int floats_per_row = head_dim;  // half elements
        for (int i = threadIdx.x; i < tile_len * floats_per_row; i += blockDim.x) {
            sk[i] = K[(tile_start * head_dim) + i];
            sv[i] = V[(tile_start * head_dim) + i];
        }
        __syncthreads();

        // Each thread processes the full tile (single-thread-per-query layout)
        if (threadIdx.x == 0) {
            for (int t = 0; t < tile_len; ++t) {
                // Dot product Q . K[t]
                float dot = 0.0f;
                const __half* k_row = sk + t * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    dot += __half2float(q_row[d]) * __half2float(k_row[d]);
                }
                dot *= scale;

                // Online softmax update
                float m_new = fmaxf(m, dot);
                float exp_shift = expf(m - m_new);
                float exp_dot   = expf(dot - m_new);

                const __half* v_row = sv + t * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    acc[d] = acc[d] * exp_shift + __half2float(v_row[d]) * exp_dot;
                }
                d = d * exp_shift + exp_dot;
                m = m_new;
            }
        }
        __syncthreads();
    }

    // Write output
    if (threadIdx.x == 0) {
        __half* out_row = out + q_idx * head_dim;
        float inv_d = 1.0f / d;
        for (int i = 0; i < head_dim; ++i) {
            out_row[i] = __float2half(acc[i] * inv_d);
        }
    }
}

void attention_ref(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       out,
    int           num_queries,
    int           seq_len,
    int           head_dim)
{
    // One block per query, one warp (query does serial work in thread 0;
    // other threads assist only with cooperative tile loads).
    int block = 32;
    // Shared memory: 2 tiles (K and V) of kRefTile rows each
    size_t smem = 2 * kRefTile * head_dim * sizeof(__half);

    attention_ref_kernel<<<num_queries, block, smem>>>(
        Q, K, V, out, num_queries, seq_len, head_dim);
}
