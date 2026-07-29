#include "attention_mixed.cuh"
#include "codebook.cuh"

#include <cuda_fp16.h>
#include <math.h>

// Tile sizes for loading codebook indices and full-precision rows.
static constexpr int kCompTile = 128;  // compressed entries per tile
static constexpr int kFullTile = 64;   // full-precision entries per tile

// Shared memory layout per block:
//   [0                     .. cb_k_floats)         -> K codebook centroids (float)
//   [cb_k_floats           .. 2*cb_k_floats)        -> V codebook centroids (float)
//   [2*cb_k_floats         .. 2*cb_k_floats + tile) -> compressed index tile (uint8)
//   above aligned to next float boundary
// Then full-precision K/V tiles follow.
//
// For simplicity we use a fixed smem budget and compute offsets at kernel launch.

// The kernel runs one block per query. threadIdx.x == 0 does all arithmetic;
// other threads cooperate on loading tiles.
__global__ void attention_mixed_kernel(
    const __half*        __restrict__ Q,         // [num_queries x head_dim]
    const CodebookIndex* __restrict__ comp_k,    // [comp_count]
    const CodebookIndex* __restrict__ comp_v,    // [comp_count]
    const __half*        __restrict__ full_k,    // [full_count x head_dim]
    const __half*        __restrict__ full_v,    // [full_count x head_dim]
    const float*         __restrict__ cb_k_data, // [codebook_size x head_dim]
    const float*         __restrict__ cb_v_data, // [codebook_size x head_dim]
    __half*              __restrict__ out,        // [num_queries x head_dim]
    int                  num_queries,
    int                  comp_count,
    int                  full_count,
    int                  head_dim,
    int                  codebook_size)
{
    extern __shared__ char smem_raw[];

    // Partition shared memory
    float*         smem_cbk    = reinterpret_cast<float*>(smem_raw);
    float*         smem_cbv    = smem_cbk + codebook_size * head_dim;
    CodebookIndex* smem_idx    = reinterpret_cast<CodebookIndex*>(smem_cbv + codebook_size * head_dim);
    // Align smem_idx_end to float boundary
    size_t idx_bytes  = ((kCompTile * sizeof(CodebookIndex) + 3) / 4) * 4;
    __half*        smem_fk = reinterpret_cast<__half*>(
        reinterpret_cast<char*>(smem_idx) + idx_bytes);
    __half*        smem_fv = smem_fk + kFullTile * head_dim;

    int q_idx = blockIdx.x;
    if (q_idx >= num_queries) return;

    // Load entire codebooks into shared memory (one-time cost)
    {
        int total = codebook_size * head_dim;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            smem_cbk[i] = cb_k_data[i];
            smem_cbv[i] = cb_v_data[i];
        }
    }
    __syncthreads();

    const __half* q_row = Q + q_idx * head_dim;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Online softmax state
    float m   = -1e30f;
    float d   = 0.0f;
    float acc[256];
    for (int i = 0; i < head_dim; ++i) acc[i] = 0.0f;

    // -----------------------------------------------------------------------
    // Pass 1: compressed region
    // -----------------------------------------------------------------------
    for (int tile_start = 0; tile_start < comp_count; tile_start += kCompTile) {
        int tile_end = min(tile_start + kCompTile, comp_count);
        int tile_len = tile_end - tile_start;

        // Cooperatively load index tile
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            smem_idx[i] = comp_k[tile_start + i];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int t = 0; t < tile_len; ++t) {
                // Reconstruct K from codebook
                const float* k_cent = smem_cbk + smem_idx[t] * head_dim;
                float dot = 0.0f;
                for (int dd = 0; dd < head_dim; ++dd) {
                    dot += __half2float(q_row[dd]) * k_cent[dd];
                }
                dot *= scale;

                // Reconstruct V centroid index (separate tile not loaded yet;
                // we load it lazily since we need the V for this token)
                CodebookIndex v_idx = comp_v[tile_start + t];
                const float* v_cent = smem_cbv + v_idx * head_dim;

                // Online softmax update
                float m_new     = fmaxf(m, dot);
                float exp_shift = expf(m - m_new);
                float exp_dot   = expf(dot - m_new);

                for (int dd = 0; dd < head_dim; ++dd) {
                    acc[dd] = acc[dd] * exp_shift + v_cent[dd] * exp_dot;
                }
                d = d * exp_shift + exp_dot;
                m = m_new;
            }
        }
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Pass 2: full-precision region
    // -----------------------------------------------------------------------
    for (int tile_start = 0; tile_start < full_count; tile_start += kFullTile) {
        int tile_end = min(tile_start + kFullTile, full_count);
        int tile_len = tile_end - tile_start;

        // Load K and V tiles cooperatively
        for (int i = threadIdx.x; i < tile_len * head_dim; i += blockDim.x) {
            smem_fk[i] = full_k[tile_start * head_dim + i];
            smem_fv[i] = full_v[tile_start * head_dim + i];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            for (int t = 0; t < tile_len; ++t) {
                const __half* k_row = smem_fk + t * head_dim;
                float dot = 0.0f;
                for (int dd = 0; dd < head_dim; ++dd) {
                    dot += __half2float(q_row[dd]) * __half2float(k_row[dd]);
                }
                dot *= scale;

                const __half* v_row = smem_fv + t * head_dim;
                float m_new     = fmaxf(m, dot);
                float exp_shift = expf(m - m_new);
                float exp_dot   = expf(dot - m_new);

                for (int dd = 0; dd < head_dim; ++dd) {
                    acc[dd] = acc[dd] * exp_shift + __half2float(v_row[dd]) * exp_dot;
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

void attention_mixed(
    const __half*         Q,
    const CodebookIndex*  comp_k,
    const CodebookIndex*  comp_v,
    const __half*         full_k,
    const __half*         full_v,
    const Codebook*       cb_k,
    const Codebook*       cb_v,
    __half*               out,
    int                   num_queries,
    int                   comp_count,
    int                   full_count,
    int                   head_dim)
{
    // Read codebook metadata from device
    int codebook_size_host;
    cudaMemcpy(&codebook_size_host,
               reinterpret_cast<const char*>(cb_k) + offsetof(Codebook, codebook_size),
               sizeof(int), cudaMemcpyDeviceToHost);

    const float* d_cbk = cb_k->centroids;
    const float* d_cbv = cb_v->centroids;

    size_t cb_floats    = (size_t)codebook_size_host * head_dim;
    size_t smem_codebooks = 2 * cb_floats * sizeof(float);
    size_t smem_comp_idx  = ((kCompTile * sizeof(CodebookIndex) + 3) / 4) * 4;
    size_t smem_full_kv   = 2 * kFullTile * head_dim * sizeof(__half);
    size_t smem_total     = smem_codebooks + smem_comp_idx + smem_full_kv;

    int block = 32;
    attention_mixed_kernel<<<num_queries, block, smem_total>>>(
        Q, comp_k, comp_v, full_k, full_v,
        d_cbk, d_cbv,
        out,
        num_queries,
        comp_count,
        full_count,
        head_dim,
        codebook_size_host);
}
