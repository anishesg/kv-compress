#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>

// Maximum supported codebook size (uint8 index range)
static constexpr int kMaxCodebookSize = 256;

// Maximum supported head dimension
static constexpr int kMaxHeadDim = 256;

// Typedef for compressed token index; uint8 supports up to 256 centroids
using CodebookIndex = uint8_t;

// A codebook for one attention head (K or V separately).
// centroids is stored row-major: centroids[i * head_dim + d] is dimension d of centroid i.
// Stored in float32 for stable distance computation during training and lookup.
struct Codebook {
    float centroids[kMaxCodebookSize * kMaxHeadDim];
    int   codebook_size;  // number of entries (1..256)
    int   head_dim;       // dimensionality of each entry
};

// Compute squared L2 distance between a float32 vector and one centroid row.
// vec has length head_dim; centroid_row points to head_dim contiguous floats.
__device__ __forceinline__ float l2_dist_sq(
    const float* __restrict__ vec,
    const float* __restrict__ centroid_row,
    int head_dim)
{
    float dist = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        float diff = vec[d] - centroid_row[d];
        dist += diff * diff;
    }
    return dist;
}

// Same as above but accepts a half-precision input vector; converts each element
// to float32 before computing the distance to avoid accumulation error.
__device__ __forceinline__ float l2_dist_sq_fp16(
    const __half* __restrict__ vec,
    const float*  __restrict__ centroid_row,
    int head_dim)
{
    float dist = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        float diff = __half2float(vec[d]) - centroid_row[d];
        dist += diff * diff;
    }
    return dist;
}

// Find the nearest centroid index for a single fp16 vector.
// Iterates over all codebook_size centroids and returns the argmin.
__device__ __forceinline__ CodebookIndex nearest_centroid_fp16(
    const __half*    __restrict__ vec,
    const Codebook*  __restrict__ cb)
{
    float best_dist = 1e30f;
    CodebookIndex best_idx = 0;
    for (int i = 0; i < cb->codebook_size; ++i) {
        float d = l2_dist_sq_fp16(vec, cb->centroids + i * cb->head_dim, cb->head_dim);
        if (d < best_dist) {
            best_dist = d;
            best_idx  = static_cast<CodebookIndex>(i);
        }
    }
    return best_idx;
}
