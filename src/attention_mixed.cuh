#pragma once

#include "codebook.cuh"
#include <cuda_fp16.h>

// Mixed-precision attention kernel.
//
// Computes attention over two regions in a single online-softmax pass:
//   1. Compressed region: comp_count tokens stored as CodebookIndex values.
//      K vectors are reconstructed from cb_k on-the-fly; V from cb_v.
//   2. Full-precision region: full_count tokens stored as fp16.
//      Oldest full-precision entry is at index 0 of full_k/full_v.
//
// Q:           [num_queries x head_dim] fp16, device.
// comp_k:      [comp_count] CodebookIndex array, device.
// comp_v:      [comp_count] CodebookIndex array, device.
// full_k:      [full_count x head_dim] fp16, device (contiguous, oldest first).
// full_v:      [full_count x head_dim] fp16, device.
// cb_k:        trained K codebook, device.
// cb_v:        trained V codebook, device.
// out:         [num_queries x head_dim] fp16, device (output).
// num_queries: typically 1 for decode.
// comp_count:  number of compressed tokens.
// full_count:  number of full-precision tokens.
// head_dim:    attention head dimension.
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
    int                   head_dim);
