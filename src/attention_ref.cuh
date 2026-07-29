#pragma once

#include <cuda_fp16.h>

// Full-precision reference attention kernel.
//
// Computes output[q] = sum_k softmax(Q[q] . K[k] / sqrt(head_dim)) * V[k]
// for a single attention head with seq_len keys and values.
//
// Q:         [num_queries x head_dim] fp16, device memory.
// K:         [seq_len     x head_dim] fp16, device memory (contiguous, not ring).
// V:         [seq_len     x head_dim] fp16, device memory (contiguous, not ring).
// out:       [num_queries x head_dim] fp16, device memory (output).
// num_queries: typically 1 for decode, or seq_len for prefill.
// seq_len:   number of key/value tokens.
// head_dim:  dimension of each head.
void attention_ref(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       out,
    int           num_queries,
    int           seq_len,
    int           head_dim);
