#pragma once

#include "codebook.cuh"
#include <cuda_fp16.h>

// Quantize a batch of fp16 KV vectors to their nearest codebook index.
//
// vectors:       [num_vectors x head_dim] fp16 matrix, device memory.
// num_vectors:   number of vectors to quantize.
// head_dim:      vector dimensionality (must match cb->head_dim).
// cb:            trained codebook, device memory.
// indices_out:   [num_vectors] output array of CodebookIndex, device memory.
void quantize_to_codebook(
    const __half*   vectors,
    int             num_vectors,
    int             head_dim,
    const Codebook* cb,
    CodebookIndex*  indices_out);
