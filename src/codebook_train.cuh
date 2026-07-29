#pragma once

#include "codebook.cuh"
#include <cuda_fp16.h>

// Train a codebook for one attention head using Lloyd's K-means algorithm.
//
// vectors:        [num_vectors x head_dim] matrix of fp16 calibration K or V vectors,
//                 row-major, in device memory.
// num_vectors:    number of calibration vectors.
// head_dim:       dimension of each vector (must be <= kMaxHeadDim).
// codebook_size:  number of centroids to learn (must be <= kMaxCodebookSize).
// num_iters:      number of Lloyd's iterations to run.
// cb_out:         output Codebook struct in device memory.
// seed:           RNG seed used for centroid initialization.
void train_codebook(
    const __half* vectors,
    int           num_vectors,
    int           head_dim,
    int           codebook_size,
    int           num_iters,
    Codebook*     cb_out,
    unsigned int  seed = 42);
