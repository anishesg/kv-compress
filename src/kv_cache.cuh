#pragma once

#include "codebook.cuh"
#include "quantize.cuh"

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>

// MixedKVCache holds the KV state for a single attention head.
//
// Tokens with age < age_threshold are stored at full fp16 precision.
// Tokens with age >= age_threshold are stored as uint8 codebook indices.
//
// "age" of a token = (current_seq_len - 1) - token_position.
// i.e., the most recently appended token has age 0.
//
// Memory layout:
//   full_k / full_v : ring buffer of the most recent `age_threshold` tokens,
//                     each entry is head_dim fp16 values.
//   comp_k / comp_v : linear array of older tokens (oldest first),
//                     each entry is 1 byte (CodebookIndex).
//
// The compress() method drains the oldest entries from the full-precision
// ring into the compressed arrays using the provided codebook.

struct MixedKVCache {
    // --- Configuration (set once at construction) ---
    int head_dim;
    int age_threshold;   // max number of full-precision entries kept
    int max_seq_len;     // maximum total sequence length supported

    // --- Full-precision ring buffer (device memory) ---
    __half* full_k;      // [age_threshold x head_dim]
    __half* full_v;      // [age_threshold x head_dim]
    int     ring_head;   // index of the oldest full-precision entry in the ring
    int     full_count;  // number of valid full-precision entries (0..age_threshold)

    // --- Compressed region (device memory) ---
    // Growing array; comp_count entries are valid.
    CodebookIndex* comp_k;   // [max_seq_len - age_threshold]
    CodebookIndex* comp_v;   // [max_seq_len - age_threshold]
    int            comp_count;

    // --- Total tokens appended so far ---
    int seq_len;

    // Allocate device memory for the cache.
    static MixedKVCache create(int head_dim, int age_threshold, int max_seq_len) {
        MixedKVCache c{};
        c.head_dim      = head_dim;
        c.age_threshold = age_threshold;
        c.max_seq_len   = max_seq_len;
        c.ring_head     = 0;
        c.full_count    = 0;
        c.comp_count    = 0;
        c.seq_len       = 0;

        cudaMalloc(&c.full_k, (size_t)age_threshold * head_dim * sizeof(__half));
        cudaMalloc(&c.full_v, (size_t)age_threshold * head_dim * sizeof(__half));
        int comp_capacity = max_seq_len - age_threshold;
        if (comp_capacity > 0) {
            cudaMalloc(&c.comp_k, (size_t)comp_capacity * sizeof(CodebookIndex));
            cudaMalloc(&c.comp_v, (size_t)comp_capacity * sizeof(CodebookIndex));
        } else {
            c.comp_k = nullptr;
            c.comp_v = nullptr;
        }
        return c;
    }

    void free_device_memory() {
        if (full_k) { cudaFree(full_k); full_k = nullptr; }
        if (full_v) { cudaFree(full_v); full_v = nullptr; }
        if (comp_k) { cudaFree(comp_k); comp_k = nullptr; }
        if (comp_v) { cudaFree(comp_v); comp_v = nullptr; }
    }

    // Append a single new token's K and V vectors (host fp16 arrays of length head_dim).
    // If the ring is full, the oldest entry is evicted; compress() must be called
    // separately to move it to the compressed region.
    void append_host(const __half* k_vec, const __half* v_vec) {
        int slot = (ring_head + full_count) % age_threshold;
        cudaMemcpy(full_k + slot * head_dim, k_vec,
                   head_dim * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(full_v + slot * head_dim, v_vec,
                   head_dim * sizeof(__half), cudaMemcpyHostToDevice);

        if (full_count < age_threshold) {
            ++full_count;
        } else {
            // Ring is full; oldest slot was overwritten (it should have been
            // compressed first via compress()).
            ring_head = (ring_head + 1) % age_threshold;
        }
        ++seq_len;
    }

    // Compress all full-precision entries that exceed the age threshold into
    // the compressed arrays. In the steady state this is called once per new
    // token after append_host() when full_count == age_threshold.
    //
    // cb_k, cb_v: trained codebooks (device pointers) for this head's K and V.
    void compress(const Codebook* cb_k, const Codebook* cb_v) {
        while (full_count > age_threshold) {
            // Quantize the oldest full-precision entry
            __half* oldest_k = full_k + ring_head * head_dim;
            __half* oldest_v = full_v + ring_head * head_dim;

            CodebookIndex* out_k = comp_k + comp_count;
            CodebookIndex* out_v = comp_v + comp_count;

            quantize_to_codebook(oldest_k, 1, head_dim, cb_k, out_k);
            quantize_to_codebook(oldest_v, 1, head_dim, cb_v, out_v);

            ring_head = (ring_head + 1) % age_threshold;
            --full_count;
            ++comp_count;
        }
    }

    // Bulk-append num_tokens tokens from device memory and compress entries
    // exceeding age_threshold. Tokens in kv_k/kv_v are stored row-major:
    // row i = token i, each row has head_dim fp16 values.
    void append_and_compress_device(
        const __half* kv_k,    // [num_tokens x head_dim] device
        const __half* kv_v,    // [num_tokens x head_dim] device
        int           num_tokens,
        const Codebook* cb_k,
        const Codebook* cb_v)
    {
        for (int t = 0; t < num_tokens; ++t) {
            // Insert new token into ring
            int slot = (ring_head + full_count) % age_threshold;
            cudaMemcpy(full_k + slot * head_dim,
                       kv_k + t * head_dim,
                       head_dim * sizeof(__half), cudaMemcpyDeviceToDevice);
            cudaMemcpy(full_v + slot * head_dim,
                       kv_v + t * head_dim,
                       head_dim * sizeof(__half), cudaMemcpyDeviceToDevice);

            if (full_count < age_threshold) {
                ++full_count;
            } else {
                // Compress the oldest entry before overwriting
                CodebookIndex* out_k = comp_k + comp_count;
                CodebookIndex* out_v = comp_v + comp_count;
                quantize_to_codebook(full_k + ring_head * head_dim, 1, head_dim, cb_k, out_k);
                quantize_to_codebook(full_v + ring_head * head_dim, 1, head_dim, cb_v, out_v);
                ring_head = (ring_head + 1) % age_threshold;
                ++comp_count;
            }
            ++seq_len;
        }
    }

    // Return a device pointer to full-precision K starting at ring offset pos.
    // pos is relative to ring_head (0 = oldest full-precision entry).
    __host__ __device__ const __half* full_k_at(int pos) const {
        int slot = (ring_head + pos) % age_threshold;
        return full_k + slot * head_dim;
    }

    __host__ __device__ const __half* full_v_at(int pos) const {
        int slot = (ring_head + pos) % age_threshold;
        return full_v + slot * head_dim;
    }

    // Return pointer to start of compressed K/V arrays (device memory).
    __host__ __device__ const CodebookIndex* comp_k_data() const { return comp_k; }
    __host__ __device__ const CodebookIndex* comp_v_data() const { return comp_v; }
};
