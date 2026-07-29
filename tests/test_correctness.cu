#include "codebook.cuh"
#include "codebook_train.cuh"
#include "quantize.cuh"
#include "kv_cache.cuh"
#include "attention_ref.cuh"
#include "attention_mixed.cuh"

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

// Generate random fp16 data on host, copy to device
static __half* make_device_fp16(int n, float lo, float hi, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<__half> h(n);
    for (int i = 0; i < n; ++i) h[i] = __float2half(dist(rng));
    __half* d;
    cudaMalloc(&d, n * sizeof(__half));
    cudaMemcpy(d, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
    return d;
}

// Copy device fp16 array to host float32 vector
static std::vector<float> to_host_float(const __half* d_ptr, int n) {
    std::vector<__half> h(n);
    cudaMemcpy(h.data(), d_ptr, n * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> f(n);
    for (int i = 0; i < n; ++i) f[i] = __half2float(h[i]);
    return f;
}

static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na == 0 || nb == 0) return 0.f;
    return static_cast<float>(dot / (sqrt(na) * sqrt(nb)));
}

static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0;
    for (size_t i = 0; i < a.size(); ++i) err = fmaxf(err, fabsf(a[i] - b[i]));
    return err;
}

struct Config {
    int seq_len;
    int head_dim;
    int codebook_size;
    int age_threshold;
};

static void run_config(const Config& cfg, std::mt19937& rng) {
    const int seq_len       = cfg.seq_len;
    const int head_dim      = cfg.head_dim;
    const int codebook_size = cfg.codebook_size;
    const int age_threshold = cfg.age_threshold;
    const int num_queries   = 1;

    // -----------------------------------------------------------------------
    // Generate KV data
    // -----------------------------------------------------------------------
    __half* d_K = make_device_fp16(seq_len * head_dim, -1.f, 1.f, rng);
    __half* d_V = make_device_fp16(seq_len * head_dim, -1.f, 1.f, rng);
    __half* d_Q = make_device_fp16(num_queries * head_dim, -1.f, 1.f, rng);

    // -----------------------------------------------------------------------
    // Reference attention over full-precision KV
    // -----------------------------------------------------------------------
    __half* d_ref_out;
    cudaMalloc(&d_ref_out, num_queries * head_dim * sizeof(__half));
    attention_ref(d_Q, d_K, d_V, d_ref_out, num_queries, seq_len, head_dim);

    // -----------------------------------------------------------------------
    // Train codebooks on the full KV data
    // -----------------------------------------------------------------------
    Codebook* d_cb_k;
    Codebook* d_cb_v;
    cudaMalloc(&d_cb_k, sizeof(Codebook));
    cudaMalloc(&d_cb_v, sizeof(Codebook));

    int train_iters = 20;
    train_codebook(d_K, seq_len, head_dim, codebook_size, train_iters, d_cb_k);
    train_codebook(d_V, seq_len, head_dim, codebook_size, train_iters, d_cb_v);

    // -----------------------------------------------------------------------
    // Build MixedKVCache by appending all tokens
    // -----------------------------------------------------------------------
    MixedKVCache cache = MixedKVCache::create(head_dim, age_threshold, seq_len + 64);
    cache.append_and_compress_device(d_K, d_V, seq_len, d_cb_k, d_cb_v);
    cudaDeviceSynchronize();

    int comp_count = cache.comp_count;
    int full_count = cache.full_count;

    // The full-precision ring may not be contiguous (ring buffer).
    // Linearize it into a fresh contiguous buffer for the mixed kernel.
    __half* d_full_k_lin = nullptr;
    __half* d_full_v_lin = nullptr;
    if (full_count > 0) {
        cudaMalloc(&d_full_k_lin, full_count * head_dim * sizeof(__half));
        cudaMalloc(&d_full_v_lin, full_count * head_dim * sizeof(__half));
        for (int i = 0; i < full_count; ++i) {
            cudaMemcpy(d_full_k_lin + i * head_dim,
                       cache.full_k_at(i),
                       head_dim * sizeof(__half), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_full_v_lin + i * head_dim,
                       cache.full_v_at(i),
                       head_dim * sizeof(__half), cudaMemcpyDeviceToDevice);
        }
    }

    // -----------------------------------------------------------------------
    // Mixed-precision attention
    // -----------------------------------------------------------------------
    __half* d_mixed_out;
    cudaMalloc(&d_mixed_out, num_queries * head_dim * sizeof(__half));
    attention_mixed(
        d_Q,
        cache.comp_k_data(), cache.comp_v_data(),
        d_full_k_lin, d_full_v_lin,
        d_cb_k, d_cb_v,
        d_mixed_out,
        num_queries, comp_count, full_count, head_dim);

    cudaDeviceSynchronize();

    // -----------------------------------------------------------------------
    // Compare outputs
    // -----------------------------------------------------------------------
    auto ref_h   = to_host_float(d_ref_out,   num_queries * head_dim);
    auto mixed_h = to_host_float(d_mixed_out, num_queries * head_dim);

    float cos_sim = cosine_similarity(ref_h, mixed_h);
    float max_err = max_abs_error(ref_h, mixed_h);

    printf("seq=%5d  head_dim=%3d  cb=%3d  age=%4d  comp=%5d  full=%4d  "
           "cos=%.6f  max_err=%.6f\n",
           seq_len, head_dim, codebook_size, age_threshold,
           comp_count, full_count, cos_sim, max_err);

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_Q);
    cudaFree(d_ref_out);
    cudaFree(d_cb_k);
    cudaFree(d_cb_v);
    cudaFree(d_mixed_out);
    if (d_full_k_lin) cudaFree(d_full_k_lin);
    if (d_full_v_lin) cudaFree(d_full_v_lin);
    cache.free_device_memory();
}

int main() {
    printf("Correctness test: mixed-precision vs full-precision attention\n");
    printf("%-7s  %-8s  %-4s  %-5s  %-7s  %-5s  %-10s  %-10s\n",
           "seq", "head_dim", "cb", "age", "comp", "full", "cos_sim", "max_err");
    printf("%s\n", std::string(80, '-').c_str());

    std::mt19937 rng(12345);

    // Configurations from the plan
    for (int seq_len : {512, 2048, 8192}) {
        for (int head_dim : {64, 128}) {
            for (int codebook_size : {64, 256}) {
                for (int age_threshold : {128, 512}) {
                    // Skip configs where age_threshold >= seq_len
                    // (no compressed entries, trivially identical to reference)
                    if (age_threshold >= seq_len) continue;
                    Config cfg{seq_len, head_dim, codebook_size, age_threshold};
                    run_config(cfg, rng);
                }
            }
        }
    }

    printf("\nDone. cos_sim >= 0.99 indicates good codebook approximation.\n");
    return 0;
}
