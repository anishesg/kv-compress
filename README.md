# kv-compress

Online KV-cache compression via per-head learned codebooks with age-based mixed-precision attention.

## Problem

KV-cache memory is the dominant bottleneck for long-context LLM serving. For a 7B parameter model with 32 attention heads and head dimension 128, storing a 128K token context in float16 requires:

```
2 (K+V) * 32 heads * 128 dim * 128K tokens * 2 bytes = 2 GB per layer
```

This limits batch size, maximum context length, and throughput on GPU memory-constrained deployments.

## Existing Approaches and Their Limitations

**Uniform quantization (int4/int8):** Reduces memory by 2-4x but applies the same precision loss to all tokens regardless of importance. Recent tokens that dominate softmax mass get the same lossy treatment as old tokens that contribute negligibly.

**Eviction policies (H2O, StreamingLLM):** Discard tokens entirely beyond a window. This causes hard information loss: evicted tokens cannot contribute to attention even if they were relevant. Accuracy degrades significantly on tasks requiring long-range retrieval.

**Grouped Query Attention (GQA):** Reduces KV heads by sharing them across query heads. Requires model retraining, does not apply to deployed models, and cannot be tuned post-hoc to trade memory for quality.

## Our Approach

Three components compose the system:

### 1. Per-Head Learned Codebooks

Each attention head projects queries and keys into a distinct subspace. K and V vectors within a single head cluster tightly around a small set of prototypes. We train a codebook of 64-256 centroids per head using K-means on calibration data. A single `uint8` index then represents what would otherwise require `head_dim * 2` bytes, giving a compression ratio of 64-128x for compressed entries.

### 2. Age-Based Compression Policy

Recent tokens dominate softmax attention mass due to recency bias and causal masking. We maintain a `float16` full-precision region for the most recent `age_threshold` tokens, and compress older entries to codebook indices. This concentrates fidelity where it matters most and applies maximum compression to tokens with the smallest expected contribution.

The MixedKVCache structure stores:
- `float16` arrays for the recent region (size: `age_threshold * head_dim * 2`)
- `uint8` index arrays for the compressed region (size: `(seq_len - age_threshold) * 2`)

### 3. Mixed-Precision Attention Kernel

A single CUDA kernel computes correct attention scores over both regions in one pass with one online softmax accumulator. Reconstructing centroids on-the-fly during tile loading avoids materializing the full decompressed KV cache in global memory. The bandwidth cost is dominated by reading `uint8` indices rather than `float16` vectors, achieving near-proportional memory bandwidth reduction.

```
Compressed region bandwidth: 1 byte/token (index lookup)
Full-precision region bandwidth: head_dim * 2 bytes/token
```

At seq_len=128K with age_threshold=512 and head_dim=128:
- Full-precision KV: 128K * 128 * 2 * 2 = 64 MB per head
- Mixed KV: 512 * 128 * 2 * 2 + 127K * 2 = 0.5 MB full + 0.25 MB indices = 0.75 MB per head
- Compression ratio: ~85x

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requires CUDA 11.8+, a GPU with compute capability 8.0+ (Ampere or newer).

## Tests

```bash
./build/test_correctness
```

Runs mixed-precision attention against a full-precision reference across multiple configurations and reports cosine similarity and max absolute error.

## Benchmarks

```bash
./build/memory_analysis
```

Prints a table of memory usage and compression ratios across sequence lengths from 1K to 128K tokens.

## Directory Structure

```
src/
  codebook.cuh          - Codebook data structures and L2 distance kernel
  codebook_train.cu/h   - K-means training on GPU
  quantize.cu/h         - Nearest-centroid quantization kernel
  kv_cache.cuh          - MixedKVCache with age-based compression
  attention_ref.cu/h    - Full-precision reference attention
  attention_mixed.cu/h  - Mixed-precision attention with codebook reconstruction
tests/
  test_correctness.cu   - Correctness validation suite
benchmarks/
  memory_analysis.cu    - Memory footprint analysis tool
```
