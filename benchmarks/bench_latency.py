"""
GPU latency benchmark: mixed-precision attention vs full-precision naive attention.

Measures wall-clock time via CUDA events across sequence lengths 1K to 64K.
Both kernels process a single query against the entire KV cache (decode step).
"""

import math
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def cuda_timer(fn, warmup: int = 10, repeats: int = 50):
    """Run fn() warmup+repeats times; return mean latency in milliseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    total_ms = 0.0
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)

    return total_ms / repeats


def bench_mixed(
    seq_len: int,
    head_dim: int = 128,
    codebook_size: int = 256,
    age_threshold: int = 128,
):
    """Benchmark mixed-precision attention for a given sequence length."""
    from kv_compress._C import train_codebook, attention_mixed  # type: ignore

    device = torch.device("cuda")
    torch.manual_seed(0)

    # Train codebooks on random calibration data
    calib = torch.randn(max(1024, seq_len // 4), head_dim,
                        dtype=torch.float16, device=device)
    _, cb_k = train_codebook(calib, codebook_size, 10, 42)
    _, cb_v = train_codebook(calib, codebook_size, 10, 43)

    comp_count = max(0, seq_len - age_threshold)
    full_count = min(seq_len, age_threshold)

    comp_k = torch.randint(0, codebook_size, (comp_count,),
                           dtype=torch.uint8, device=device)
    comp_v = torch.randint(0, codebook_size, (comp_count,),
                           dtype=torch.uint8, device=device)
    full_k = torch.randn(full_count, head_dim, dtype=torch.float16, device=device)
    full_v = torch.randn(full_count, head_dim, dtype=torch.float16, device=device)
    q = torch.randn(1, head_dim, dtype=torch.float16, device=device)
    out = torch.empty(1, head_dim, dtype=torch.float16, device=device)

    def run():
        attention_mixed(q, comp_k, comp_v, full_k, full_v, cb_k, cb_v)

    from kv_compress._C import free_codebook  # type: ignore
    latency = cuda_timer(run)
    free_codebook(cb_k)
    free_codebook(cb_v)
    return latency


def bench_full(seq_len: int, head_dim: int = 128):
    """Benchmark full-precision naive attention (attention_ref) for comparison."""
    # Use torch scaled_dot_product_attention as the reference baseline
    device = torch.device("cuda")
    torch.manual_seed(0)

    q = torch.randn(1, 1, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(1, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(1, seq_len, head_dim, dtype=torch.float16, device=device)

    def run():
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    return cuda_timer(run)


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    head_dim    = 128
    codebook_size = 256
    age_threshold = 128

    print(f"{'seq_len':>8}  {'full_ms':>10}  {'mixed_ms':>10}  {'speedup':>8}")
    print("-" * 45)

    for seq_len in seq_lengths:
        t_full  = bench_full(seq_len, head_dim)
        t_mixed = bench_mixed(seq_len, head_dim, codebook_size, age_threshold)
        speedup = t_full / t_mixed if t_mixed > 0 else float("inf")
        print(f"{seq_len:>8}  {t_full:>10.3f}  {t_mixed:>10.3f}  {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
