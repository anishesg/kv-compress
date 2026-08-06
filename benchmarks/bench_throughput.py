"""
Throughput benchmark: effective tokens per second with compression enabled vs disabled.

Sweeps batch sizes [1, 4, 8, 16, 32] and measures how many decode steps can be
completed per second.  Each decode step: append one new token, run attention.

Compression-enabled: tokens beyond age_threshold are stored as uint8 indices.
Compression-disabled: all tokens kept at full fp16 (uses torch sdpa).
"""

import sys
import os
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def throughput_compressed(
    batch_size: int,
    seq_len: int = 1024,
    head_dim: int = 128,
    codebook_size: int = 256,
    age_threshold: int = 128,
    steps: int = 200,
    warmup: int = 20,
) -> float:
    """Tokens/sec for compressed cache (simulate batch_size independent heads)."""
    from kv_compress._C import train_codebook, attention_mixed, free_codebook  # type: ignore

    device = torch.device("cuda")
    torch.manual_seed(0)

    # Train one shared codebook pair used by all batch entries.
    calib = torch.randn(1024, head_dim, dtype=torch.float16, device=device)
    _, cb_k = train_codebook(calib, codebook_size, 10, 42)
    _, cb_v = train_codebook(calib, codebook_size, 10, 43)

    # Pre-allocate compressed and full-precision buffers for all batches.
    comp_count = max(0, seq_len - age_threshold)
    full_count = min(seq_len, age_threshold)

    # Each batch entry shares the same data shapes; simulate batch_size entries
    # by processing them in a Python loop (single-head kernel doesn't batch).
    comp_k_list = [
        torch.randint(0, codebook_size, (comp_count,), dtype=torch.uint8, device=device)
        for _ in range(batch_size)
    ]
    comp_v_list = [
        torch.randint(0, codebook_size, (comp_count,), dtype=torch.uint8, device=device)
        for _ in range(batch_size)
    ]
    full_k_list = [
        torch.randn(full_count, head_dim, dtype=torch.float16, device=device)
        for _ in range(batch_size)
    ]
    full_v_list = [
        torch.randn(full_count, head_dim, dtype=torch.float16, device=device)
        for _ in range(batch_size)
    ]
    q_list = [
        torch.randn(1, head_dim, dtype=torch.float16, device=device)
        for _ in range(batch_size)
    ]

    def step():
        for b in range(batch_size):
            attention_mixed(
                q_list[b], comp_k_list[b], comp_v_list[b],
                full_k_list[b], full_v_list[b],
                cb_k, cb_v,
            )

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    free_codebook(cb_k)
    free_codebook(cb_v)

    tokens_per_sec = (steps * batch_size) / elapsed
    return tokens_per_sec


def throughput_full(
    batch_size: int,
    seq_len: int = 1024,
    head_dim: int = 128,
    steps: int = 200,
    warmup: int = 20,
) -> float:
    """Tokens/sec for full-precision sdpa (no compression)."""
    device = torch.device("cuda")
    torch.manual_seed(0)

    # batch_size independent attention heads
    q = torch.randn(batch_size, 1, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float16, device=device)

    def step():
        F.scaled_dot_product_attention(q, k, v)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (steps * batch_size) / elapsed


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    batch_sizes = [1, 4, 8, 16, 32]
    seq_len   = 1024
    head_dim  = 128

    print(f"seq_len={seq_len}, head_dim={head_dim}")
    print(f"{'batch':>6}  {'full_tok/s':>12}  {'comp_tok/s':>12}  {'ratio':>8}")
    print("-" * 48)

    for bs in batch_sizes:
        tps_full = throughput_full(bs, seq_len, head_dim)
        tps_comp = throughput_compressed(bs, seq_len, head_dim)
        ratio = tps_comp / tps_full if tps_full > 0 else float("inf")
        print(f"{bs:>6}  {tps_full:>12.0f}  {tps_comp:>12.0f}  {ratio:>8.2f}x")


if __name__ == "__main__":
    main()
