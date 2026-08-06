"""
Correctness test: CompressedKVCache mixed attention vs torch.nn.functional.scaled_dot_product_attention.

The test fills the cache past the age_threshold so that some tokens are compressed,
then compares the mixed-attention output against full-precision sdpa.
"""

import math
import torch
import torch.nn.functional as F

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kv_compress import CompressedKVCache


def build_reference_attn(q, k_all, v_all):
    """Full-precision scaled dot-product attention via PyTorch."""
    # q: [1, D], k_all: [S, D], v_all: [S, D]
    q_3d = q.unsqueeze(0).float()           # [1, 1, D]
    k_3d = k_all.unsqueeze(0).float()       # [1, S, D]
    v_3d = v_all.unsqueeze(0).float()       # [1, S, D]
    out = F.scaled_dot_product_attention(q_3d, k_3d, v_3d)
    return out.squeeze(0).squeeze(0).half() # [D]


def test_mixed_vs_sdpa(
    head_dim: int = 64,
    codebook_size: int = 64,
    age_threshold: int = 32,
    num_tokens: int = 128,
    num_calib: int = 512,
    atol: float = 0.1,
    rtol: float = 0.1,
):
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Calibration data
    calib_k = torch.randn(num_calib, head_dim, dtype=torch.float16, device=device)
    calib_v = torch.randn(num_calib, head_dim, dtype=torch.float16, device=device)

    cache = CompressedKVCache(
        head_dim=head_dim,
        age_threshold=age_threshold,
        codebook_size=codebook_size,
        max_seq_len=num_tokens + age_threshold,
        device=device,
    )
    cache.train(calib_k, calib_v, num_iters=20)

    # Generate random K/V tokens and fill the cache
    keys   = torch.randn(num_tokens, head_dim, dtype=torch.float16, device=device)
    values = torch.randn(num_tokens, head_dim, dtype=torch.float16, device=device)
    cache.append_batch(keys, values)

    assert cache.comp_count > 0, "Expected some compressed tokens after filling past threshold"
    assert cache.full_count == age_threshold, f"Expected full_count={age_threshold}, got {cache.full_count}"

    # Query
    q = torch.randn(head_dim, dtype=torch.float16, device=device)

    # Mixed-precision attention output
    out_mixed = cache.attention(q)  # [D] fp16

    # Full-precision reference: all tokens at full precision
    out_ref = build_reference_attn(q, keys, values)  # [D] fp16

    # The mixed output will differ from the reference due to quantization;
    # we check the cosine similarity and relative error are within tolerance.
    cos_sim = F.cosine_similarity(
        out_mixed.float().unsqueeze(0),
        out_ref.float().unsqueeze(0),
    ).item()

    rel_err = (out_mixed.float() - out_ref.float()).norm() / (out_ref.float().norm() + 1e-8)

    print(f"  head_dim={head_dim}, codebook_size={codebook_size}, "
          f"age_threshold={age_threshold}, tokens={num_tokens}")
    print(f"  comp_count={cache.comp_count}, full_count={cache.full_count}")
    print(f"  cosine_similarity={cos_sim:.4f}, relative_error={rel_err:.4f}")

    assert cos_sim > 0.9, f"Cosine similarity too low: {cos_sim:.4f}"
    assert rel_err < atol, f"Relative error too high: {rel_err:.4f} (atol={atol})"

    print("  PASSED")


def test_all_full_precision(head_dim: int = 64, codebook_size: int = 64):
    """When no tokens have been compressed (seq_len <= age_threshold), outputs match exactly."""
    device = torch.device("cuda")
    torch.manual_seed(1)

    age_threshold = 64
    num_tokens = 32  # fewer than age_threshold, so nothing is compressed

    calib_k = torch.randn(256, head_dim, dtype=torch.float16, device=device)
    calib_v = torch.randn(256, head_dim, dtype=torch.float16, device=device)

    cache = CompressedKVCache(
        head_dim=head_dim,
        age_threshold=age_threshold,
        codebook_size=codebook_size,
        max_seq_len=4096,
        device=device,
    )
    cache.train(calib_k, calib_v)

    keys   = torch.randn(num_tokens, head_dim, dtype=torch.float16, device=device)
    values = torch.randn(num_tokens, head_dim, dtype=torch.float16, device=device)
    cache.append_batch(keys, values)

    assert cache.comp_count == 0, "No tokens should be compressed"

    q = torch.randn(head_dim, dtype=torch.float16, device=device)
    out_mixed = cache.attention(q).float()
    out_ref   = build_reference_attn(q, keys, values).float()

    # With no compression, outputs should match to fp16 precision.
    max_diff = (out_mixed - out_ref).abs().max().item()
    print(f"  all-full-precision max_diff={max_diff:.6f}")
    assert max_diff < 1e-2, f"Outputs diverged unexpectedly: max_diff={max_diff}"
    print("  PASSED")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        raise SystemExit(0)

    print("test_all_full_precision:")
    test_all_full_precision()

    print("test_mixed_vs_sdpa (head_dim=64, codebook=64, age=32, tokens=128):")
    test_mixed_vs_sdpa(head_dim=64, codebook_size=64, age_threshold=32, num_tokens=128)

    print("test_mixed_vs_sdpa (head_dim=128, codebook=256, age=64, tokens=256):")
    test_mixed_vs_sdpa(
        head_dim=128, codebook_size=256, age_threshold=64, num_tokens=256, atol=0.15
    )

    print("\nAll tests passed.")
