"""
Autoregressive generation demo with online KV-cache compression.

Simulates a single-head decode loop for 4096 tokens. At each step:
  1. A random query is generated (stands in for the model's last-token hidden state).
  2. Mixed-precision attention is run over the full history.
  3. The new token's K/V vectors are appended and the oldest token may be compressed.

Prints cache state, memory usage, and output quality metrics every 512 tokens.
No real language model is needed; the demo validates the infrastructure end-to-end.
"""

import sys
import os
import math

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kv_compress import CompressedKVCache
from kv_compress.adaptive import AdaptiveThresholdController


HEAD_DIM      = 128
CODEBOOK_SIZE = 256
AGE_THRESHOLD = 256    # keep most recent 256 tokens at full precision
MAX_SEQ_LEN   = 5120   # enough for 4096 generated + some initial context
NUM_TOKENS    = 4096
PRINT_EVERY   = 512

CALIB_SIZE    = 2048   # calibration vectors for codebook training


def cosine_sim_fp(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item()


def build_full_attn_output(
    q: torch.Tensor,
    k_history: torch.Tensor,
    v_history: torch.Tensor,
) -> torch.Tensor:
    """Full-precision reference attention over the entire history."""
    q3 = q.float().unsqueeze(0).unsqueeze(0)    # [1, 1, D]
    k3 = k_history.float().unsqueeze(0)          # [1, S, D]
    v3 = v_history.float().unsqueeze(0)          # [1, S, D]
    return F.scaled_dot_product_attention(q3, k3, v3).squeeze(0).squeeze(0).half()


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    device = torch.device("cuda")
    torch.manual_seed(42)
    print(f"Running {NUM_TOKENS}-token autoregressive demo on {device}")
    print(f"  head_dim={HEAD_DIM}, codebook_size={CODEBOOK_SIZE}, "
          f"age_threshold={AGE_THRESHOLD}")
    print()

    # ------------------------------------------------------------------ #
    # 1. Train codebooks from calibration data.                            #
    # ------------------------------------------------------------------ #
    calib_k = torch.randn(CALIB_SIZE, HEAD_DIM, dtype=torch.float16, device=device)
    calib_v = torch.randn(CALIB_SIZE, HEAD_DIM, dtype=torch.float16, device=device)

    cache = CompressedKVCache(
        head_dim=HEAD_DIM,
        age_threshold=AGE_THRESHOLD,
        codebook_size=CODEBOOK_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        device=device,
    )
    cache.train(calib_k, calib_v, num_iters=20)

    # Set up adaptive controller (gentle: step=32, memory bounds generous).
    controller = AdaptiveThresholdController(
        cache,
        min_threshold=64,
        max_threshold=512,
        low_water_frac=0.05,
        high_water_frac=0.70,
        step=32,
        device=device,
    )

    # ------------------------------------------------------------------ #
    # 2. Simulate generation.                                              #
    # ------------------------------------------------------------------ #
    # Keep a full-precision reference history to measure output quality.
    k_history = torch.zeros(NUM_TOKENS, HEAD_DIM, dtype=torch.float16, device=device)
    v_history = torch.zeros(NUM_TOKENS, HEAD_DIM, dtype=torch.float16, device=device)

    sum_cos_sim   = 0.0
    sum_rel_err   = 0.0
    report_count  = 0

    for step in range(NUM_TOKENS):
        # New K/V vectors for this token (stand-in for model projections).
        k_new = torch.randn(HEAD_DIM, dtype=torch.float16, device=device)
        v_new = torch.randn(HEAD_DIM, dtype=torch.float16, device=device)

        # Query for this decode step.
        q = torch.randn(HEAD_DIM, dtype=torch.float16, device=device)

        # Append to cache BEFORE running attention (causal: all prior tokens visible).
        k_history[step] = k_new
        v_history[step] = v_new
        cache.append(k_new, v_new)

        # Run mixed-precision attention.
        out_mixed = cache.attention(q)

        # Periodically compare against full-precision reference.
        if (step + 1) % PRINT_EVERY == 0:
            seq_so_far = step + 1
            out_ref = build_full_attn_output(q, k_history[:seq_so_far], v_history[:seq_so_far])

            cos_sim = cosine_sim_fp(out_mixed, out_ref)
            rel_err = (out_mixed.float() - out_ref.float()).norm().item() / \
                      (out_ref.float().norm().item() + 1e-8)
            sum_cos_sim  += cos_sim
            sum_rel_err  += rel_err
            report_count += 1

            mem_mb = cache.memory_bytes() / 1e6
            dev_idx = device.index if device.index is not None else 0
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
            free_frac = free_bytes / total_bytes

            # Update adaptive threshold.
            new_thresh = controller.update()

            print(
                f"  step {seq_so_far:>5} | "
                f"comp={cache.comp_count:>5} full={cache.full_count:>4} | "
                f"cos_sim={cos_sim:.4f} rel_err={rel_err:.4f} | "
                f"cache_mem={mem_mb:.2f}MB | "
                f"gpu_free={free_frac*100:.1f}% | "
                f"threshold={new_thresh}"
            )

    print()
    if report_count > 0:
        print(f"Average cosine similarity over {report_count} checkpoints: "
              f"{sum_cos_sim/report_count:.4f}")
        print(f"Average relative error: {sum_rel_err/report_count:.4f}")


if __name__ == "__main__":
    main()
