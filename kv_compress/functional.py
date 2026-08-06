"""
Low-level functional API for users who want direct control over codebook ops
without the CompressedKVCache abstraction.
"""

from __future__ import annotations
from typing import Tuple

import torch


def train_codebook(
    vectors: torch.Tensor,
    codebook_size: int = 256,
    num_iters: int = 20,
    seed: int = 42,
) -> Tuple[torch.Tensor, int]:
    """Train a K-means codebook from fp16 calibration vectors.

    Parameters
    ----------
    vectors : torch.Tensor
        [N, D] float16 CUDA tensor of calibration vectors.
    codebook_size : int
        Number of centroids (1..256).
    num_iters : int
        Number of Lloyd's algorithm iterations.
    seed : int
        RNG seed for centroid initialization.

    Returns
    -------
    centroids : torch.Tensor
        [codebook_size, D] float32 CPU tensor of trained centroids.
    codebook_ptr : int
        Raw device pointer to the allocated Codebook struct.
        Must be freed with free_codebook() when no longer needed.
    """
    from kv_compress._C import train_codebook as _train  # type: ignore

    if not vectors.is_cuda():
        raise ValueError("vectors must be a CUDA tensor")
    vectors = vectors.contiguous().to(torch.float16)
    centroids, ptr = _train(vectors, codebook_size, num_iters, seed)
    return centroids, ptr


def free_codebook(codebook_ptr: int) -> None:
    """Free device memory for a codebook allocated by train_codebook().

    Parameters
    ----------
    codebook_ptr : int
        Pointer returned by train_codebook().
    """
    from kv_compress._C import free_codebook as _free  # type: ignore
    _free(codebook_ptr)


def quantize(vectors: torch.Tensor, codebook_ptr: int) -> torch.Tensor:
    """Quantize fp16 vectors to uint8 codebook indices.

    Parameters
    ----------
    vectors : torch.Tensor
        [N, D] float16 CUDA tensor.
    codebook_ptr : int
        Device codebook pointer from train_codebook().

    Returns
    -------
    torch.Tensor
        [N] uint8 CUDA tensor of nearest-centroid indices.
    """
    from kv_compress._C import quantize as _quantize  # type: ignore

    if not vectors.is_cuda():
        raise ValueError("vectors must be a CUDA tensor")
    vectors = vectors.contiguous().to(torch.float16)
    return _quantize(vectors, codebook_ptr)


def mixed_attention(
    q: torch.Tensor,
    comp_k: torch.Tensor,
    comp_v: torch.Tensor,
    full_k: torch.Tensor,
    full_v: torch.Tensor,
    cb_k_ptr: int,
    cb_v_ptr: int,
) -> torch.Tensor:
    """Mixed-precision attention over a compressed + full-precision KV cache.

    The function runs a single online-softmax pass over the compressed region
    (tokens reconstructed from codebook) followed by the full-precision region.
    Token order is: compressed (oldest) then full-precision (newest).

    Parameters
    ----------
    q : torch.Tensor
        [num_queries, head_dim] float16 CUDA tensor.
    comp_k : torch.Tensor
        [comp_count] uint8 CUDA tensor of K codebook indices.
    comp_v : torch.Tensor
        [comp_count] uint8 CUDA tensor of V codebook indices.
    full_k : torch.Tensor
        [full_count, head_dim] float16 CUDA tensor, oldest token at row 0.
    full_v : torch.Tensor
        [full_count, head_dim] float16 CUDA tensor.
    cb_k_ptr : int
        Device codebook pointer for K (from train_codebook()).
    cb_v_ptr : int
        Device codebook pointer for V (from train_codebook()).

    Returns
    -------
    torch.Tensor
        [num_queries, head_dim] float16 CUDA tensor.
    """
    from kv_compress._C import attention_mixed as _attn  # type: ignore

    q       = q.contiguous().to(torch.float16)
    comp_k  = comp_k.contiguous()
    comp_v  = comp_v.contiguous()
    full_k  = full_k.contiguous().to(torch.float16)
    full_v  = full_v.contiguous().to(torch.float16)

    return _attn(q, comp_k, comp_v, full_k, full_v, cb_k_ptr, cb_v_ptr)
