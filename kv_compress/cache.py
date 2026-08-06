"""
CompressedKVCache: manages the full KV-cache lifecycle for a single attention head.

Tokens younger than age_threshold are stored at full fp16 precision.
Older tokens are compressed to uint8 codebook indices.  The codebook is trained
once on a user-supplied calibration batch and reused for the lifetime of the cache.
"""

from __future__ import annotations
from typing import Optional

import torch


class CompressedKVCache:
    """KV cache with online age-based compression for one attention head.

    Parameters
    ----------
    head_dim : int
        Dimension of each K/V vector.
    age_threshold : int
        Number of the most recent tokens to keep at full fp16 precision.
        Tokens older than this are compressed to codebook indices.
    codebook_size : int
        Number of centroids per codebook (K and V use separate codebooks).
    max_seq_len : int
        Maximum total sequence length the cache will hold.
    device : torch.device or str
        CUDA device the cache lives on.
    """

    def __init__(
        self,
        head_dim: int,
        age_threshold: int,
        codebook_size: int = 256,
        max_seq_len: int = 32768,
        device: torch.device | str = "cuda",
    ) -> None:
        self.head_dim = head_dim
        self.age_threshold = age_threshold
        self.codebook_size = codebook_size
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)

        # Codebook device pointers; None until train() is called.
        self._cb_k_ptr: Optional[int] = None
        self._cb_v_ptr: Optional[int] = None

        # Full-precision ring buffer: shape [age_threshold, head_dim] fp16.
        self._full_k = torch.zeros(
            age_threshold, head_dim, dtype=torch.float16, device=self.device
        )
        self._full_v = torch.zeros(
            age_threshold, head_dim, dtype=torch.float16, device=self.device
        )
        self._ring_head = 0    # index of the oldest entry in the ring
        self._full_count = 0   # number of valid entries in the ring

        # Compressed region: grows as tokens age out of the ring.
        comp_capacity = max_seq_len - age_threshold
        self._comp_k = torch.zeros(
            comp_capacity, dtype=torch.uint8, device=self.device
        )
        self._comp_v = torch.zeros(
            comp_capacity, dtype=torch.uint8, device=self.device
        )
        self._comp_count = 0
        self.seq_len = 0

    # ------------------------------------------------------------------
    # Codebook training
    # ------------------------------------------------------------------

    def train(
        self,
        calib_k: torch.Tensor,
        calib_v: torch.Tensor,
        num_iters: int = 20,
        seed: int = 42,
    ) -> None:
        """Train K and V codebooks from calibration data.

        Parameters
        ----------
        calib_k : torch.Tensor
            [N, head_dim] fp16 CUDA tensor of calibration K vectors.
        calib_v : torch.Tensor
            [N, head_dim] fp16 CUDA tensor of calibration V vectors.
        num_iters : int
            Number of Lloyd/K-means iterations.
        seed : int
            RNG seed for centroid initialization.
        """
        from kv_compress._C import train_codebook, free_codebook  # type: ignore

        if calib_k.device != self.device:
            calib_k = calib_k.to(self.device)
        if calib_v.device != self.device:
            calib_v = calib_v.to(self.device)

        calib_k = calib_k.contiguous().to(torch.float16)
        calib_v = calib_v.contiguous().to(torch.float16)

        if self._cb_k_ptr is not None:
            free_codebook(self._cb_k_ptr)
        if self._cb_v_ptr is not None:
            free_codebook(self._cb_v_ptr)

        _, self._cb_k_ptr = train_codebook(calib_k, self.codebook_size, num_iters, seed)
        _, self._cb_v_ptr = train_codebook(calib_v, self.codebook_size, num_iters, seed + 1)

    def is_trained(self) -> bool:
        return self._cb_k_ptr is not None and self._cb_v_ptr is not None

    # ------------------------------------------------------------------
    # Appending tokens
    # ------------------------------------------------------------------

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append a single token's K and V vectors and compress if needed.

        Parameters
        ----------
        k : torch.Tensor
            [head_dim] or [1, head_dim] fp16 tensor.
        v : torch.Tensor
            [head_dim] or [1, head_dim] fp16 tensor.
        """
        if k.dim() == 2:
            k = k.squeeze(0)
        if v.dim() == 2:
            v = v.squeeze(0)

        k = k.contiguous().to(torch.float16).to(self.device)
        v = v.contiguous().to(torch.float16).to(self.device)

        write_slot = (self._ring_head + self._full_count) % self.age_threshold

        if self._full_count == self.age_threshold:
            # Ring is full; compress the oldest entry before overwriting it.
            self._compress_oldest()

        self._full_k[write_slot] = k
        self._full_v[write_slot] = v
        self._full_count = min(self._full_count + 1, self.age_threshold)
        self.seq_len += 1

    def append_batch(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append multiple tokens at once.

        Parameters
        ----------
        k : torch.Tensor
            [T, head_dim] fp16 tensor.
        v : torch.Tensor
            [T, head_dim] fp16 tensor.
        """
        k = k.contiguous().to(torch.float16).to(self.device)
        v = v.contiguous().to(torch.float16).to(self.device)
        T = k.size(0)
        for t in range(T):
            self.append(k[t], v[t])

    def _compress_oldest(self) -> None:
        """Move the oldest full-precision entry into the compressed arrays."""
        if not self.is_trained():
            raise RuntimeError("Call train() before appending tokens past age_threshold")

        from kv_compress._C import quantize  # type: ignore

        oldest_k = self._full_k[self._ring_head].unsqueeze(0)  # [1, D]
        oldest_v = self._full_v[self._ring_head].unsqueeze(0)

        idx_k = quantize(oldest_k, self._cb_k_ptr)  # [1] uint8
        idx_v = quantize(oldest_v, self._cb_v_ptr)

        self._comp_k[self._comp_count] = idx_k[0]
        self._comp_v[self._comp_count] = idx_v[0]
        self._comp_count += 1
        self._ring_head = (self._ring_head + 1) % self.age_threshold
        self._full_count -= 1

    # ------------------------------------------------------------------
    # Attention computation
    # ------------------------------------------------------------------

    def attention(self, q: torch.Tensor) -> torch.Tensor:
        """Run mixed-precision attention for one or more query vectors.

        Parameters
        ----------
        q : torch.Tensor
            [num_queries, head_dim] or [head_dim] fp16 CUDA tensor.

        Returns
        -------
        torch.Tensor
            [num_queries, head_dim] fp16 CUDA tensor.
        """
        if not self.is_trained():
            raise RuntimeError("Call train() before running attention")

        squeeze = q.dim() == 1
        if squeeze:
            q = q.unsqueeze(0)

        q = q.contiguous().to(torch.float16).to(self.device)

        from kv_compress._C import attention_mixed  # type: ignore

        # Build a contiguous full-precision K/V view ordered oldest-first.
        full_k, full_v = self._ordered_full()

        comp_k = self._comp_k[: self._comp_count].contiguous()
        comp_v = self._comp_v[: self._comp_count].contiguous()

        out = attention_mixed(
            q,
            comp_k,
            comp_v,
            full_k,
            full_v,
            self._cb_k_ptr,
            self._cb_v_ptr,
        )
        return out.squeeze(0) if squeeze else out

    def _ordered_full(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full-precision K/V in chronological order (oldest first)."""
        if self._full_count == 0:
            empty = torch.zeros(0, self.head_dim, dtype=torch.float16, device=self.device)
            return empty, empty

        if self._full_count < self.age_threshold:
            # Ring hasn't wrapped; data starts at index 0.
            k = self._full_k[: self._full_count].contiguous()
            v = self._full_v[: self._full_count].contiguous()
        else:
            # Ring has wrapped; oldest entry is at ring_head.
            tail = self._full_k[self._ring_head:]
            head = self._full_k[: self._ring_head]
            k = torch.cat([tail, head], dim=0).contiguous()
            tail_v = self._full_v[self._ring_head:]
            head_v = self._full_v[: self._ring_head]
            v = torch.cat([tail_v, head_v], dim=0).contiguous()
        return k, v

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def comp_count(self) -> int:
        return self._comp_count

    @property
    def full_count(self) -> int:
        return self._full_count

    def memory_bytes(self) -> int:
        """Approximate device memory usage in bytes."""
        full_bytes = 2 * self.age_threshold * self.head_dim * 2  # fp16
        comp_bytes = 2 * self._comp_count                         # uint8
        return full_bytes + comp_bytes

    def __del__(self) -> None:
        try:
            from kv_compress._C import free_codebook  # type: ignore
            if self._cb_k_ptr is not None:
                free_codebook(self._cb_k_ptr)
            if self._cb_v_ptr is not None:
                free_codebook(self._cb_v_ptr)
        except Exception:
            pass
