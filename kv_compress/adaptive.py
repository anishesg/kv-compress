"""
Adaptive age threshold controller.

Monitors GPU memory pressure and adjusts the age_threshold of a
CompressedKVCache: when free memory drops below a low-water mark the threshold
is tightened (more tokens compressed); when headroom is comfortable it can be
relaxed up to a user-specified maximum.
"""

from __future__ import annotations

import torch


class AdaptiveThresholdController:
    """Adjusts CompressedKVCache.age_threshold based on GPU memory pressure.

    The controller is designed to be called after every N token appends.
    It reads the current free/total GPU memory ratio and clamps the threshold
    between min_threshold and max_threshold.

    Parameters
    ----------
    cache : CompressedKVCache
        The cache whose age_threshold will be adjusted.
    min_threshold : int
        Minimum allowed value for age_threshold (hard floor; must be >= 1).
    max_threshold : int
        Maximum allowed value for age_threshold (hard ceiling).
    low_water_frac : float
        Free-memory fraction below which the threshold is reduced.
        E.g. 0.15 means "tighten compression when less than 15% of GPU memory is free."
    high_water_frac : float
        Free-memory fraction above which the threshold may be relaxed.
        E.g. 0.40 means "loosen compression when more than 40% is free."
    step : int
        Amount by which threshold is adjusted per controller call.
    device : torch.device or None
        GPU device to query; defaults to the device of `cache`.
    """

    def __init__(
        self,
        cache,
        min_threshold: int = 16,
        max_threshold: int = 512,
        low_water_frac: float = 0.15,
        high_water_frac: float = 0.40,
        step: int = 16,
        device: torch.device | None = None,
    ) -> None:
        if min_threshold < 1:
            raise ValueError("min_threshold must be >= 1")
        if max_threshold < min_threshold:
            raise ValueError("max_threshold must be >= min_threshold")
        if not (0.0 < low_water_frac < high_water_frac < 1.0):
            raise ValueError("require 0 < low_water_frac < high_water_frac < 1")

        self.cache = cache
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.low_water_frac = low_water_frac
        self.high_water_frac = high_water_frac
        self.step = step
        self.device = device or cache.device

    def memory_free_fraction(self) -> float:
        """Return the fraction of total GPU memory that is currently free."""
        dev_idx = self.device.index if self.device.index is not None else 0
        free, total = torch.cuda.mem_get_info(dev_idx)
        return free / total if total > 0 else 1.0

    def update(self) -> int:
        """Check memory pressure and adjust cache.age_threshold if needed.

        Returns the new (or unchanged) age_threshold.
        """
        free_frac = self.memory_free_fraction()
        current   = self.cache.age_threshold

        if free_frac < self.low_water_frac:
            # Memory is tight; compress more aggressively.
            new_threshold = max(self.min_threshold, current - self.step)
        elif free_frac > self.high_water_frac:
            # Headroom exists; allow more tokens at full precision.
            new_threshold = min(self.max_threshold, current + self.step)
        else:
            new_threshold = current

        if new_threshold != current:
            self._apply_threshold(new_threshold)

        return self.cache.age_threshold

    def _apply_threshold(self, new_threshold: int) -> None:
        """Apply a new age_threshold to the cache.

        When tightening (new < old): tokens in the full-precision ring that
        exceed the new threshold are immediately compressed.
        When loosening (new > old): the full-precision ring capacity grows;
        no data movement is needed until new tokens are appended.

        The ring buffer tensors are resized to match the new threshold.
        """
        cache = self.cache
        old_threshold = cache.age_threshold

        if new_threshold == old_threshold:
            return

        # Materialize full-precision data in chronological order.
        full_k, full_v = cache._ordered_full()  # [full_count, D]

        if new_threshold < old_threshold:
            # Compress the oldest (full_count - new_threshold) entries.
            overflow = cache._full_count - new_threshold
            if overflow > 0 and cache.is_trained():
                from kv_compress._C import quantize  # type: ignore
                for t in range(overflow):
                    k_vec = full_k[t].unsqueeze(0)
                    v_vec = full_v[t].unsqueeze(0)
                    idx_k = quantize(k_vec, cache._cb_k_ptr)
                    idx_v = quantize(v_vec, cache._cb_v_ptr)
                    # Extend compressed arrays if needed
                    if cache._comp_count >= cache._comp_k.size(0):
                        extra = max(256, cache._comp_k.size(0))
                        cache._comp_k = torch.cat(
                            [cache._comp_k,
                             torch.zeros(extra, dtype=torch.uint8, device=cache.device)]
                        )
                        cache._comp_v = torch.cat(
                            [cache._comp_v,
                             torch.zeros(extra, dtype=torch.uint8, device=cache.device)]
                        )
                    cache._comp_k[cache._comp_count] = idx_k[0]
                    cache._comp_v[cache._comp_count] = idx_v[0]
                    cache._comp_count += 1
                full_k = full_k[overflow:]
                full_v = full_v[overflow:]

        # Rebuild the ring buffer at the new capacity.
        keep = min(full_k.size(0), new_threshold)
        new_ring_k = torch.zeros(
            new_threshold, cache.head_dim, dtype=torch.float16, device=cache.device
        )
        new_ring_v = torch.zeros(
            new_threshold, cache.head_dim, dtype=torch.float16, device=cache.device
        )
        if keep > 0:
            new_ring_k[:keep] = full_k[-keep:]
            new_ring_v[:keep] = full_v[-keep:]

        cache._full_k     = new_ring_k
        cache._full_v     = new_ring_v
        cache._ring_head  = 0
        cache._full_count = keep
        cache.age_threshold = new_threshold

    def status(self) -> dict:
        """Return a snapshot of current memory stats and threshold."""
        return {
            "free_frac":     self.memory_free_fraction(),
            "age_threshold": self.cache.age_threshold,
            "comp_count":    self.cache.comp_count,
            "full_count":    self.cache.full_count,
        }
