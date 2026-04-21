"""KV-cache utilities for DFlash inference.

Provides cache structures used during autoregressive decoding,
including a sliding-window variant for long-context generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class CacheEntry:
    """Stores key/value tensors for a single layer."""

    keys: torch.Tensor    # (batch, heads, seq, head_dim)
    values: torch.Tensor  # (batch, heads, seq, head_dim)

    @property
    def seq_len(self) -> int:
        return self.keys.shape[2]


class DynamicCache:
    """Grows unboundedly; suitable for short-to-medium generation."""

    def __init__(self) -> None:
        self._entries: list[Optional[CacheEntry]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def update(
        self,
        layer_idx: int,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return the full accumulated tensors."""
        # Extend list if needed
        while len(self._entries) <= layer_idx:
            self._entries.append(None)

        entry = self._entries[layer_idx]
        if entry is None:
            self._entries[layer_idx] = CacheEntry(new_keys, new_values)
        else:
            cat_k = torch.cat([entry.keys, new_keys], dim=2)
            cat_v = torch.cat([entry.values, new_values], dim=2)
            self._entries[layer_idx] = CacheEntry(cat_k, cat_v)

        e = self._entries[layer_idx]
        return e.keys, e.values

    def get(self, layer_idx: int) -> Optional[CacheEntry]:
        if layer_idx < len(self._entries):
            return self._entries[layer_idx]
        return None

    def seq_len(self, layer_idx: int = 0) -> int:
        entry = self.get(layer_idx)
        return entry.seq_len if entry is not None else 0

    def clear(self) -> None:
        self._entries.clear()


class SlidingWindowCache(DynamicCache):
    """Keeps only the most recent `window_size` tokens per layer.

    Useful for long-context generation where full KV retention is too
    expensive in memory.
    """

    def __init__(self, window_size: int) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size

    def update(
        self,
        layer_idx: int,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        full_k, full_v = super().update(layer_idx, new_keys, new_values)

        if full_k.shape[2] > self.window_size:
            trimmed_k = full_k[:, :, -self.window_size:, :].contiguous()
            trimmed_v = full_v[:, :, -self.window_size:, :].contiguous()
            self._entries[layer_idx] = CacheEntry(trimmed_k, trimmed_v)
            return trimmed_k, trimmed_v

        return full_k, full_v


def make_cache(
    cache_type: str = "dynamic",
    window_size: int = 512,
) -> DynamicCache:
    """Factory helper to create a cache by name.

    Args:
        cache_type: One of ``"dynamic"`` or ``"sliding_window"``.
        window_size: Only used when ``cache_type="sliding_window"``.
            Default changed to 512 (was 256 upstream) — fits better on
            my 24 GB GPU for the models I typically run.

    Returns:
        A freshly initialised cache instance.
    """
    if cache_type == "sliding_window":
        return SlidingWindowCache(window_size=window_size)
    if cache_type == "dynamic":
        return DynamicCache()
    raise ValueError(f"Unknown cache_type {cache_type!r}. Choose 'dynamic' or 'sliding_window'.")
