"""Pluggable window normalization — the definition of "same shape".

Each normalizer maps a raw matrix of log-returns (M, W) to (Z, valid):
Z is the normalized float32 matrix, valid marks rows that are usable
(e.g. zero-variance windows cannot be z-scored and are excluded).
The choice of normalizer is an identity field: a different shape
definition is a different hypothesis in the ledger.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped

# (raw log-return matrix) -> (normalized float32 matrix, per-row usable flag)
Normalizer = Callable[
    [Float[np.ndarray, "M D"]],
    tuple[Float[np.ndarray, "M D"], Bool[np.ndarray, " M"]],
]

NORMALIZERS: dict[str, Normalizer] = {}

_EPS = 1e-12


def register(name: str) -> Callable[[Normalizer], Normalizer]:
    def deco(fn: Normalizer) -> Normalizer:
        NORMALIZERS[name] = fn
        return fn

    return deco


@register("logret_zscore")
@jaxtyped(typechecker=beartype)
def logret_zscore(
    X: Float[np.ndarray, "M D"],
) -> tuple[Float[np.ndarray, "M D"], Bool[np.ndarray, " M"]]:
    """Z-score each window of log-returns: removes price level (already gone
    in returns) and window volatility — pure geometry remains."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    valid = sd[:, 0] > _EPS
    Z = (X - mu) / np.where(sd > _EPS, sd, 1.0)
    return Z.astype(np.float32), valid


@register("logret_raw")
@jaxtyped(typechecker=beartype)
def logret_raw(
    X: Float[np.ndarray, "M D"],
) -> tuple[Float[np.ndarray, "M D"], Bool[np.ndarray, " M"]]:
    """Log-returns without z-scoring: volatility stays part of the shape."""
    return X.astype(np.float32), np.ones(len(X), dtype=bool)


def normalize(
    name: str, X: Float[np.ndarray, "M D"]
) -> tuple[Float[np.ndarray, "M D"], Bool[np.ndarray, " M"]]:
    if name not in NORMALIZERS:
        raise KeyError(f"Unknown normalization {name!r}; available: {sorted(NORMALIZERS)}")
    return NORMALIZERS[name](X)
