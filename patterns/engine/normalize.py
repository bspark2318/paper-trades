"""Pluggable window normalization — the definition of "same shape".

Each normalizer maps a raw matrix of log-returns (M, W) to (Z, valid):
Z is the normalized float32 matrix, valid marks rows that are usable
(e.g. zero-variance windows cannot be z-scored and are excluded).
The choice of normalizer is an identity field: a different shape
definition is a different hypothesis in the ledger.
"""

from __future__ import annotations

import numpy as np

NORMALIZERS: dict = {}

_EPS = 1e-12


def register(name: str):
    def deco(fn):
        NORMALIZERS[name] = fn
        return fn

    return deco


@register("logret_zscore")
def logret_zscore(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score each window of log-returns: removes price level (already gone
    in returns) and window volatility — pure geometry remains."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    valid = sd[:, 0] > _EPS
    Z = (X - mu) / np.where(sd > _EPS, sd, 1.0)
    return Z.astype(np.float32), valid


@register("logret_raw")
def logret_raw(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Log-returns without z-scoring: volatility stays part of the shape."""
    return X.astype(np.float32), np.ones(len(X), dtype=bool)


def normalize(name: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if name not in NORMALIZERS:
        raise KeyError(f"Unknown normalization {name!r}; available: {sorted(NORMALIZERS)}")
    return NORMALIZERS[name](X)
