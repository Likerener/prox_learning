"""Bootstrap confidence intervals and paired p-values.

PROJECT.md §4.1 statistical protocol:
  * Bootstrap 95% CI on every reported number.
  * Paired bootstrap p-values for PLA vs VLM-only ACT.
  * Threshold p < 0.05.

All functions are deterministic given a seed.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    successes: np.ndarray,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Returns (mean, low, high) of the success-rate bootstrap distribution."""
    rng = np.random.default_rng(seed)
    successes = np.asarray(successes, dtype=np.float64)
    n = len(successes)
    samples = rng.choice(successes, size=(n_resamples, n), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(successes.mean()), float(low), float(high)


def paired_bootstrap_p(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """Paired bootstrap p-value for H0: mean(a) == mean(b).

    ``a`` and ``b`` are per-episode 0/1 success arrays of equal length, with
    paired ordering (same seed, same scene).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    obs = diff.mean()
    centered = diff - obs
    samples = rng.choice(centered, size=(n_resamples, n), replace=True).mean(axis=1)
    return float((np.abs(samples) >= abs(obs)).mean())
