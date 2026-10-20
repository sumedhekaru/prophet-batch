"""Synthetic test series (no pytest dependency, so make_golden.py can import it)."""

import numpy as np
import pandas as pd


def make_series(n_days: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Daily series with trend, weekly + yearly seasonality, and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    y = (
        10
        + 0.01 * t
        + 2 * np.sin(2 * np.pi * t / 7)
        + 3 * np.sin(2 * np.pi * t / 365.25)
        + rng.normal(0, 0.5, n_days)
    )
    return pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=n_days), "y": y})
