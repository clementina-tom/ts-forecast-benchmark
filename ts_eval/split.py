"""Data splitting utilities for leakage-safe walk-forward evaluation."""

from __future__ import annotations

from typing import Generator

import numpy as np


def walk_forward_split(
    n_samples: int,
    *,
    train_size: int,
    horizon: int = 1,
    step: int = 1,
    max_splits: int | None = None,
    expanding: bool = True,
    window_size: int | None = None,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate index splits for true walk-forward backtesting.

    Parameters
    ----------
    n_samples:
        Total number of observations in the series.
    train_size:
        Initial training size (if expanding=True), or rolling window length
        (if expanding=False and window_size is None).
    horizon:
        Forecast horizon per split.
    step:
        Number of observations to advance between successive splits.
    max_splits:
        Optional cap on number of generated splits.
    expanding:
        If True, training window expands over time.
        If False, training window rolls over time.
    window_size:
        Explicit rolling window size. Only used when expanding=False.

    Yields
    ------
    (train_idx, test_idx):
        Numpy arrays containing train and test indices for each split.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if train_size <= 0:
        raise ValueError("train_size must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    if train_size + horizon > n_samples:
        raise ValueError("train_size + horizon must be <= n_samples")

    if not expanding:
        window = window_size if window_size is not None else train_size
        if window <= 0:
            raise ValueError("window_size must be > 0")
    else:
        window = None

    splits = 0
    train_end = train_size

    while train_end + horizon <= n_samples:
        if expanding:
            train_start = 0
        else:
            train_start = max(0, train_end - int(window))

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(train_end, train_end + horizon)
        yield train_idx, test_idx

        splits += 1
        if max_splits is not None and splits >= max_splits:
            break
        train_end += step
