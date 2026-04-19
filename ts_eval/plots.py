"""Visualization helpers for backtesting output."""

from __future__ import annotations

import numpy as np

from .backtest import BacktestResult


def plot_backtest(
    y_full: np.ndarray,
    result: BacktestResult,
    *,
    title: str = "Walk-forward backtest",
    show_splits: bool = False,
    figsize: tuple[int, int] = (12, 5),
):
    """Plot actuals and stitched forecast points from a BacktestResult."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("matplotlib is required for plot_backtest") from exc

    y_full = np.asarray(y_full, dtype=float).reshape(-1)
    ax = plt.figure(figsize=figsize).gca()

    ax.plot(np.arange(len(y_full)), y_full, label="Actual", linewidth=1.8)

    # Plot stitched backtest predictions over evaluation horizon.
    eval_start = len(y_full) - len(result.y_pred)
    pred_x = np.arange(eval_start, eval_start + len(result.y_pred))
    ax.plot(pred_x, result.y_pred, label="Forecast", linestyle="--", linewidth=1.8)

    if show_splits and len(result.split_ids) > 0:
        unique_splits = np.unique(result.split_ids)
        for sid in unique_splits:
            first_idx = pred_x[np.where(result.split_ids == sid)[0][0]]
            ax.axvline(first_idx, color="gray", alpha=0.15, linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.tight_layout()
    return ax
