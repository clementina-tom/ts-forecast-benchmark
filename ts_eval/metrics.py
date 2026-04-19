"""Forecast error metrics and a registry for custom metric extension."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _as_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    return y_true, y_pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-8) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-8) -> float:
    y_true, y_pred = _as_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_train: np.ndarray | None = None,
    seasonality: int = 1,
    eps: float = 1e-8,
) -> float:
    """Mean Absolute Scaled Error using in-sample naive baseline."""
    y_true, y_pred = _as_arrays(y_true, y_pred)

    if y_train is None:
        y_train = y_true
    y_train = np.asarray(y_train, dtype=float).reshape(-1)

    if seasonality <= 0:
        raise ValueError("seasonality must be > 0")
    if len(y_train) <= seasonality:
        raise ValueError("y_train length must be > seasonality")

    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.maximum(np.mean(naive_errors), eps)
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def default_metrics() -> dict[str, MetricFn]:
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
    }
