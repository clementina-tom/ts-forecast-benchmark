"""Model-agnostic walk-forward backtesting framework."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any

import numpy as np

from .metrics import MetricFn, default_metrics
from .split import walk_forward_split


@dataclass(slots=True)
class BacktestResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    split_ids: np.ndarray
    metrics: dict[str, float]
    metric_by_split: dict[str, list[float]]


ForecastFn = Any


def _invoke_forecaster(
    forecaster: ForecastFn,
    y_train: np.ndarray,
    horizon: int,
    *,
    X_train: np.ndarray | None = None,
    X_future: np.ndarray | None = None,
    context: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Invoke arbitrary forecasting callable in a flexible way.

    Supported patterns (examples):
      * f(y_train, horizon)
      * f(y_train=y_train, horizon=horizon)
      * f(y_train=y_train, X_train=X_train, X_future=X_future, horizon=horizon)
      * sklearn-like object exposing fit/predict
    """
    context = context or {}

    if hasattr(forecaster, "fit") and hasattr(forecaster, "predict"):
        model = forecaster
        if X_train is None:
            model.fit(np.arange(len(y_train)).reshape(-1, 1), y_train)
            future_idx = np.arange(len(y_train), len(y_train) + horizon).reshape(-1, 1)
            pred = model.predict(future_idx)
        else:
            model.fit(X_train, y_train)
            if X_future is None:
                raise ValueError("X_future must be supplied when X_train is provided")
            pred = model.predict(X_future)
        return np.asarray(pred, dtype=float).reshape(-1)

    sig = signature(forecaster)
    kwargs: dict[str, Any] = {}
    for name in sig.parameters:
        if name in {"y_train", "train", "series"}:
            kwargs[name] = y_train
        elif name in {"horizon", "steps", "n_steps"}:
            kwargs[name] = horizon
        elif name == "X_train":
            kwargs[name] = X_train
        elif name in {"X_future", "X_test"}:
            kwargs[name] = X_future
        elif name == "context":
            kwargs[name] = context

    if kwargs:
        pred = forecaster(**kwargs)
    else:
        pred = forecaster(y_train, horizon)
    return np.asarray(pred, dtype=float).reshape(-1)


def _invoke_metric(
    metric_fn: MetricFn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    context: dict[str, Any],
) -> float:
    sig = signature(metric_fn)
    if len(sig.parameters) <= 2:
        return float(metric_fn(y_true, y_pred))
    return float(metric_fn(y_true, y_pred, context=context))


def evaluate_forecast(
    y: np.ndarray,
    forecaster: ForecastFn,
    *,
    X: np.ndarray | None = None,
    train_size: int,
    horizon: int = 1,
    step: int = 1,
    expanding: bool = True,
    window_size: int | None = None,
    max_splits: int | None = None,
    metrics: dict[str, MetricFn] | None = None,
    keep_split_metrics: bool = True,
) -> BacktestResult:
    """Run leakage-safe walk-forward backtesting for any forecasting callable."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if X is not None:
        X = np.asarray(X)
        if len(X) != len(y):
            raise ValueError("X and y must have same number of rows")

    metric_fns = default_metrics() if metrics is None else metrics

    all_true: list[float] = []
    all_pred: list[float] = []
    all_split_ids: list[int] = []
    metric_by_split: dict[str, list[float]] = {k: [] for k in metric_fns}

    split_iter = walk_forward_split(
        len(y),
        train_size=train_size,
        horizon=horizon,
        step=step,
        max_splits=max_splits,
        expanding=expanding,
        window_size=window_size,
    )

    for split_id, (train_idx, test_idx) in enumerate(split_iter):
        y_train = y[train_idx]
        y_test = y[test_idx]
        X_train = X[train_idx] if X is not None else None
        X_future = X[test_idx] if X is not None else None

        context = {
            "split_id": split_id,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "y_train": y_train,
            "y_test": y_test,
            "horizon": horizon,
        }

        preds = _invoke_forecaster(
            forecaster,
            y_train,
            len(test_idx),
            X_train=X_train,
            X_future=X_future,
            context=context,
        )

        if len(preds) != len(y_test):
            raise ValueError(
                f"Forecaster returned {len(preds)} predictions, expected {len(y_test)}"
            )

        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())
        all_split_ids.extend([split_id] * len(test_idx))

        if keep_split_metrics:
            for name, metric_fn in metric_fns.items():
                metric_by_split[name].append(
                    _invoke_metric(metric_fn, y_test, preds, context=context)
                )

    y_true_arr = np.asarray(all_true, dtype=float)
    y_pred_arr = np.asarray(all_pred, dtype=float)

    overall_context = {
        "y": y,
        "train_size": train_size,
        "horizon": horizon,
        "step": step,
        "expanding": expanding,
    }
    overall = {
        name: _invoke_metric(fn, y_true_arr, y_pred_arr, context=overall_context)
        for name, fn in metric_fns.items()
    }

    return BacktestResult(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        split_ids=np.asarray(all_split_ids, dtype=int),
        metrics=overall,
        metric_by_split=metric_by_split,
    )
