# ts-forecast-benchmark

`ts_eval` is a reusable, model-agnostic toolkit for **leakage-safe time-series backtesting**.

It is designed for real-world forecasting pipelines and supports:

- **Model-agnostic forecasting**: plain callables, sklearn-style estimators (`fit/predict`), and adapters for statsmodels/deep learning.
- **True walk-forward evaluation**: expanding or rolling windows with no future leakage.
- **Extensible metrics**: built-in `rmse`, `mae`, `mape`, `smape`, `mase`, and easy custom metric injection.
- **Production-friendly outputs**: per-split + overall metrics, stitched predictions for monitoring and plotting.

## Package layout

```text
ts_eval/
  split.py       # walk_forward_split()
  metrics.py     # rmse, mae, mape, smape, mase
  backtest.py    # evaluate_forecast()
  plots.py       # plot_backtest()
```

## Quick start

```python
import numpy as np
from ts_eval import evaluate_forecast

# Sample series
y = np.sin(np.linspace(0, 30, 300)) + 0.1 * np.random.randn(300)

# Any callable that accepts (y_train, horizon)
def naive_last(y_train, horizon):
    return np.repeat(y_train[-1], horizon)

result = evaluate_forecast(
    y,
    forecaster=naive_last,
    train_size=120,
    horizon=12,
    step=12,
    expanding=True,
)

print(result.metrics)
# {'rmse': ..., 'mae': ..., 'mape': ..., 'smape': ...}
```

## Model adapters (examples)

### ARIMA/SARIMA (statsmodels-style callable)

```python
def arima_forecaster(y_train, horizon):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(y_train, order=(2, 1, 2)).fit()
    pred = model.forecast(steps=horizon)
    return pred
```

### LSTM / deep model wrapper

```python
class LSTMForecaster:
    def __call__(self, y_train, horizon, context=None):
        # train/update model on y_train
        # return horizon-step forecast as a 1D array
        return my_lstm_predict(y_train, horizon)
```

### XGBoost / sklearn estimator

```python
from xgboost import XGBRegressor
from ts_eval import evaluate_forecast

model = XGBRegressor(n_estimators=200, max_depth=4)

# supply supervised features for each timestep
X = make_tabular_features(...)  # shape (n_samples, n_features)

eval_result = evaluate_forecast(
    y,
    forecaster=model,   # has fit/predict, handled automatically
    X=X,
    train_size=200,
    horizon=24,
    step=24,
    expanding=False,
    window_size=300,
)
```

## Custom metrics plug-in

Metric functions can be either:
- `fn(y_true, y_pred)`
- `fn(y_true, y_pred, context=...)`

```python
from ts_eval import evaluate_forecast, rmse


def p90_abs_error(y_true, y_pred):
    import numpy as np
    return float(np.quantile(np.abs(y_true - y_pred), 0.90))

metrics = {
    "rmse": rmse,
    "p90_abs_error": p90_abs_error,
}

result = evaluate_forecast(
    y,
    forecaster=naive_last,
    train_size=120,
    horizon=12,
    metrics=metrics,
)
```

## Plotting

```python
from ts_eval.plots import plot_backtest

ax = plot_backtest(y, result, show_splits=True)
```

## Notes for production usage

- Use `expanding=True` to mimic retraining as new data arrives.
- Use `expanding=False` + `window_size` for bounded-memory rolling training.
- Keep feature creation strictly causal for each split.
- Pair with your model registry/orchestration layer and persist `result.metrics` + `result.metric_by_split` for monitoring.
