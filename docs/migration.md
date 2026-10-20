# From a stock Prophet loop to prophet-batch

The usual ways to run Prophet on many series, and the prophet-batch equivalent. The fitted models are ordinary
`Prophet` objects either way, so everything after fitting (plots, `model_to_json`, `performance_metrics`, your own
post-processing) stays the same.

Example data: long format, one row per (series, date).

```python
import numpy as np
import pandas as pd
from prophet import Prophet

def make_series(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 50 + 0.05 * t + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, n)
    return pd.DataFrame({"ds": pd.date_range("2023-01-01", periods=n), "y": y})

df = pd.concat([make_series(365, i).assign(unique_id=f"sku_{i}") for i in range(4)], ignore_index=True)
```

## Fit and forecast

Stock, one series at a time (or the same function in a `ProcessPoolExecutor`):

```python
forecasts_stock = {}
for sid, g in df.groupby("unique_id"):
    m = Prophet().fit(g[["ds", "y"]])
    forecasts_stock[sid] = m.predict(m.make_future_dataframe(periods=28))
```

prophet-batch:

```python
from prophet_batch import fit_many, predict_many

models = fit_many(df, n_jobs=-1)
forecasts = predict_many(models, periods=28, seed=0, n_jobs=-1)
assert list(forecasts["sku_0"].columns) == list(forecasts_stock["sku_0"].columns)
```

- `Prophet(...)` arguments go to `fit_many(df, **kwargs)`; per-series setup (holidays, `add_regressor`,
  `add_seasonality`) goes in a `make_model(series_id)` function that returns a fresh model.
- `m.fit(df, **kwargs)` arguments go to `fit_many(..., fit_kwargs=...)` (a dict, or a function of the series id).
- `np.random.seed(...)` before `predict` becomes `predict_many(..., seed=...)`: per-series seeds that do not depend
  on the order or the number of workers.
- If you predict on the history only for the point forecast, add `history_uncertainty="analytic"` (exact intervals,
  much faster) or `False` (no history intervals).

## Futures with regressors or caps

Stock: build each future frame and call `m.predict(future)`. prophet-batch: pass them as a dict.

```python
futures = {sid: m.make_future_dataframe(periods=14) for sid, m in models.items()}
forecasts = predict_many(models, futures=futures, seed=0)
```

## Cross-validation

Stock:

```python
from prophet.diagnostics import cross_validation

cv_stock = cross_validation(models["sku_0"], horizon="14 days", period="60 days", initial="200 days")
```

prophet-batch (same columns; every series' refits are batched):

```python
from prophet_batch import cross_validation_many

cvs = cross_validation_many(models, horizon="14 days", period="60 days", initial="200 days", seed=0)
assert list(cvs["sku_0"].columns) == list(cv_stock.columns)
```

## One model, seeded

```python
from prophet_batch import predict

fc = predict(models["sku_0"], futures["sku_0"], rng=np.random.default_rng(1))
```

## What changes in the numbers

Series with linear or flat growth are fit by an exact optimizer instead of Stan's L-BFGS, so their parameters, and
therefore their forecasts, differ slightly from stock (the log posterior is never lower). Holdout accuracy is the
same on public benchmarks; 80% intervals are slightly narrower. Logistic, MCMC and regressor-predictor series are fit
by Stan exactly as before. Details: [guarantees.md](guarantees.md).
