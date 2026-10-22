# prophet-batch

Fit and forecast many [Prophet](https://github.com/facebook/prophet) models at once, on top of the stock
`prophet` package (≥ 1.5):

- **`fit_many`** fits thousands of series together with batched MAP solvers instead of one Stan run per series.
  The result for each series is an ordinary `Prophet` model whose log posterior is at least as high as Stan's.
- **`predict_many`** forecasts them in parallel with per-series seeds: a forecast depends only on
  (seed, series id), not on worker count, order, or which other series are predicted.
- **`cross_validation_many`** runs stock-format cross-validation for many models at once.
- **`predict`** is `Prophet.predict` with a per-call random generator and faster history-interval modes.

Measured against stock Prophet 1.5.0 on the same 10 workers (fit + forecast, six public datasets, 38,905 series):
the same holdout accuracy, and 1.5× (short series) to 26.5× (M4 Daily) faster with full history intervals.
Guarantees and how they are tested: [docs/guarantees.md](https://github.com/sumedhekaru/prophet-batch/blob/main/docs/guarantees.md). Runnable use cases with accuracy
and speed against stock Prophet on macOS, Windows, Linux (Docker), Cloud Run and Spark (Dataproc):
[prophet-batch-examples](https://github.com/sumedhekaru/prophet-batch-examples).

Status: alpha (0.1.1): the public API (below) is stable in intent but may still change before 1.0.

## Install

```sh
pip install prophet-batch
```

Requires Python ≥ 3.10 and `prophet>=1.5` (the stock PyPI wheel; no Stan build needed).

## Quick start

Data is in long format: one row per (series, date), with a series-id column, `ds` and `y`.

```python
import numpy as np
import pandas as pd

def make_series(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 10 + 0.02 * t + 2 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=n), "y": y})

df = pd.concat([make_series(400, i).assign(unique_id=f"store_{i}") for i in range(8)], ignore_index=True)
```

```python
from prophet_batch import fit_many, predict_many

models = fit_many(df, n_jobs=-1)          # {series id: fitted Prophet}
forecasts = predict_many(models, periods=28, seed=42, n_jobs=-1)
forecasts["store_0"][["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
```

`n_jobs=-1` uses every CPU available to the process (in a container: its CPU limit, which `os.cpu_count()`
ignores); results are bit-identical for any `n_jobs`. Each model is a normal `Prophet` object,
so `plot`, `plot_components`, `model_to_json`, `cross_validation` and the rest of the stock API work on it.

### Configuring the models

Keyword arguments go to `Prophet()` for every series. For per-series configuration (holidays, extra regressors,
custom seasonalities), pass a function that returns a fresh, unfitted model for a series id:

```python
from prophet import Prophet

def make_model(series_id):
    m = Prophet(seasonality_mode="multiplicative" if series_id.endswith("7") else "additive")
    m.add_country_holidays("US")
    return m

models = fit_many(df, make_model=make_model)
```

With `n_jobs != 1`, `make_model` must be picklable (a module-level function, not a lambda). Series whose models
come out identical and that have the same dates and regressor values (e.g. one holiday calendar and a regressor
shared by all series) are preprocessed once, as without `make_model`; the results are the same either way.

Other options: `fit_kwargs` (arguments for `Prophet.fit`, as a dict or a per-series function, used by series that
are fit by Stan), and `on_error="skip"` (leave out series that Prophet can't fit, e.g. fewer than 2 non-NaN rows,
with a warning, instead of raising).

### What is batched

Linear and flat growth are fit by the batched solvers, with any mix of additive and multiplicative components
(seasonalities, holidays, extra regressors, conditional seasonalities). Additive models are solved exactly (the MAP
is a convex problem); multiplicative ones to a verified stationary point. Everything else (logistic growth,
`mcmc_samples > 0`, regressors with `regressor_predictor`) and any series the solver can't handle is fit by
`Prophet.fit` per series, with exactly the result of calling it directly.

### Forecasts and history intervals

`predict_many(models, periods=..., ...)` builds each future with `make_future_dataframe`; pass
`futures={series id: DataFrame}` instead for regressors or logistic caps. `history_uncertainty` controls the
intervals on history rows, which are most of the cost when the history is included:

| `history_uncertainty` | history rows | future rows | speed |
|---|---|---|---|
| `True` (default) | sampled, as stock Prophet | sampled | like stock |
| `"analytic"` (MAP fits) | exact closed form: the limit of the sampled intervals | sampled, as with `False` | ≈ `False` |
| `False` | NaN | sampled (identical to predicting the future rows alone) | fastest |

`yhat` is the same in every mode. The future rows' intervals are equally valid draws in each mode, but not the same
numbers between `True` and the other two: with `True` the series' random stream is first used for the history rows.

```python
forecasts = predict_many(models, periods=28, seed=42, history_uncertainty="analytic")
```

Prophet's forecast frame has three columns per component (each holiday and regressor included), so with many
holidays and series it gets large. `columns=` keeps only what you need; the workers drop the rest before sending
results back, which saves memory and time (M5, 1,437 series with 30 holidays: 2.3 GB → 0.1 GB):

```python
forecasts = predict_many(models, periods=28, seed=42, columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
```

### Single models

`prophet_batch.predict` adds the same options to one model, with a numpy `Generator` for the draws. With the
default arguments it is bit-identical to `model.predict(df)`.

```python
from prophet_batch import predict

model = models["store_0"]
fc = predict(model, model.make_future_dataframe(28), rng=np.random.default_rng(0),
             history_uncertainty="analytic")
```

### Cross-validation

Same arguments and output columns as `prophet.diagnostics.cross_validation`, for many models:

```python
from prophet.diagnostics import performance_metrics
from prophet_batch import cross_validation_many

cvs = cross_validation_many(models, horizon="14 days", period="60 days", initial="250 days", seed=1)
metrics = {sid: performance_metrics(cv) for sid, cv in cvs.items()}
```

`as_frame=True` returns one DataFrame with a `unique_id` column.

## Reproducibility

`predict_many(seed=...)` and `cross_validation_many(seed=...)` give every series its own generator, derived from
(seed, series id) with `numpy.random.SeedSequence`, so results do not depend on `n_jobs`, order or batch
composition. Without a seed, a fresh one is drawn, logged, and stored in each forecast's `attrs["seed"]`.

## API stability

The public API is what `prophet_batch` exports at the top level: `fit_many`, `predict_many`,
`cross_validation_many`, `predict` and `series_rng`. Everything else (the solvers and helpers in
`prophet_batch.batch`, such as `solve_map` or `log_posterior`, and the private modules) is internal and may change
between releases.

## Limitations

- Logistic growth, MCMC and regressor predictors are fit by Stan (correct, but not faster than stock).
- Prediction intervals are about 0.04–0.6 percentage points narrower than stock's on public data: the exact MAP
  fits a little tighter than Stan's L-BFGS, which stops short of the optimum.
- With few, short series, process start-up dominates: use `n_jobs=1` there. For small jobs, fewer workers are often
  faster than `n_jobs=-1`, especially on Windows, where each worker is a newly started process that imports Prophet
  (Windows laptop, 400 items: 5.8 s with 4 workers vs 11.0 s with 16).
- **Windows speed (fixed in 0.1.1):** 0.1.0 was slow on Windows because cmdstanpy starts a `where.exe` process each
  time a Prophet model is created there (≈ 100 ms per model). Since 0.1.1, `fit_many` and `cross_validation_many`
  reuse one loaded Stan model per process; results are unchanged. On a Windows laptop (Intel Core i7-1260P, 16
  logical CPUs), 1,437 M5 items: 3.1× → 11.5× faster than a 16-process stock Prophet pool (total 125.8 s → 30.3 s),
  in line with macOS (10.2×) and Linux (8.6×). Stock Prophet on Windows still pays this cost per model.

More: [docs/guarantees.md](https://github.com/sumedhekaru/prophet-batch/blob/main/docs/guarantees.md) (what is guaranteed and how it is tested) and
[docs/migration.md](https://github.com/sumedhekaru/prophet-batch/blob/main/docs/migration.md) (from a stock loop or process pool).

## License

MIT. Contains code derived from Prophet (Copyright (c) Facebook, Inc. and its affiliates).
