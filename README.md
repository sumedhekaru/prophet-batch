# prophet-batch

[![PyPI](https://img.shields.io/pypi/v/prophet-batch)](https://pypi.org/project/prophet-batch/)
[![Python](https://img.shields.io/pypi/pyversions/prophet-batch)](https://pypi.org/project/prophet-batch/)
[![CI](https://github.com/sumedhekaru/prophet-batch/actions/workflows/ci.yml/badge.svg)](https://github.com/sumedhekaru/prophet-batch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/pypi/l/prophet-batch)](https://github.com/sumedhekaru/prophet-batch/blob/main/LICENSE)

**Forecast thousands to millions of series with [Prophet](https://github.com/facebook/prophet), about 10× faster than
a stock Prophet process pool on the same cores, with the same accuracy.** prophet-batch runs on top of the stock
`prophet` package (≥ 1.5): your models stay ordinary `Prophet` objects, and your data stays in your hands.

- **`fit_many`** fits all series together with exact batched solvers instead of one Stan optimization per series.
  Every fit reaches a log posterior at least as high as Stan's (checked on every one of 38,905 public series).
- **`predict_many`** forecasts them in parallel, reproducibly: each series has its own seeded random stream.
- **`forecast_many` and `Pool`** fit and forecast in one step for large catalogs: batch after batch from your
  own files or database, with flat memory, workers kept between batches.
- **`cross_validation_many`** and **`predict`**: stock-format cross-validation for many models, and stock's predict
  with a seedable generator and faster history intervals for one.

**See it in use:** [prophet-batch-examples](https://github.com/sumedhekaru/prophet-batch-examples): runnable use cases
with accuracy and speed against stock Prophet. Walmart grocery forecasting on a laptop, in Docker, as a Cloud Run job
and on Spark (Dataproc); six public forecasting datasets; and a large CSV forecast in batches with flat memory.

## Results

1,437 Walmart grocery items (M5, FOODS, daily, 28 days ahead), default `Prophet()`, fit + forecast, against stock
Prophet run the way people parallelize it: a process pool on every core. prophet-batch 0.1.1 from PyPI, median of 3
runs ([all tables, and the code to rerun them](https://github.com/sumedhekaru/prophet-batch-examples/tree/main/walmart_grocery_forecasting)):

| environment | stock Prophet, process pool | prophet-batch | faster |
|---|---:|---:|---:|
| MacBook, Apple M4, 10 cores | 53.1 s | 5.2 s | **10.2×** |
| Linux container (Docker), same Mac | 57.2 s | 6.7 s | **8.6×** |
| Cloud Run job, 4 vCPU | 254.4 s | 23.6 s | **10.9×** |
| Windows laptop, Intel i7-1260P, 16 threads | 298.9 s | 25.8 s | **11.6×** |
| Spark on Dataproc Serverless, 2 × 4 cores (execution¹) | 234.5 s | 24.2 s | **9.7×** |

¹ The Spark job's execution phase; end to end 1.9×, because each batch also spends ≈ 185 s on platform start-up and
loading, the same for both.

- **Same accuracy:** WAPE 0.3438 vs 0.3446, MASE 1.4261 vs 1.4279 (difference within noise), on every platform;
  with holidays and with holidays + regressors as well (7.4–11.2× there).
- **Six public datasets** (M5 item and item × store, M4 Daily, Weekly, Hourly, Tourism Monthly: 38,905 series, the
  same 10 workers; measured before the first release, re-run on 0.1.2 to come): the same holdout accuracy, slightly
  better beyond noise on four of them, and 1.5× (short series) to **26.5×** (M4 Daily, long series) faster with full
  history intervals.
- **At scale:** 15,000 item × store series in one call (Apple M4; four forecast columns, analytic history
  intervals): `forecast_many` 26.5 s and 4.0 GB at peak, vs 33.7 s and 6.3 GB with `fit_many` + `predict_many`.
  Streaming a CSV in batches through one `Pool`: the file grew 8.7× (169 MB → 1.5 GB), memory 0.44 → 0.55 GB.

## Why it is fast, and what you can rely on

- **Exact batched fits.** For Prophet's linear-growth model the MAP is a convex problem (a lasso on the changepoints
  from their Laplace prior), solved exactly for many series at once; multiplicative seasonality by a verified
  Gauss–Newton refinement. Stan's L-BFGS stops a little short of that optimum, so prophet-batch's fits come out at
  least as good: on every series of every benchmark so far. Logistic growth, MCMC and regressor predictors go to
  Stan, with exactly its result.
- **Shared work.** Series with the same dates and configuration share their features and design matrices.
- **Exact history intervals.** `history_uncertainty="analytic"` gives the closed form that sampled history intervals
  converge to, instead of 1,000 simulated paths per series: most of the forecast time otherwise.
- **Bit-identical results** for any `n_jobs`, `batch_size`, series order and split of the input: a series' fit
  depends only on its own data, and its forecast only on (seed, series id). Across CPU types they agree to ≈ 1e-9.
- **Stock behaviour where it matters.** Defaults follow stock Prophet; fitted models work with `plot`,
  `plot_components`, `model_to_json`, `cross_validation` and the rest of the stock API (41 callables checked
  bit-identical).
- **Bounded memory, any platform.** Workers prepare at most about 100 series at a time, sized to half the available
  memory (container limits included); one Stan model is reused per process (Windows); `n_jobs=-1` respects container
  CPU limits. CI on Linux, macOS and Windows, Python 3.10–3.13.

Guarantees and the tests behind them: [docs/guarantees.md](https://github.com/sumedhekaru/prophet-batch/blob/main/docs/guarantees.md).

Status: alpha (0.1.1): the public API (below) is stable in intent but may still change before 1.0.

## Install

```sh
pip install prophet-batch
```

Requires Python ≥ 3.10 and `prophet>=1.5` (the stock PyPI wheel; no Stan build needed).

## Quick start

Data is in long format: one row per (series, date), with a series-id column, `ds` and `y`. The snippets in this
README run in order (later ones use `df` and `models` from here), and are run in CI.

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

Forecasts only, in one call (stock Prophet's columns, one long frame with the series id first):

```python
from prophet_batch import forecast_many

fc = forecast_many(df, periods=28, seed=42, n_jobs=-1)
fc[fc["unique_id"] == "store_0"][["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
```

Or keep the fitted models (ordinary `Prophet` objects) and forecast them:

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
are fit by Stan), `on_error="skip"` (leave out series that Prophet can't fit, e.g. fewer than 2 non-NaN rows,
with a warning, instead of raising), and `batch_size` (the most series a worker prepares and solves at once;
`fit_many` and `cross_validation_many`). By default it is chosen automatically: up to 100 series (larger batches
are not faster), fewer so every worker gets several tasks, and fewer again so the batches in flight use at most
half of the available memory, which matters for long series. Results are bit-identical for any `batch_size`.

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

### Large catalogs: forecast_many and Pool

`forecast_many` fits and forecasts in one step and returns one long DataFrame: the series id first, then exactly what
stock Prophet's `model.predict(model.make_future_dataframe(periods))` returns for each series (equal to
`predict_many(fit_many(df), ...)` for that series and seed). No fitted model is kept or sent between processes, so
it is faster and uses less memory than the two steps; on 15,000 M5 item × store series (10 workers, Apple M4):
138.5 s and a 7.2 GB peak with stock's full output vs 147.9 s and 8.6 GB, and 26.5 s and 4.0 GB vs 33.7 s and 6.3 GB
with `columns=["ds", "yhat", "yhat_lower", "yhat_upper"]` and `history_uncertainty="analytic"`.

```python
from prophet_batch import Pool, forecast_many

fc = forecast_many(df, periods=28, seed=0, n_jobs=2)   # every series, stock's columns; n_jobs=-1: all CPUs


def read_batches():                                    # stands for your data flow: files, a database, a queue
    ids = df["unique_id"].unique()
    for i in range(0, len(ids), 4):
        yield df[df["unique_id"].isin(ids[i:i + 4])]


written = []
with Pool(n_jobs=2) as pool:                           # workers started once, kept for every batch
    for batch in read_batches():
        written.append(pool.forecast_many(batch, periods=28, seed=0, include_history=False,
                                          columns=["ds", "yhat", "yhat_lower", "yhat_upper"]))
```

prophet-batch only computes: reading and writing stay in your code, in any format. `pool.submit(batch, ...)` returns
a `concurrent.futures.Future`, so you can read the next batch or write the last one while this one runs. Options as
in `fit_many` (`make_model`, `fit_kwargs`, `on_error`, `batch_size`, Prophet arguments) and `predict_many` (`seed`,
`freq`, `include_history`, `history_uncertainty`, `columns`); `futures` (a long frame with the id column, or a dict)
for regressors and caps; `return_params=True` also returns each series' fitted `model.params`. Series with
different components (e.g. holidays for some) share one frame: a column a series doesn't have is NaN there. A
streaming example (read a CSV in batches, forecast, write, with flat memory):
[prophet-batch-examples](https://github.com/sumedhekaru/prophet-batch-examples).

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

The public API is what `prophet_batch` exports at the top level: `fit_many`, `predict_many`, `forecast_many`,
`Pool`, `cross_validation_many`, `predict` and `series_rng`. Everything else (the solvers and helpers in
`prophet_batch.batch`, such as `solve_map` or `log_posterior`, and the private modules) is internal and may change
between releases.

## Limitations

- Logistic growth, MCMC and regressor predictors are fit by Stan (correct, but not faster than stock).
- Prediction intervals are about 0.04–0.6 percentage points narrower than stock's on public data: the exact MAP
  fits a little tighter than Stan's L-BFGS, which stops short of the optimum.
- **Memory**: each worker holds one batch at a time (about 0.5–0.6 GB per worker for M5's 1,941-day series), but
  the models `fit_many` returns stay in your process: ≈ 0.12 MB per series at that length, mostly the history
  frame every stock Prophet model keeps (30,490 such series: ≈ 4 GB for the models alone). When you only need the
  forecasts, `forecast_many` keeps no models; batch through a `Pool` for any number of series.
- With few, short series, process start-up dominates: use `n_jobs=1` there. For small jobs, fewer workers are often
  faster than `n_jobs=-1`, especially on Windows, where each worker is a newly started process that imports Prophet
  (Windows laptop, 400 items: 5.8 s with 4 workers vs 11.0 s with 16).
- **Windows**: stock Prophet pays ≈ 100 ms per model there (cmdstanpy starts a `where.exe` process each time a
  model is created); prophet-batch avoids it since 0.1.1 by reusing one loaded Stan model per process, which is
  why its Windows speed-up (11.6×) is in line with the other platforms.

More: [docs/guarantees.md](https://github.com/sumedhekaru/prophet-batch/blob/main/docs/guarantees.md) (what is guaranteed and how it is tested) and
[docs/migration.md](https://github.com/sumedhekaru/prophet-batch/blob/main/docs/migration.md) (from a stock loop or process pool).

## License

MIT. Contains code derived from Prophet (Copyright (c) Facebook, Inc. and its affiliates).
