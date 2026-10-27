# prophet-batch

[![PyPI](https://img.shields.io/pypi/v/prophet-batch)](https://pypi.org/project/prophet-batch/)
[![Python](https://img.shields.io/pypi/pyversions/prophet-batch)](https://pypi.org/project/prophet-batch/)
[![CI](https://github.com/sumedhekaru/prophet-batch/actions/workflows/ci.yml/badge.svg)](https://github.com/sumedhekaru/prophet-batch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/pypi/l/prophet-batch)](https://github.com/sumedhekaru/prophet-batch/blob/main/LICENSE)

**Forecast thousands to millions of series with [Prophet](https://github.com/facebook/prophet), 10–37× faster than
a stock Prophet process pool on the same cores, with the same accuracy.** prophet-batch runs on top of the stock
`prophet` package (≥ 1.5): your models stay ordinary `Prophet` objects, and your data stays in your hands.

- **`fit_many`** fits all series together with exact batched solvers instead of one Stan optimization per series.
  Every fit reaches a log posterior at least as high as Stan's (checked on every one of 38,905 public series).
- **`predict_many`** forecasts them in parallel, reproducibly: each series has its own seeded random stream.
- **`forecast_many` and `Pool`** fit and forecast in one step for large catalogs: batch after batch from your
  own files or database, with flat memory, workers kept between batches.
- **`cross_validation_many`** and **`predict`**: stock-format cross-validation for many models, and stock's predict
  with a seedable generator and faster history intervals for one.

**Read the story:** [Up to 37× faster Prophet: fitting thousands of series in one call](https://medium.com/@sumedhekaru/up-to-37-faster-prophet-fitting-thousands-of-series-in-one-call-f532fae5c9e7) (Medium): the problem, how it works, and the benchmarks.

**See it in use:** [prophet-batch-examples](https://github.com/sumedhekaru/prophet-batch-examples): runnable use cases
with accuracy and speed against stock Prophet. Walmart grocery forecasting on a laptop, in Docker, as a Cloud Run job
and on Spark (Dataproc); six public forecasting datasets; and a large CSV forecast in batches with flat memory.

## Results

1,437 Walmart grocery items (M5, FOODS, daily, 28 days ahead), default `Prophet()`, fit + forecast with
`forecast_many` (analytic history intervals), against stock Prophet run the way people parallelize it: a process pool
on every core. prophet-batch 0.1.2 from PyPI, median of 3 runs ([all tables, and the code to rerun
them](https://github.com/sumedhekaru/prophet-batch-examples/tree/main/walmart_grocery_forecasting)):

| environment | stock Prophet, process pool | prophet-batch | faster |
|---|---:|---:|---:|
| MacBook, Apple M4, 10 cores | 55.5 s | 4.1 s | **13.6×** |
| Linux container (Docker), same Mac | 59.2 s | 5.0 s | **11.5×** |
| Cloud Run job, 4 vCPU | 249.1 s | 20.4 s | **13.0×** |
| Windows laptop, Intel i7-1260P, 16 threads | 397.4 s | 23.4 s | **15.8×** |
| Spark on Dataproc Serverless, 2 × 4 cores (execution¹) | 244.1 s | 25.0 s | **9.8×** |

¹ The Spark job's execution phase (`fit_many` + `predict_many` per group); end to end 2.2×, because each batch also
spends ≈ 130 s on platform start-up, loading and warm-up, the same for both. On larger jobs the end-to-end gain grows:
6.7× on 30,490 series ([Spark results](https://github.com/sumedhekaru/prophet-batch-examples/blob/main/public_datasets/results/dataproc-us-east1.md)).

- **Same accuracy:** WAPE 0.3438 vs 0.3446, MASE 1.4261 vs 1.4279 (difference within noise), on every platform; with
  holidays and with holidays + regressors as well (10.2–13.2× there).
- **Fair comparisons:** with default settings (history intervals simulated like stock) the gain is 3.2–4.8×; against
  stock Prophet's fastest setup (future rows only, prophet-batch the same) it stays 11–39× on the public datasets
  ([page](https://github.com/sumedhekaru/prophet-batch-examples/blob/main/public_datasets/results/future-only-macos-m4.md)). Prophet ≥ 1.1.2 already includes the vectorized predict
  of O. Matar (2021), so these gains come on top of it.
- **Six public datasets** (38,905 series: M5 item and item × store, M4 Daily, Weekly, Hourly, Tourism Monthly; Apple M4,
  10 workers): **37×** on M4 Daily's long series, **15×** on 30,490 item × store series and on M5 items, 2.7–8× on the
  small ones; accuracy slightly better beyond noise on four datasets and within noise on two; fit at least as good as
  Stan's on 38,905 of 38,905 series ([results](https://github.com/sumedhekaru/prophet-batch-examples/blob/main/public_datasets/results/macos-m4.md); Linux, Cloud Run and Spark
  pages next to it). On one core, fit + forecast is 17× faster than stock: the gain is in the work, not the parallelism.
- **At scale:** 1,000,072 series streamed from one file through a `Pool` in 33 minutes on the MacBook, with ≈ 2 GB of
  memory in total, the same as for 3,000 series ([streaming example](https://github.com/sumedhekaru/prophet-batch-examples/blob/main/streaming_csv/results/macos-m4.md)).

## Examples and benchmarks

[prophet-batch-examples](https://github.com/sumedhekaru/prophet-batch-examples) is the companion repository: runnable use cases that install prophet-batch from PyPI
and report accuracy and speed against stock Prophet, with one command per environment (laptop, Docker, Cloud Run job,
Spark on Dataproc) and every result page reproducible from its run records.

| use case | what it shows |
|---|---|
| [walmart_grocery_forecasting](https://github.com/sumedhekaru/prophet-batch-examples/tree/main/walmart_grocery_forecasting) | 1,437 Walmart grocery items, three models (default, holidays, holidays + regressors), on macOS, Linux, Windows, Cloud Run and Spark |
| [public_datasets](https://github.com/sumedhekaru/prophet-batch-examples/tree/main/public_datasets) | six public datasets (38,905 series): accuracy, fit quality vs Stan, speed at scale, scaling with series and workers |
| [streaming_csv](https://github.com/sumedhekaru/prophet-batch-examples/tree/main/streaming_csv) | a file of any size forecast in batches through one `Pool` with bounded memory: a million series on a laptop |

## Why it is fast, and what you can rely on

- **Exact batched fits.** For Prophet's linear-growth model the MAP is a regularized regression (a lasso on the
  changepoints from their Laplace prior, ridge on the rest), as O. Matar noted in 2022 when approximating it; here
  that convex problem is solved exactly for many series at once (feature-sign search); multiplicative seasonality by a verified
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

Status: alpha (0.1.2): the public API (below) is stable in intent but may still change before 1.0. What changed in each
release: [CHANGELOG.md](https://github.com/sumedhekaru/prophet-batch/blob/main/CHANGELOG.md).

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
`solver="stan"` fits every series by `Prophet.fit` instead (see "Choosing the solver" below).

### What is batched

Linear and flat growth are fit by the batched solvers, with any mix of additive and multiplicative components
(seasonalities, holidays, extra regressors, conditional seasonalities). Additive models are solved exactly (the MAP
is a convex problem); multiplicative ones to a verified stationary point. Everything else (logistic growth,
`mcmc_samples > 0`, regressors with `regressor_predictor`) and any series the solver can't handle is fit by
`Prophet.fit` per series, with exactly the result of calling it directly.

### Choosing the solver

`solver="batch"` (the default) is right for most workloads: many series of the same shape are solved together, and
on every public benchmark it is several times faster than Stan. Its cost per series falls as more series of the
same length and configuration are solved together, because its dense linear algebra is shared across them. Stan's
L-BFGS never forms those matrices, so for one series at a time with many parameters it can be the faster choice:

| workload (weekly, multiplicative, 6 holidays ± 15 days, 233 parameters) | `solver="batch"` | `solver="stan"` | stock `Prophet.fit` |
|---|---|---|---|
| 32 series of one length, per series | 28 ms | 31 ms | 81 ms |
| one series alone (e.g. backtest cutoffs, each its own length) | 71 ms | 30 ms | 80 ms |

(Apple M4, 104 weekly rows, `fit_kwargs={"algorithm": "LBFGS", "iter": 1000, "tol_obj": 1e-9}`, one process.)
`solver="stan"` fits every series exactly as `Prophet.fit` does (bit for bit, with your `fit_kwargs`), loads Stan's
model once per process, and builds holiday features with prophet-batch's faster code, so it is faster than stock
Prophet too. Use it when your series are fit one at a time and the model is wide (many holiday or regressor columns
relative to the history length), or when you need Stan's exact fit. Otherwise, put more series in each call: a
catalog of items on one calendar is batched by length, so passing all items' cutoffs together helps more than
switching solvers.

```python
fc = forecast_many(df, periods=13, freq="W-SAT", make_model=make_model, solver="stan", seed=0)
```

To reuse Stan's loaded model in your own loop of `Prophet.fit` calls, wrap it in `prophet_batch.reuse_stan_model()`:

```python
import prophet_batch

with prophet_batch.reuse_stan_model():   # the compiled model is loaded once in this process
    models = {sid: make_model(sid).fit(g.drop(columns="unique_id")) for sid, g in df.groupby("unique_id")}
```

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

When the columns you ask for need no single seasonality, holiday or regressor component (`yhat`, `trend`,
`additive_terms`, `multiplicative_terms`, their bounds, `ds`, `cap`, `floor`), only the totals are computed, with the
same values.

Different horizons per series: pass each series its own future frame (every series in `df` needs one).

```python
horizon = {"store_0": 7, "store_1": 28, "store_2": 90}          # days ahead, per series
some = df[df["unique_id"].isin(horizon)]
last = some.groupby("unique_id")["ds"].max()
futures = {sid: pd.DataFrame({"ds": pd.date_range(last[sid], periods=h + 1, freq="D")[1:]})
           for sid, h in horizon.items()}
fc = forecast_many(some, futures=futures, seed=0, columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
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
`Pool`, `cross_validation_many`, `predict`, `reuse_stan_model` and `series_rng`. Everything else (the solvers and helpers in
`prophet_batch.batch`, such as `solve_map` or `log_posterior`, and the private modules) is internal and may change
between releases.

## Limitations

- Logistic growth, MCMC and regressor predictors are fit by Stan (correct, but not faster than stock).
- **Wide models fit one series at a time**: the batched solver's cost grows with the cube of the number of
  parameters, and it is shared only among series of the same length. A multiplicative weekly model with 233
  parameters fit alone takes 71 ms with the batched solver vs 30 ms with `solver="stan"`. See "Choosing the
  solver".
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
- **Apple M4 with numpy 1.x**: the OpenBLAS bundled with numpy 1.22–1.26 can hang (0% CPU) in `np.linalg.solve`
  on M4 Macs, also in Linux containers on them, with no other package involved; numpy 1.21 crashes. prophet-batch's
  solvers use it far more than stock Prophet, so they hit it first. Use numpy ≥ 2.0, or set
  `OPENBLAS_NUM_THREADS=1` before starting Python. (Seen on an M4; numpy 1.x has not been tested on other chips.)

More: [docs/guarantees.md](https://github.com/sumedhekaru/prophet-batch/blob/main/docs/guarantees.md) (what is guaranteed and how it is tested) and
[docs/migration.md](https://github.com/sumedhekaru/prophet-batch/blob/main/docs/migration.md) (from a stock loop or process pool).

## License

MIT. Contains code derived from Prophet (Copyright (c) Facebook, Inc. and its affiliates).
