# Changelog

All notable changes to prophet-batch. Versions follow [semantic versioning](https://semver.org/); before 1.0 the public
API may still change between minor versions.

## 0.1.3 (2026-10-26)

### Added

- **`solver="batch" | "stan"`** on `fit_many`, `forecast_many` and `Pool.forecast_many`. `"batch"` (default) is
  unchanged; `"stan"` fits every series exactly as `Prophet.fit` (bit for bit, with `fit_kwargs`), with the Stan model
  loaded once per process and the faster holiday features below: 2.7× faster than stock `Prophet.fit` on a wide weekly
  multiplicative model. For models fit one series at a time where Stan beats the batched solver.
- **`prophet_batch.reuse_stan_model()`**: the context manager fit_many uses to load Prophet's Stan model once per
  process, now public, for your own loops of `Prophet.fit` calls. The private name `_reuse_stan_model` remains as an
  alias for this release.

### Performance

- `fit_many` / `forecast_many` on a single series no longer preprocess it twice (the memory estimate for automatic
  batch sizes did it once more): 1.2–1.6× faster per call on a wide weekly multiplicative model; results unchanged.
- Holiday features are built with a vectorized, bit-identical version of Prophet's `make_holiday_features` inside
  prophet-batch's calls (Prophet's own looks up every holiday date and window offset one at a time): preprocessing a
  model with 6 holidays over 22 years and ±15-day windows takes 7 ms instead of 55 ms. It is used only when the
  installed Prophet's function is the one it mirrors (1.5.0); subclasses that override it keep theirs.
- `predict_many` / `forecast_many` with `columns=` that need no individual component (`yhat`, `trend`,
  `additive_terms`, `multiplicative_terms`, their bounds, `ds`, `cap`, `floor`) compute only the additive and
  multiplicative totals, not every seasonality and holiday component: values unchanged, 20% faster predict on a model
  with 7 holiday components.

### Fixed

- `forecast_many` dropped rows with a missing series id without a word (`fit_many` already raised); it now raises the
  same `ValueError`, also for a long `futures` frame. A missing id column raises a `ValueError` naming `id_col`
  instead of a bare `KeyError`.
- An error while predicting a series (e.g. a NaN regressor value in its future) now names the series, as fit errors
  already did, in `predict_many` and `forecast_many`.

### Documentation

- README results on 0.1.2 from PyPI (`forecast_many`, five platforms, six public datasets, Spark at scale, a million
  series streamed), the comparison against stock Prophet's fastest setup, and credit to O. Matar for the lasso
  formulation of Prophet's MAP and the vectorized predict now in Prophet.
- README: an "Examples and benchmarks" section pointing to the companion repository's use cases.
- README: a link to the article on Medium.

## 0.1.2 (2026-10-22)

### Added

- **`forecast_many(df, periods=..., ...)`**: fit and forecast many series in one step, returning one long DataFrame
  with the series id first and, for each series, exactly the columns stock Prophet's
  `model.predict(model.make_future_dataframe(periods))` returns. Equal to `predict_many(fit_many(df), ...)` for each
  series and seed, bit for bit; no fitted model is kept or sent between processes. `futures=` (a long frame with the
  id column, or a dict) for regressors and caps; `return_params=True` also returns each series' `model.params`;
  `columns=`, `include_history=False` and `history_uncertainty="analytic"` trim the output without changing values.
- **`Pool(n_jobs=...)`**: worker processes kept across `forecast_many` calls, for forecasting many batches (read from
  your own files or database) without paying process start-up each time; `pool.submit(...)` returns a
  `concurrent.futures.Future`, so reading and writing can overlap with computing.
- **`batch_size`** on `fit_many` and `cross_validation_many` (and `forecast_many`): the most series a worker prepares
  and solves at once. By default chosen automatically: up to 100 series (larger batches are not faster), at least four
  tasks per worker, and small enough that the batches in flight use at most half of the available memory (container
  limits included). Results are bit-identical for any `batch_size`.

### Changed

- **Faster parallel fits.** `fit_many(n_jobs > 1)` fits in smaller batches and no longer copies the whole input before
  starting the workers: on 30,490 M5 item × store series (10 workers, Apple M4) 31.3 → 23.2 s; on 15,000 series the
  parent's memory peak during the fit 6.8 → 4.4 GB.
- **No second model build for scheduling.** The parallel path used to build every series' model once in the parent
  (to send Stan-fit series first) before starting the workers; with a `make_model` that calls `add_country_holidays`
  this doubled model construction (≈ 150 s for 30,490 series). It now runs only in parallel runs and within a 1 s
  budget: on a model with US holidays and a regressor, `fit_many(n_jobs=10)` 5.18 → 3.34 s for 1,000 series.
- `fit_many(n_jobs=1)` fits in batches too (same results), which keeps memory bounded for long series.

### Fixed

- Large parallel fits held each worker's whole share of the series at once (n / 4·workers series): memory per worker
  grew with the dataset. Workers now hold one bounded batch at a time.

## 0.1.1 (2026-10-22)

### Fixed

- **Windows speed.** cmdstanpy starts a `where.exe` process each time a Prophet model is created on Windows (≈ 100 ms
  per model), which dominated `fit_many` there. `fit_many` and `cross_validation_many` now reuse one loaded Stan model
  per process (results unchanged): on a Windows laptop (i7-1260P, 16 threads, 1,437 M5 items) 3.1× → 11.6× faster
  than a 16-process stock Prophet pool.

## 0.1.0 (2026-10-22)

First release: `fit_many` (batched exact MAP fits, at least Stan's log posterior), `predict_many` (parallel, seeded
per series), `cross_validation_many` (stock format) and `predict` (seedable, with analytic history intervals), on top
of stock `prophet` ≥ 1.5. Python 3.10–3.13 on Linux, macOS and Windows.
