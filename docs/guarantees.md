# Guarantees and limitations

What prophet-batch promises, the tolerance of each promise, and the test that checks it (paths relative to this
repository; the suite runs on stock PyPI prophet in CI on Linux, macOS and Windows, Python 3.10–3.13, on every push to main).

## Same results as stock Prophet where nothing changes

- **`prophet_batch.predict` with default arguments is bit-identical to `model.predict(df)`**, for every model
  kind (linear, multiplicative, logistic, flat, holidays + regressors, regressor predictors, non-vectorized).
  `tests/test_predict.py::test_default_is_stock_predict`
- **Series that are not batched are fit by `Prophet.fit`, with exactly its result** (logistic growth,
  `mcmc_samples > 0`, regressor predictors, and any series the batched solver rejects).
  `tests/test_fit_many.py::TestFallbackAndEdgeCases`, `tests/test_properties.py::test_fallback_and_constant_match_prophet_fit`
- **The stock API works on fitted models**: all 41 model-dependent public callables of prophet 1.5.0 (predict,
  sampling, plots, cross-validation, metrics, serialization, utilities) give output bit-identical to a stock model
  holding the same parameters. `tests/test_api_parity.py::test_api_matches_stock`
- **Shared work does not change results**: series with identical models, dates and regressor values share their
  preprocessing in `fit_many` and their feature matrices in `predict_many`; fits and forecasts are bit-identical to
  doing it per series. `tests/test_fit_many.py::TestSharedDates`, `tests/test_predict_many.py::TestFeatureCache`
- **`solver="stan"` is `Prophet.fit`**: every series is fit by `make_model(sid).fit(series_df, **fit_kwargs)`, bit for
  bit, for any `n_jobs` and `batch_size`. `tests/test_solver.py`
- **Faster holiday features, same values**: inside prophet-batch's calls, Prophet's `make_holiday_features` is replaced
  by a vectorized version with bit-identical output (columns, order, prior scales, errors), only when the installed
  function's source is the one it mirrors (prophet 1.5.0); otherwise Prophet's own is used.
  `tests/test_holidays.py`
- **`columns=` does not change values**: when the requested columns need no single component, only the component
  totals are computed; the requested columns are bit-identical to the full forecast's, for every model kind and
  history mode. `tests/test_predict_many.py::TestColumns::test_totals_only_equals_full_forecast`
- **Invalid series behave as in `Prophet.fit`**: the same error (with the series id in the message), or, with
  `on_error="skip"`, the series is left out with a warning. `tests/test_edge_cases.py::test_same_behavior_as_prophet_fit`,
  `tests/test_fit_many.py::TestOnError`

## Batched fits are verified optima

- **Additive models: the exact MAP.** The posterior mode is a convex problem (least squares with ridge and lasso
  penalties, sigma in closed form); every fit is checked against the optimality (KKT) conditions, tolerance
  max(1e-8, 10 × the float64 floor). `tests/test_properties.py::test_additive_is_exact_optimum`
- **Multiplicative models: a verified stationary point** (block KKT ≤ 1e-3, checked on every fit); a fit that
  does not reach it goes to `Prophet.fit`. `tests/test_properties.py::test_multiplicative_is_stationary`,
  `tests/test_multiplicative.py::test_converges_to_stationary_point_better_than_stan`
- **Log posterior at least Stan's** (−1e-6 additive, −1e-4 multiplicative), on random configurations at daily,
  hourly, weekly and monthly frequency, and on every one of 38,905 series of six public datasets.
  `tests/test_properties.py::test_log_posterior_at_least_stan`
- **No degenerate fits**: a series that can be fit exactly (no finite MAP; sigma would reach 0) goes to
  `Prophet.fit`. `tests/test_properties.py::test_accepted_fits_have_sigma_above_floor`,
  `tests/test_fit_many.py::TestFallbackAndEdgeCases`
- **One bad series never aborts the call**: a solver failure (singular system, non-finite or unconverged
  solution) is isolated to its series, which is fit by `Prophet.fit`, on the additive and multiplicative paths.
  `tests/test_fit_many.py::TestSolverFailures`
- **Input problems raise up front, the same way for any `n_jobs`**: missing `ds`/`y` columns (Prophet's own error),
  rows without a series id, an invalid Prophet configuration. `tests/test_fit_many.py::TestInputValidation`

## Reproducibility

- **`fit_many` results are bit-identical for any `n_jobs`, series order and batch composition** (a series' fit
  depends only on its own data and configuration). `tests/test_properties.py::test_fit_independent_of_n_jobs`,
  `tests/test_properties.py::test_fit_independent_of_order`, `tests/test_properties.py::test_fit_independent_of_batch_mates`
- **`fit_many` and `cross_validation_many` results are bit-identical for any `batch_size`** (given, automatic, or
  memory-limited automatic), in one process and in parallel; output order and skipped series are unchanged.
  `tests/test_batch_size.py::test_in_process_identical_for_any_batch_size`,
  `tests/test_batch_size.py::test_parallel_identical_for_any_batch_size`,
  `tests/test_batch_size.py::test_cross_validation_identical_for_batch_size`,
  `tests/test_batch_size.py::TestPlan::test_automatic_memory_limited`
- **`forecast_many` equals `predict_many(fit_many(df), ...)` per series**, bit for bit, for any `n_jobs`,
  `batch_size`, split of the input into calls, order and `Pool`; its default columns and point forecasts are stock
  Prophet's. `tests/test_forecast_many.py::test_equals_fit_many_then_predict_many`,
  `tests/test_forecast_many.py::test_default_is_stock_format_and_values`,
  `tests/test_forecast_many.py::test_independent_of_input_split`,
  `tests/test_forecast_many.py::TestPool::test_reuse_submit_and_concurrent_calls`
- **`predict_many(seed=...)`: a forecast depends only on (seed, series id)**, not on `n_jobs`, order or which other
  series are predicted; the same in another process. `tests/test_predict_many.py::test_same_results_for_any_n_jobs`,
  `tests/test_predict_many.py::test_independent_of_order_and_subset`, `tests/test_series_rng.py::test_same_in_another_process`
- **Without a seed**, a fresh one is drawn and stored in `attrs["seed"]`; passing it back reproduces the run.
  `tests/test_predict_many.py::test_seed_none_is_recorded_and_reproducible`
- **`cross_validation_many`**: stock output format; Stan-fit series identical to stock `cross_validation`; seeded
  and independent of order and n_jobs. `tests/test_cv_many.py::test_same_format_as_stock_and_stan_series_identical`,
  `tests/test_cv_many.py::test_seeded_and_independent_of_order_and_grouping`, `tests/test_cv_many.py::test_parallel_matches_serial`
- Bit-identity holds on one machine and software stack. Across platforms, results agree to floating-point
  rounding (BLAS and libm differ in the last bits).

## History intervals

- **`history_uncertainty=False`**: history intervals NaN; future intervals exactly those of predicting the future
  rows alone. `tests/test_history_uncertainty.py::test_future_intervals_equal_future_only_predict`
- **`history_uncertainty="analytic"`** (MAP fits; MCMC raises): history bounds are `yhat ± z·sigma_obs·y_scale` and
  trend bounds equal the trend, the exact limit of the sampled intervals (checked against 50,000-sample runs to
  within 5 Monte Carlo standard errors); future rows and all other columns equal `False`'s.
  `tests/test_history_uncertainty.py::test_analytic_history_and_future_equal_false`,
  `tests/test_history_uncertainty.py::test_analytic_is_the_limit_of_sampling`,
  `tests/test_history_uncertainty.py::test_analytic_rejects_mcmc`
- They differ from the default 1,000-sample bounds by the default's own Monte Carlo noise (rms ≈ 0.06·sigma at
  80%).

## The package on stock Prophet

- **The seeded sampling is the fork's code**: `prophet_batch._seeded` is generated from the prophet-dev fork's
  `forecaster.py`, and its forecasts equal the fork's for the same seed in every scenario and history mode.
  `tests/test_predict.py::test_seeded_matches_fork`,
  `tests/test_predict.py::test_generated_module_is_in_sync_with_fork`
- **Prophet's code is never modified on disk, and models are left alone**: seeded predictions use a subclass view
  of the model. `tests/test_predict.py::test_seeded_view_shares_state_and_leaves_model_alone`. Two class attributes
  are swapped only while a prophet-batch call (or `reuse_stan_model()`) runs, and restored on exit, also on errors:
  `CmdStanPyBackend.load_model` (one loaded Stan model per process; `tests/test_stan_model_reuse.py`) and
  `Prophet.make_holiday_features` (the bit-identical vectorized version; subclasses that override it keep theirs;
  `tests/test_holidays.py`). Other threads using Prophet during such a call see the same results.

## Limitations

- **Not batched**: logistic growth, MCMC, regressor predictors. They are fit by Stan (correct, not faster than stock).
- **Wide models fit one series at a time** can be faster with `solver="stan"` (README, "Choosing the solver"); the
  default is not switched automatically, because a series' solver must not depend on its batch mates.
- **Slightly narrower intervals**: on six public datasets, 80% intervals covered 0.04–0.6 percentage points less than
  stock's. The exact MAP fits a little tighter (smaller sigma_obs, sparser trend changes) than Stan's L-BFGS, which
  stops short of the optimum. Point accuracy is the same (holdout MASE within ±1.3%).
- **Forecasts differ slightly from stock** for batched series, because the fit is the true optimum rather than
  Stan's approximation of it; the log posterior is never lower.
- **Small workloads**: with few short series, worker start-up dominates (Tourism Monthly, 366 series of ~300 rows:
  1.5× vs stock); use `n_jobs=1` there.
- **Prophet subclasses that override `predict`**: `prophet_batch.predict` (and so `predict_many`) uses Prophet's
  predict logic with seeded sampling, not the override, and warns.
  `tests/test_predict.py::test_subclass_predict_override_warns`
- **`n_jobs != 1` uses the `spawn` start method**: `make_model` and `fit_kwargs` callables must be picklable
  (module-level functions).
