# 0.1.2: forecast_many and Pool (design)

## Why

Real catalogs reach millions of series; users read their data in large batches from files or databases, forecast,
and write the results out, with the three running at once. prophet-batch should be the compute engine inside any such
flow: DataFrame in, DataFrame out, no reading or writing, no hidden state, and nothing that forces a data layout on the
user. Today the large-scale path is `fit_many` → models in the parent → `predict_many` → models out to workers again:
on M5 item × store (30,490 series) the parent holds 3.7 GB of models (mostly history frames) only to send them back
out, and every call starts a new process pool (Prophet import per worker: 0.6 s on an M4, ≈ 5 s on Dataproc).

## Decisions (user, 2026-10-22)

- Compute only; defaults follow stock Prophet; opt-outs only for speed and memory; no streaming helper; standard
  `concurrent.futures`; names `forecast_many`, `Pool` (`.forecast_many`, `.submit`), `return_params`.

## API

```python
forecast_many(df, periods=None, futures=None, freq='D', include_history=True, seed=None, n_jobs=1,
              history_uncertainty=True, columns=None, return_params=False, id_col='unique_id',
              make_model=None, fit_kwargs=None, on_error='raise', batch_size=None, **prophet_kwargs)
    -> pd.DataFrame                                   # or (forecasts, params) with return_params=True

with Pool(n_jobs=-1) as pool:                         # workers started once (Prophet imported up front)
    pool.forecast_many(df, periods=28, ...)           # same arguments as forecast_many, without n_jobs
    pool.submit(df, periods=28, ...)                  # concurrent.futures.Future of the same result
```

- **Output** (default = stock): one long DataFrame: `id_col` first, then exactly the columns
  `model.predict(model.make_future_dataframe(periods, freq, include_history))` returns for each series (19 for default
  `Prophet()`), series in order of first appearance in `df`, rows in date order. `attrs['seed']` holds the seed.
- **Equal to the two-step path:** a series' rows equal `predict_many(fit_many(df, ...), periods, seed=seed, ...)`'s
  frame for it, bit for bit, for any `n_jobs`, `batch_size`, pool, batch split of the input and order.
- **Opt-outs** (values unchanged): `columns=` (e.g. `['ds', 'yhat', 'yhat_lower', 'yhat_upper']`; `id_col` always
  kept), `include_history=False`, `history_uncertainty='analytic'` / `False`.
- **`return_params=True`:** also returns `{series id: model.params}` (stock's fitted-parameter dict of arrays) so users
  can keep or explain fits without keeping models.
- **Regressors / caps:** `futures` as a long DataFrame with `id_col` (rows: the future dates, columns: ds, regressors,
  cap/floor) or a dict `{series id: future frame}`; give `futures` or `periods`, as in predict_many.
- **Configuration:** `make_model`, `fit_kwargs`, `**prophet_kwargs`, `on_error`, `batch_size` exactly as fit_many;
  skipped series are absent from the output (warning, as fit_many).
- **Seed:** as predict_many (`None` draws one and logs it). Within a `Pool`, each call takes its own seed.

## How

- A task = one batch of consecutive series (fit_many's batch planner, the same automatic size). The worker runs
  `fit_many(task, n_jobs=1)`, then predicts the task's models (predict_many's per-series generators and feature cache),
  and sends back **one** long frame for the task (plus params if asked): no models cross process boundaries.
- Tasks are sliced and submitted at most 2·W at a time; results are concatenated in task order.
- `n_jobs=1`: the same loop in this process, no pool.
- `Pool`: a spawn `ProcessPoolExecutor` created on entry, with an initializer that imports prophet and prophet_batch;
  `.forecast_many` submits its tasks to it; `.submit` runs `.forecast_many` on a small coordinating thread and returns
  its Future (several submits may be in flight; their tasks share the workers). `close()` / leaving the `with` block
  shuts the workers down; using a closed pool raises.

## Tests

Stock equality per series (default columns and values: point forecasts equal stock's `predict` on the same fitted
model; intervals equal predict_many's seeded ones); equality with fit_many + predict_many (additive, multiplicative,
holidays + regressors via futures, Stan-fit logistic); independence of `n_jobs`, `batch_size`, input split and order;
opt-outs change no values; `return_params` equals each model's params; `on_error='skip'`; validation (futures vs
periods, columns); `Pool` reuse across calls, `submit` futures, concurrent submits, closed-pool error.

## Steps

- [ ] 1. `forecast_many` (in-process and parallel) + tests.
- [ ] 2. `Pool` + tests.
- [ ] 3. Measure on M5 item × store vs fit_many + predict_many (parent peak, time; default and trimmed columns).
- [ ] 4. README (the pattern, the opt-outs and what they save), guarantees.md.
