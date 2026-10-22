# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""forecast_many and Pool: fit and forecast many series in one step, as compute only.

The user's code reads and writes the data; these take a long DataFrame of series and return
a long DataFrame of forecasts. Each worker fits a batch of series (fit_many) and forecasts
them (predict_many's per-series generators), so no fitted model crosses a process boundary,
and a Pool keeps its workers between calls.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Hashable, Literal, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from prophet_batch.batch import (
    _check_batch_size, _keep_columns, _kwargs_for, _plan_tasks, _predict_chunk, _resolve_n_jobs,
    _reuse_stan_model, fit_many,
)

if TYPE_CHECKING:
    from prophet.forecaster import Prophet

    from prophet_batch.batch import _FitKwargs

logger: logging.Logger = logging.getLogger('prophet_batch')

def _forecast_task(
    df: pd.DataFrame,
    id_col: str,
    make_model: Callable[[Hashable], Prophet] | None,
    prophet_kwargs: dict[str, Any],
    on_error: str,
    fit_kwargs: _FitKwargs,
    batch_size: int,
    futures: Mapping[Hashable, pd.DataFrame] | None,
    periods: int | None,
    freq: str,
    include_history: bool,
    seed: int,
    history_uncertainty: bool | Literal['analytic'],
    columns: list[str] | None,
    return_params: bool,
) -> tuple[pd.DataFrame, dict[Hashable, dict[str, np.ndarray]] | None]:
    """Worker: fit one batch of series and forecast them; one long frame for the batch."""
    models = fit_many(df, id_col=id_col, make_model=make_model, on_error=on_error,
                      fit_kwargs=fit_kwargs, batch_size=batch_size, **prophet_kwargs)
    items = []
    for sid, model in models.items():
        future = (futures[sid] if futures is not None
                  else model.make_future_dataframe(cast(int, periods), freq=freq,
                                                   include_history=include_history))
        items.append((sid, model, future))
    results = _predict_chunk(items, seed, history_uncertainty, columns)
    if results:
        out = pd.concat([fc for _, fc in results], ignore_index=True)
        ids = np.empty(len(results), dtype=object)
        ids[:] = [sid for sid, _ in results]
        out.insert(0, id_col, np.repeat(ids, [len(fc) for _, fc in results]))
    else:
        out = pd.DataFrame({id_col: []})
    out.attrs = {}
    params = {sid: model.params for sid, model in models.items()} if return_params else None
    return out, params


def _forecast(
    executor: Executor | None,
    n_jobs: int,
    df: pd.DataFrame,
    periods: int | None,
    futures: Any,
    freq: str,
    include_history: bool,
    seed: int | None,
    history_uncertainty: bool | Literal['analytic'],
    columns: str | Sequence[str] | None,
    return_params: bool,
    id_col: str,
    make_model: Callable[[Hashable], Prophet] | None,
    fit_kwargs: _FitKwargs,
    on_error: str,
    batch_size: int | None,
    prophet_kwargs: dict[str, Any],
) -> pd.DataFrame | tuple[pd.DataFrame, dict[Hashable, dict[str, np.ndarray]]]:
    """forecast_many on an executor (None: in this process), with n_jobs workers."""
    import pickle

    from prophet.forecaster import Prophet

    if (futures is None) == (periods is None):
        raise ValueError('Pass exactly one of futures or periods.')
    keep = _keep_columns(columns)
    if keep is not None:
        keep = [c for c in keep if c != id_col]
    if make_model is not None and prophet_kwargs:
        raise ValueError('Pass either make_model or Prophet keyword arguments, not both.')
    if on_error not in ('raise', 'skip'):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}.")
    _check_batch_size(batch_size)
    kwargs_for = _kwargs_for(fit_kwargs)
    if ('ds' not in df) or ('y' not in df):
        raise ValueError('Dataframe must have columns "ds" and "y" with the dates and '
                         'values respectively.')
    if make_model is None:
        Prophet(**prophet_kwargs)  # invalid configuration: raise now
    if executor is not None:
        for name, fn in (('make_model', make_model), ('fit_kwargs', fit_kwargs)):
            if fn is not None:
                try:
                    pickle.dumps(fn)
                except Exception as e:
                    raise ValueError(f'With n_jobs != 1, {name} must be picklable (a module-level '
                                     'function, not a lambda or closure).') from e
    if seed is None:
        seed = cast(int, np.random.SeedSequence().entropy)
        logger.info('forecast_many: no seed given, using seed=%d', seed)
    future_by_id: Mapping[Hashable, pd.DataFrame] | None = None
    if isinstance(futures, pd.DataFrame):
        if id_col not in futures:
            raise ValueError(f'futures must have the id column {id_col!r}.')
        future_by_id = {sid: g.drop(columns=id_col).reset_index(drop=True)
                        for sid, g in futures.groupby(id_col, sort=False)}
    elif futures is not None:
        future_by_id = futures

    rows_by_id, ids, stan_ids, tasks, note = _plan_tasks(
        df, id_col, make_model, prophet_kwargs, kwargs_for, batch_size, n_jobs)
    if future_by_id is not None:
        missing = [sid for sid in ids if sid not in future_by_id]
        if missing:
            raise ValueError(f'futures has no future for series {missing[:5]}.')

    def task_args(task: list[Hashable]) -> tuple:
        chunk = df.iloc[np.sort(np.concatenate([rows_by_id[sid] for sid in task]))]
        sub = None if future_by_id is None else {sid: future_by_id[sid] for sid in task}
        return (chunk, id_col, make_model, prophet_kwargs, on_error, fit_kwargs, len(task), sub,
                periods, freq, include_history, seed, history_uncertainty, keep, return_params)

    parts: list[tuple[pd.DataFrame, dict | None]] = []
    if executor is None:
        for task in tasks:
            parts.append(_forecast_task(*task_args(task)))
    else:
        # Tasks are sliced as they are submitted, at most 2 * n_jobs at a time; results are
        # collected in task order.
        pending: deque[Future] = deque()
        for task in tasks:
            pending.append(executor.submit(_forecast_task, *task_args(task)))
            if len(pending) >= 2 * n_jobs:
                parts.append(pending.popleft().result())
        while pending:
            parts.append(pending.popleft().result())
    frames = [f for f, _ in parts if len(f)]
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame({id_col: pd.Series([], dtype=object)}))
    if stan_ids and len(out):
        # Stan-fit series were sent first (scheduling): put every series' rows back in order
        # of first appearance in df (rows of a series are contiguous: move whole blocks).
        col = out[id_col].to_numpy()
        starts = np.flatnonzero(np.r_[True, col[1:] != col[:-1]])
        ends = np.r_[starts[1:], len(col)]
        rank = {sid: i for i, sid in enumerate(ids)}
        order = sorted(range(len(starts)), key=lambda b: rank[cast(Hashable, col[starts[b]])])
        out = out.iloc[np.concatenate([np.arange(starts[b], ends[b]) for b in order])]
        out = out.reset_index(drop=True)
    done = set(out[id_col].unique()) if len(out) else set()
    logger.info('forecast_many: %d series (%d via Prophet.fit) in %d tasks on %d worker '
                'process(es) (%s), %d skipped', len(ids), len(stan_ids), len(tasks), n_jobs,
                note or 'Prophet.fit only', len(ids) - len(done))
    out.attrs['seed'] = seed
    if not return_params:
        return out
    params: dict[Hashable, dict[str, np.ndarray]] = {}
    for _, p in parts:
        params.update(cast(dict, p))
    return out, {sid: params[sid] for sid in ids if sid in params}


def _spawn_executor(n_jobs: int) -> Executor:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context('spawn'),
                               initializer=_warm_worker)


def _warm_worker() -> None:
    """Pool initializer: import Prophet once per worker, before the first task arrives."""
    import prophet.forecaster  # noqa: F401


@_reuse_stan_model()
def forecast_many(
    df: pd.DataFrame,
    periods: int | None = None,
    futures: pd.DataFrame | Mapping[Hashable, pd.DataFrame] | None = None,
    freq: str = 'D',
    include_history: bool = True,
    seed: int | None = None,
    n_jobs: int = 1,
    history_uncertainty: bool | Literal['analytic'] = True,
    columns: str | Sequence[str] | None = None,
    return_params: bool = False,
    id_col: str = 'unique_id',
    make_model: Callable[[Hashable], Prophet] | None = None,
    fit_kwargs: _FitKwargs = None,
    on_error: str = 'raise',
    batch_size: int | None = None,
    **prophet_kwargs: Any,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[Hashable, dict[str, np.ndarray]]]:
    """Fit and forecast many series in one step, returning one long DataFrame of forecasts.

    For each series, the forecast is what stock Prophet gives for
    `model.predict(model.make_future_dataframe(periods, freq, include_history))` (all
    columns, history rows included), with the series id in the first column; bit-identical
    to predict_many(fit_many(df, ...), ..., seed=seed) for that series, for any n_jobs,
    batch_size, split of the input into calls, and order. No fitted model is kept or sent
    between processes, so memory is bounded by the batches in flight and the output.

    Parameters
    ----------
    df: Long-format DataFrame with columns `id_col`, ds, y (and any regressor columns).
    periods, freq, include_history: Build each future with
        model.make_future_dataframe(periods, freq, include_history). Give periods or futures.
    futures: The future frames instead: a long DataFrame with `id_col` (ds and any
        regressor, cap or floor columns) or a dict mapping series id to its future frame.
    seed: Integer seed for the uncertainty intervals, as in predict_many; None draws one,
        which is logged and stored in the result's `attrs['seed']`.
    n_jobs: Number of worker processes; 1 runs in this process, -1 uses all available
        CPUs (respecting a container's CPU limit). Use Pool to keep workers across calls.
    history_uncertainty, columns: As in predict_many (columns: the forecast columns to keep;
        the id column is always kept).
    return_params: Also return each series' fitted parameters (model.params), as a dict
        mapping series id to the parameter dict.
    id_col, make_model, fit_kwargs, on_error, batch_size, prophet_kwargs: As in fit_many.
        Skipped series (on_error='skip') are absent from the result.

    Returns
    -------
    The forecasts as one DataFrame (series in order of first appearance in df), or
    (forecasts, params) with return_params=True. When series have different components
    (e.g. holidays for some only), the frame has every series' columns, NaN where a
    component doesn't apply to a series.
    """
    n_jobs = _resolve_n_jobs(n_jobs)
    args = (df, periods, futures, freq, include_history, seed, history_uncertainty, columns,
            return_params, id_col, make_model, fit_kwargs, on_error, batch_size, prophet_kwargs)
    if n_jobs == 1:
        return _forecast(None, 1, *args)
    executor = _spawn_executor(n_jobs)
    try:
        return _forecast(executor, n_jobs, *args)
    finally:
        executor.shutdown(wait=True)


class Pool:
    """Worker processes kept across forecast_many calls, for many batches of series.

        with Pool(n_jobs=-1) as pool:
            for batch in my_reader():
                my_writer(pool.forecast_many(batch, periods=28))

    Workers start (and import Prophet) once, when the pool is entered or first used.
    forecast_many takes the same arguments as the function of that name, without n_jobs;
    submit returns a concurrent.futures.Future of the same result, so the caller can read
    the next batch or write the last one while this one is computed. close() (or leaving
    the with block) waits for running calls and stops the workers.
    """

    def __init__(self, n_jobs: int = -1) -> None:
        self.n_jobs = _resolve_n_jobs(n_jobs)
        self._executor: Executor | None = None
        self._threads: ThreadPoolExecutor | None = None
        self._closed = False
        self._lock = threading.Lock()

    def __enter__(self) -> Pool:
        self._start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError('This Pool is closed.')
            if self._threads is None:
                self._threads = ThreadPoolExecutor(max_workers=4, thread_name_prefix='prophet_batch')
                if self.n_jobs > 1:
                    self._executor = _spawn_executor(self.n_jobs)

    @_reuse_stan_model()
    def forecast_many(
        self,
        df: pd.DataFrame,
        periods: int | None = None,
        futures: pd.DataFrame | Mapping[Hashable, pd.DataFrame] | None = None,
        freq: str = 'D',
        include_history: bool = True,
        seed: int | None = None,
        history_uncertainty: bool | Literal['analytic'] = True,
        columns: str | Sequence[str] | None = None,
        return_params: bool = False,
        id_col: str = 'unique_id',
        make_model: Callable[[Hashable], Prophet] | None = None,
        fit_kwargs: _FitKwargs = None,
        on_error: str = 'raise',
        batch_size: int | None = None,
        **prophet_kwargs: Any,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[Hashable, dict[str, np.ndarray]]]:
        """forecast_many on this pool's workers (see prophet_batch.forecast_many)."""
        self._start()
        return _forecast(self._executor, self.n_jobs, df, periods, futures, freq, include_history,
                         seed, history_uncertainty, columns, return_params, id_col, make_model,
                         fit_kwargs, on_error, batch_size, prophet_kwargs)

    def submit(self, df: pd.DataFrame, **kwargs: Any) -> Future:
        """Start forecast_many(df, **kwargs) and return a Future of its result."""
        self._start()
        return cast(ThreadPoolExecutor, self._threads).submit(self.forecast_many, df, **kwargs)

    def close(self) -> None:
        """Wait for running calls, then stop the workers. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            threads, executor = self._threads, self._executor
        if threads is not None:
            threads.shutdown(wait=True)
        if executor is not None:
            executor.shutdown(wait=True)
