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

from prophet_batch._holidays import fast_holidays
from prophet_batch.batch import (
    _check_batch_size, _check_ids, _check_solver, _keep_columns, _kwargs_for, _plan_tasks, _predict_chunk,
    _resolve_n_jobs, fit_many, reuse_stan_model,
)

if TYPE_CHECKING:
    from prophet.forecaster import Prophet

    from prophet_batch.batch import _FitKwargs

logger: logging.Logger = logging.getLogger('prophet_batch')

class _Part:
    """One task's forecasts, column by column: each column one 1-D array over the task's
    rows (series after series), so the parent can assemble the output a column at a time
    and free each task's copy of it as it goes (peak ≈ the output plus one column)."""

    def __init__(self, ids: list[Hashable], lens: list[int], cols: dict[str, np.ndarray]) -> None:
        self.ids, self.lens, self.cols = ids, lens, cols

    @classmethod
    def of(cls, results: list[tuple[Hashable, pd.DataFrame]]) -> _Part:
        names = list(dict.fromkeys(c for _, fc in results for c in fc.columns))
        cols = {}
        for c in names:
            cols[c] = np.concatenate([fc[c].to_numpy() if c in fc else np.full(len(fc), np.nan)
                                      for _, fc in results]) if results else np.array([])
        return cls([sid for sid, _ in results], [len(fc) for _, fc in results], cols)


class _Assembler:
    """Builds the output frame while tasks finish: each output column is allocated once (row
    counts per series are known up front), and every task's arrays are copied into place and
    dropped as soon as the task returns, so the peak is the output plus the tasks in flight.

    A series whose row count differs from the expected one (or that was skipped) is handled
    at the end with one extra copy: correct output either way.
    """

    def __init__(self, order: list[Hashable], expected: Mapping[Hashable, int], id_col: str) -> None:
        self.order, self.expected, self.id_col = order, expected, id_col
        self.offset: dict[Hashable, int] = {}
        pos = 0
        for sid in order:
            self.offset[sid] = pos
            pos += expected[sid]
        self.n = pos
        self.cols: dict[str, np.ndarray] = {}
        self.filled: set[Hashable] = set()
        self.extra: dict[Hashable, dict[str, np.ndarray]] = {}

    def _column(self, dtype: np.dtype) -> np.ndarray:
        if dtype.kind == 'f':
            return np.full(self.n, np.nan, dtype=dtype)
        if dtype.kind in 'mM':
            return np.full(self.n, np.array('NaT', dtype=dtype), dtype=dtype)
        return np.full(self.n, None, dtype=object)

    def add(self, part: _Part) -> None:
        start = 0
        for sid, n in zip(part.ids, part.lens):
            rows = slice(start, start + n)
            start += n
            if n != self.expected.get(sid):
                self.extra[sid] = {c: a[rows].copy() for c, a in part.cols.items()}
                continue
            at = self.offset[sid]
            for c, a in part.cols.items():
                col = self.cols.get(c)
                if col is None:
                    col = self.cols[c] = self._column(a.dtype)
                col[at:at + n] = a[rows]
            self.filled.add(sid)

    def finish(self) -> pd.DataFrame:
        done = [sid for sid in self.order if sid in self.filled or sid in self.extra]
        ids = np.empty(len(done), dtype=object)
        ids[:] = done
        if not self.extra and len(done) == len(self.order):
            out = {self.id_col: np.repeat(ids, [self.expected[sid] for sid in done]), **self.cols}
            return pd.DataFrame(out, copy=False)
        # skipped series, or row counts that differ from the expected: one copy, in order
        for extra in self.extra.values():
            for c, a in extra.items():
                if c not in self.cols:
                    self.cols[c] = self._column(a.dtype)
        lens = [len(next(iter(self.extra[sid].values()))) if sid in self.extra else self.expected[sid]
                for sid in done]
        out = {self.id_col: np.repeat(ids, lens)}
        for c, col in self.cols.items():
            pieces = []
            for sid, n in zip(done, lens):
                if sid in self.extra:
                    a = self.extra[sid].get(c)
                    pieces.append(a if a is not None else self._column(col.dtype)[:n])
                else:
                    pieces.append(col[self.offset[sid]:self.offset[sid] + n])
            out[c] = np.concatenate(pieces) if pieces else col[:0]
        return pd.DataFrame(out, copy=False)


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
    solver: str = 'batch',
) -> tuple[_Part, dict[Hashable, dict[str, np.ndarray]] | None]:
    """Worker: fit one batch of series and forecast them; the batch's forecasts as one _Part."""
    models = fit_many(df, id_col=id_col, make_model=make_model, on_error=on_error,
                      fit_kwargs=fit_kwargs, batch_size=batch_size, solver=solver,
                      **prophet_kwargs)
    items = []
    for sid, model in models.items():
        future = (futures[sid] if futures is not None
                  else model.make_future_dataframe(cast(int, periods), freq=freq,
                                                   include_history=include_history))
        items.append((sid, model, future))
    results = _predict_chunk(items, seed, history_uncertainty, columns)
    params = {sid: model.params for sid, model in models.items()} if return_params else None
    return _Part.of(results), params


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
    solver: str = 'batch',
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
    _check_solver(solver)
    kwargs_for = _kwargs_for(fit_kwargs)
    if ('ds' not in df) or ('y' not in df):
        raise ValueError('Dataframe must have columns "ds" and "y" with the dates and '
                         'values respectively.')
    _check_ids(df, id_col)
    if isinstance(futures, pd.DataFrame):
        _check_ids(futures, id_col, 'futures')
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
        future_by_id = {sid: g.drop(columns=id_col).reset_index(drop=True)
                        for sid, g in futures.groupby(id_col, sort=False)}
    elif futures is not None:
        future_by_id = futures

    rows_by_id, ids, stan_ids, tasks, note = _plan_tasks(
        df, id_col, make_model, prophet_kwargs, kwargs_for, batch_size, n_jobs, solver)
    if future_by_id is not None:
        missing = [sid for sid in ids if sid not in future_by_id]
        if missing:
            raise ValueError(f'futures has no future for series {missing[:5]}.')

    def task_args(task: list[Hashable]) -> tuple:
        chunk = df.iloc[np.sort(np.concatenate([rows_by_id[sid] for sid in task]))]
        sub = None if future_by_id is None else {sid: future_by_id[sid] for sid in task}
        return (chunk, id_col, make_model, prophet_kwargs, on_error, fit_kwargs, len(task), sub,
                periods, freq, include_history, seed, history_uncertainty, keep, return_params,
                solver)

    # Rows per series in the output: its future frame, or its history rows (make_future_dataframe
    # takes the distinct history dates: the same unless dates repeat, which the assembler handles)
    # plus the periods.
    if future_by_id is not None:
        expected = {sid: len(future_by_id[sid]) for sid in ids}
    else:
        expected = {sid: (len(rows_by_id[sid]) if include_history else 0) + cast(int, periods)
                    for sid in ids}
    assembler = _Assembler(ids, expected, id_col)
    params: dict[Hashable, dict[str, np.ndarray]] = {}

    def collect(result: tuple[_Part, dict | None]) -> None:
        part, p = result
        assembler.add(part)
        if p is not None:
            params.update(p)

    if executor is None:
        for task in tasks:
            collect(_forecast_task(*task_args(task)))
    else:
        # Tasks are sliced as they are submitted, at most 2 * n_jobs at a time; results are
        # copied into the output as they arrive, in task order.
        pending: deque[Future] = deque()
        for task in tasks:
            pending.append(executor.submit(_forecast_task, *task_args(task)))
            if len(pending) >= 2 * n_jobs:
                collect(pending.popleft().result())
        while pending:
            collect(pending.popleft().result())
    out = assembler.finish()
    del assembler
    done = set(out[id_col].unique()) if len(out) else set()
    logger.info('forecast_many: %d series (%d via Prophet.fit) in %d tasks on %d worker '
                'process(es) (%s), %d skipped', len(ids), len(stan_ids), len(tasks), n_jobs,
                note or 'Prophet.fit only', len(ids) - len(done))
    out.attrs['seed'] = seed
    if not return_params:
        return out
    return out, {sid: params[sid] for sid in ids if sid in params}


def _spawn_executor(n_jobs: int) -> Executor:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context('spawn'),
                               initializer=_warm_worker)


def _warm_worker() -> None:
    """Pool initializer: import Prophet once per worker, before the first task arrives."""
    import prophet.forecaster  # noqa: F401


@fast_holidays()
@reuse_stan_model()
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
    solver: str = 'batch',
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
    id_col, make_model, fit_kwargs, on_error, batch_size, solver, prophet_kwargs: As in
        fit_many.
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
            return_params, id_col, make_model, fit_kwargs, on_error, batch_size, prophet_kwargs,
            solver)
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

    @fast_holidays()
    @reuse_stan_model()
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
        solver: str = 'batch',
        **prophet_kwargs: Any,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[Hashable, dict[str, np.ndarray]]]:
        """forecast_many on this pool's workers (see prophet_batch.forecast_many)."""
        self._start()
        return _forecast(self._executor, self.n_jobs, df, periods, futures, freq, include_history,
                         seed, history_uncertainty, columns, return_params, id_col, make_model,
                         fit_kwargs, on_error, batch_size, prophet_kwargs, solver)

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
