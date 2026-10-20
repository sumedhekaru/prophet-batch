"""C8.2: fit_many(n_jobs > 1) is bit-identical to n_jobs=1, across every fitting path."""

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from api import fit_many

from synthetic import make_series


def make_model(sid):
    """Module-level (picklable): the config depends on the series id."""
    if sid.startswith("logistic"):
        return Prophet(growth="logistic")
    if sid.startswith("mult"):
        return Prophet(seasonality_mode="multiplicative")
    return Prophet()


def _frames():
    frames = {}
    for i in range(6):
        frames[f"add{i}"] = make_series(seed=i)                       # shared dates
    for i in range(3):
        frames[f"mult{i}"] = make_series(n_days=600, seed=10 + i).assign(y=lambda d: d["y"] + 20)
    frames["const"] = make_series(n_days=300).assign(y=3.0)
    lg = make_series(n_days=500, seed=20)
    frames["logistic0"] = lg.assign(cap=lg["y"].max() * 1.2)
    frames["short"] = make_series(n_days=250, seed=21)
    return frames


def _long(frames):
    return pd.concat([f.assign(unique_id=k) for k, f in frames.items()], ignore_index=True)


def _assert_identical(a, b):
    assert list(a) == list(b)
    for sid in a:
        assert a[sid].params.keys() == b[sid].params.keys(), sid
        for k in a[sid].params:
            np.testing.assert_array_equal(a[sid].params[k], b[sid].params[k], err_msg=f"{sid}.{k}")


def test_make_model_paths_identical():
    df = _long(_frames())
    _assert_identical(fit_many(df, make_model=make_model), fit_many(df, make_model=make_model, n_jobs=4))


def test_shared_dates_path_identical():
    frames = {k: v for k, v in _frames().items() if not k.startswith(("logistic", "mult"))}
    df = _long(frames)
    _assert_identical(fit_many(df), fit_many(df, n_jobs=3))


def test_lambda_make_model_rejected():
    with pytest.raises(ValueError, match="picklable"):
        fit_many(_long({"a": make_series()}), make_model=lambda sid: Prophet(), n_jobs=2)


def test_parallel_models_predict():
    df = _long(_frames())
    models = fit_many(df, make_model=make_model, n_jobs=2)
    fc = models["mult1"].predict(models["mult1"].make_future_dataframe(7))
    assert fc["yhat"].notna().all()


def test_result_independent_of_batch_mates():
    frames = _frames()
    together = fit_many(_long(frames), make_model=make_model)
    for sid in ("short", "mult1", "add0"):
        alone = fit_many(_long({sid: frames[sid]}), make_model=make_model)
        _assert_identical({sid: together[sid]}, alone)


def test_stan_series_are_split_out_for_scheduling():
    """E3.2: the parent puts Stan-fit series in their own tasks; a make_model error is left to the worker."""
    from api import _split_stan_series

    ids = ["add0", "logistic0", "mult0", "logistic1"]
    assert _split_stan_series(ids, make_model, {}) == (["logistic0", "logistic1"], ["add0", "mult0"])
    assert _split_stan_series(ids, None, {"growth": "logistic"}) == (ids, [])
    assert _split_stan_series(ids, None, {}) == ([], ids)

    def broken(sid):
        raise RuntimeError("no model")

    assert _split_stan_series(["x"], broken, {}) == ([], ["x"])


def test_bunched_stan_series_identical_for_any_n_jobs():
    frames = _frames()
    order = ["logistic0"] + [k for k in frames if k != "logistic0"]  # Stan series first, contiguous
    df = _long({k: frames[k] for k in order})
    _assert_identical(fit_many(df, make_model=make_model), fit_many(df, make_model=make_model, n_jobs=3))
