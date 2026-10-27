"""0.1.3: solver='stan' fits every series exactly as Prophet.fit; solver='batch' is the default."""

import warnings

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from api import fit_many

import prophet_batch
from synthetic import make_series

FIT_KW = {"algorithm": "LBFGS", "iter": 1000, "tol_obj": 1e-9}
HOLIDAYS = pd.DataFrame({
    "holiday": ["a"] * 3 + ["b"] * 3,
    "ds": pd.to_datetime(["2019-07-04", "2020-07-04", "2021-07-04", "2019-12-25", "2020-12-25", "2021-12-25"]),
    "lower_window": [-3] * 6, "upper_window": [3] * 6,
})


def make_model(sid):
    if sid.startswith("mult"):
        return Prophet(seasonality_mode="multiplicative", holidays=HOLIDAYS)
    if sid.startswith("logistic"):
        return Prophet(growth="logistic")
    return Prophet(holidays=HOLIDAYS)


def _frames():
    frames = {f"add{i}": make_series(n_days=700, seed=i) for i in range(2)}
    frames.update({f"mult{i}": make_series(n_days=500 + 50 * i, seed=10 + i).assign(y=lambda d: d["y"] + 20)
                   for i in range(2)})
    lg = make_series(n_days=400, seed=20)
    frames["logistic0"] = lg.assign(cap=lg["y"].max() * 1.2)
    frames["const"] = make_series(n_days=300).assign(y=3.0)
    return frames


def _long(frames):
    return pd.concat([f.assign(unique_id=k) for k, f in frames.items()], ignore_index=True)


def _assert_identical(a, b):
    assert list(a) == list(b)
    for sid in a:
        assert a[sid].params.keys() == b[sid].params.keys(), sid
        for k in a[sid].params:
            np.testing.assert_array_equal(a[sid].params[k], b[sid].params[k], err_msg=f"{sid}.{k}")


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class TestSolverStan:
    def test_equals_prophet_fit(self):
        frames = _frames()
        fitted = fit_many(_long(frames), make_model=make_model, solver="stan", fit_kwargs=FIT_KW)
        stock = {sid: make_model(sid).fit(f, **FIT_KW) for sid, f in frames.items()}
        _assert_identical(fitted, stock)
        # Prophet.fit itself sets a constant series' parameters without Stan
        assert [sid for sid, m in fitted.items() if m.stan_fit is None] == ["const"]

    def test_independent_of_n_jobs_and_batch_size(self):
        df = _long(_frames())
        ref = fit_many(df, make_model=make_model, solver="stan")
        _assert_identical(fit_many(df, make_model=make_model, solver="stan", batch_size=1), ref)
        _assert_identical(fit_many(df, make_model=make_model, solver="stan", n_jobs=2), ref)

    def test_forecast_many_matches_fit_then_predict(self):
        df = _long({k: f for k, f in _frames().items() if not k.startswith("logistic")})
        fc = prophet_batch.forecast_many(df, periods=10, make_model=make_model, solver="stan", seed=3)
        models = fit_many(df, make_model=make_model, solver="stan")
        ref = prophet_batch.predict_many(models, periods=10, seed=3)
        assert list(fc["unique_id"].unique()) == list(ref)
        for sid, g in fc.groupby("unique_id", sort=False):
            got = g[list(ref[sid].columns)].reset_index(drop=True)
            pd.testing.assert_frame_equal(got, ref[sid].reset_index(drop=True), check_exact=True)

    def test_pool(self):
        df = _long({k: f for k, f in _frames().items() if k.startswith(("add", "mult"))})
        ref = prophet_batch.forecast_many(df, periods=5, make_model=make_model, solver="stan", seed=1)
        with prophet_batch.Pool(n_jobs=2) as pool:
            got = pool.forecast_many(df, periods=5, make_model=make_model, solver="stan", seed=1)
        pd.testing.assert_frame_equal(got, ref, check_exact=True)


class TestSolverArgument:
    def test_default_is_batch(self):
        df = _long({k: f for k, f in _frames().items() if k.startswith("add")})
        _assert_identical(fit_many(df, make_model=make_model), fit_many(df, make_model=make_model, solver="batch"))
        assert all(m.stan_fit is None for m in fit_many(df, make_model=make_model).values())

    @pytest.mark.parametrize("bad", ["auto", "Stan", None, 1])
    def test_invalid(self, bad):
        df = _long({"add0": make_series(n_days=100)})
        with pytest.raises(ValueError, match="solver"):
            fit_many(df, solver=bad)
        with pytest.raises(ValueError, match="solver"):
            prophet_batch.forecast_many(df, periods=3, solver=bad)
