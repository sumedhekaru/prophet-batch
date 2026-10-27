"""0.1.1: fit_many and cross_validation_many reuse one loaded Stan model per process, with unchanged results."""

import warnings

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from prophet.models import CmdStanPyBackend
from api import MODULE, cross_validation_many, fit_many

from synthetic import make_series

CV = dict(horizon="30 days", period="120 days", initial="600 days")


def make_model(sid):
    if sid.startswith("logistic"):
        return Prophet(growth="logistic")
    if sid.startswith("mult"):
        return Prophet(seasonality_mode="multiplicative")
    return Prophet()


def _frames():
    frames = {f"add{i}": make_series(seed=i) for i in range(3)}
    for i in range(2):
        frames[f"mult{i}"] = make_series(n_days=600, seed=10 + i).assign(y=lambda d: d["y"] + 20)
    frames["const"] = make_series(n_days=300).assign(y=3.0)
    lg = make_series(n_days=500, seed=20)
    frames["logistic0"] = lg.assign(cap=lg["y"].max() * 1.2)
    return frames


def _long(frames):
    return pd.concat([f.assign(unique_id=k) for k, f in frames.items()], ignore_index=True)


def _fit(df, reuse, monkeypatch, **kwargs):
    monkeypatch.setattr(MODULE, "_REUSE_STAN_MODEL", reuse)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_many(df, **kwargs)


def _assert_identical(a, b):
    assert list(a) == list(b)
    for sid in a:
        assert a[sid].params.keys() == b[sid].params.keys(), sid
        for k in a[sid].params:
            np.testing.assert_array_equal(a[sid].params[k], b[sid].params[k], err_msg=f"{sid}.{k}")


def test_mixed_batch_identical_with_and_without_reuse(monkeypatch):
    df = _long(_frames())
    _assert_identical(_fit(df, False, monkeypatch, make_model=make_model),
                      _fit(df, True, monkeypatch, make_model=make_model))


def test_stan_fallback_matches_prophet_fit(monkeypatch):
    frames = _frames()
    fitted = _fit(_long(frames), True, monkeypatch, make_model=make_model)
    assert fitted["logistic0"].stan_fit is not None
    stock = Prophet(growth="logistic").fit(frames["logistic0"])
    _assert_identical({"logistic0": fitted["logistic0"]}, {"logistic0": stock})


def test_model_loaded_once_per_fit_many(monkeypatch):
    original = CmdStanPyBackend.load_model
    calls = []

    def counting(self):
        calls.append(1)
        return original(self)

    # A new function: _reuse_stan_model caches by the original load_model, so this one loads afresh.
    monkeypatch.setattr(CmdStanPyBackend, "load_model", counting)
    frames = {f"s{i}": make_series(n_days=200, seed=i) for i in range(20)}
    _fit(_long(frames), True, monkeypatch)
    assert len(calls) == 1


def test_every_model_has_its_own_backend(monkeypatch):
    fitted = _fit(_long(_frames()), True, monkeypatch, make_model=make_model)
    backends = [m.stan_backend for m in fitted.values()]
    assert len({id(b) for b in backends}) == len(backends)
    assert len({id(b.model) for b in backends}) == 1  # only the compiled model is shared
    assert [sid for sid, m in fitted.items() if m.stan_backend.stan_fit is not None] == ["logistic0"]


def test_load_model_restored_after_return_and_after_raise(monkeypatch):
    original = CmdStanPyBackend.__dict__["load_model"]
    monkeypatch.setattr(MODULE, "_REUSE_STAN_MODEL", True)
    models = fit_many(_long({"a": make_series(n_days=200)}))
    assert CmdStanPyBackend.__dict__["load_model"] is original
    cross_validation_many(models, seed=0, horizon="20 days", initial="120 days", period="40 days")
    assert CmdStanPyBackend.__dict__["load_model"] is original
    with pytest.raises(ValueError):
        fit_many(_long({"short": make_series(n_days=1)}))
    assert CmdStanPyBackend.__dict__["load_model"] is original
    with pytest.raises(Exception, match="has not been fit"):
        cross_validation_many({"unfit": Prophet()}, seed=0, horizon="20 days")
    assert CmdStanPyBackend.__dict__["load_model"] is original


def test_cross_validation_many_identical_with_and_without_reuse(monkeypatch):
    # Additive, constant and logistic (Stan) series, long enough for the CV windows.
    frames = {f"add{i}": make_series(seed=i) for i in range(2)}
    frames["const"] = make_series().assign(y=3.0)
    lg = make_series(seed=20)
    frames["logistic0"] = lg.assign(cap=lg["y"].max() * 1.2)
    models = _fit(_long(frames), True, monkeypatch, make_model=make_model)
    out = {}
    for reuse in (False, True):
        monkeypatch.setattr(MODULE, "_REUSE_STAN_MODEL", reuse)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[reuse] = cross_validation_many(models, seed=1, **CV)
    assert list(out[False]) == list(out[True])
    for sid in out[False]:
        pd.testing.assert_frame_equal(out[False][sid], out[True][sid])


def test_public_reuse_stan_model_around_own_fit_loop(monkeypatch):
    # 0.1.3: a user's own Prophet.fit loop inside prophet_batch.reuse_stan_model() loads the model once
    import prophet_batch

    original = CmdStanPyBackend.load_model
    calls = []

    def counting(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(CmdStanPyBackend, "load_model", counting)
    frames = {f"s{i}": make_series(n_days=200, seed=i) for i in range(4)}
    with prophet_batch.reuse_stan_model():
        fitted = {sid: Prophet().fit(f) for sid, f in frames.items()}
    assert len(calls) == 1
    assert CmdStanPyBackend.load_model is counting  # restored on exit
    stock = {sid: Prophet().fit(f) for sid, f in frames.items()}
    _assert_identical(fitted, stock)
    assert MODULE._reuse_stan_model is prophet_batch.reuse_stan_model  # the old name, for one release
