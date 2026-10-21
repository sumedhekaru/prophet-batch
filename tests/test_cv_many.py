"""E4.1: cross_validation_many matches stock cross_validation's format and, for Stan-fit series, its forecasts."""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from prophet.diagnostics import cross_validation, prophet_copy
from prophet.metrics import performance_metrics
from api import cross_validation_many, fit_many

from synthetic import make_series
from api import BATCH_LOGGER  # noqa: E402

CV = dict(horizon="30 days", period="120 days", initial="600 days")


def make_model(sid):
    return Prophet(growth="logistic") if sid.startswith("lg") else Prophet()


def _frames():
    frames = {f"s{i}": make_series(seed=i) for i in range(3)}
    lg = make_series(seed=9)
    frames["lg"] = lg.assign(cap=lg["y"].max() * 1.2)
    return frames


@pytest.fixture(scope="module")
def models():
    long = pd.concat([f.assign(unique_id=k) for k, f in _frames().items()], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_many(long, make_model=make_model)


@pytest.fixture(scope="module")
def cvs(models):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cross_validation_many(models, seed=1, **CV)


def test_same_format_as_stock_and_stan_series_identical(models, cvs):
    assert list(cvs) == list(models)
    for sid, cv in cvs.items():
        stock = cross_validation(models[sid], disable_tqdm=True, **CV)
        assert list(cv.columns) == list(stock.columns) and (cv.dtypes == stock.dtypes).all()
        pd.testing.assert_frame_equal(cv[["ds", "y", "cutoff"]], stock[["ds", "y", "cutoff"]])
        if models[sid].stan_fit is not None:  # Stan series: the same refit, so the same yhat
            np.testing.assert_array_equal(cv["yhat"], stock["yhat"])


def test_batched_refits_are_at_least_as_good_as_stan(models, cvs):
    m = models["s0"]
    for cutoff in cvs["s0"]["cutoff"].unique():
        hist = m.history[m.history["ds"] <= cutoff]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ours = fit_many(hist.assign(unique_id=0), make_model=lambda _: prophet_copy(m, cutoff))[0]
            stan = prophet_copy(m, cutoff).fit(hist)
        assert ours.stan_fit is None
        assert ours.params["lp__"][0, 0] >= stan.params["lp__"][0, 0] - 1e-6


def test_metrics_and_plots_work(cvs):
    import matplotlib

    matplotlib.use("Agg")
    from prophet.plot import plot_cross_validation_metric

    for cv in cvs.values():
        pm = performance_metrics(cv)
        assert {"mse", "rmse", "mae", "coverage"} <= set(pm.columns) and len(pm)
        plot_cross_validation_metric(cv, "mape")


def test_seeded_and_independent_of_order_and_grouping(models, cvs, monkeypatch):
    from api import MODULE as batch

    reversed_models = dict(reversed(list(models.items())))
    monkeypatch.setattr(batch, "_CV_GROUP", 1)  # every series in its own fit_many call
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = cross_validation_many(reversed_models, seed=1, **CV)
    assert list(again) == list(reversed_models)
    for sid in models:
        pd.testing.assert_frame_equal(again[sid], cvs[sid])


def test_same_without_shared_preprocessing(models, cvs, monkeypatch):
    """F3.2: refits at a cutoff share preprocessing across series; the result is unchanged."""
    from api import MODULE as batch

    monkeypatch.setattr(batch, "_MAX_CONFIGS_PER_KEY", 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = cross_validation_many(models, seed=1, **CV)
    for sid in models:
        pd.testing.assert_frame_equal(again[sid], cvs[sid])


def test_as_frame(models, cvs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = cross_validation_many(models, seed=1, as_frame=True, id_col="series", **CV)
    assert list(frame.columns) == ["series"] + list(cvs["s0"].columns)
    for sid, cv in cvs.items():
        part = frame[frame["series"] == sid].drop(columns="series").reset_index(drop=True)
        pd.testing.assert_frame_equal(part, cv)


def test_explicit_cutoffs(models):
    cutoffs = [pd.Timestamp("2021-12-01"), pd.Timestamp("2022-03-01")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = cross_validation_many({"s1": models["s1"]}, horizon="30 days", cutoffs=cutoffs, seed=2)["s1"]
        stock = cross_validation(models["s1"], horizon="30 days", cutoffs=cutoffs, disable_tqdm=True)
    assert list(cv["cutoff"].unique()) == cutoffs
    pd.testing.assert_frame_equal(cv[["ds", "y", "cutoff"]], stock[["ds", "y", "cutoff"]])


def test_cutoffs_as_datetime_and_datetime64(models):
    """D4.4.1: stock cross_validation accepts datetime and np.datetime64 cutoffs; so does ours."""
    import datetime

    for cutoff in (datetime.datetime(2021, 12, 1), np.datetime64("2021-12-01")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = cross_validation_many({"s1": models["s1"]}, horizon="30 days", cutoffs=[cutoff], seed=2)["s1"]
            stock = cross_validation(models["s1"], horizon="30 days", cutoffs=[cutoff], disable_tqdm=True)
        pd.testing.assert_frame_equal(cv[["ds", "y", "cutoff"]], stock[["ds", "y", "cutoff"]])


def test_on_error(models, caplog):
    short = fit_many(make_series(n_days=300).assign(unique_id="short"))  # shorter than initial + horizon
    mixed = {"s0": models["s0"], "short": short["short"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"^series 'short': "):
            cross_validation_many(mixed, seed=1, **CV)
        with caplog.at_level(logging.WARNING, logger=BATCH_LOGGER):
            out = cross_validation_many(mixed, seed=1, on_error="skip", **CV)
    assert list(out) == ["s0"] and "skipping series 'short'" in caplog.text


def test_fit_kwargs_are_reused():
    lg = _frames()["lg"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = fit_many(lg.assign(unique_id="lg"), make_model=make_model, fit_kwargs={"algorithm": "Newton"})
        cv = cross_validation_many(ours, seed=3, **CV)["lg"]
        stock = cross_validation(Prophet(growth="logistic").fit(lg, algorithm="Newton"), disable_tqdm=True, **CV)
    np.testing.assert_array_equal(cv["yhat"], stock["yhat"])


def test_parallel_matches_serial(models, cvs):
    """E4.2: n_jobs > 1 gives the same results as serial, in the same order."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        par = cross_validation_many(models, seed=1, n_jobs=3, **CV)
    assert list(par) == list(cvs)
    for sid in cvs:
        pd.testing.assert_frame_equal(par[sid], cvs[sid])
    short = fit_many(make_series(n_days=300).assign(unique_id="short"))
    with pytest.raises(ValueError, match=r"^series 'short': "):
        cross_validation_many({"s0": models["s0"], "short": short["short"]}, seed=1, n_jobs=2, **CV)
