"""C4: predict(history_uncertainty=False)."""

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from scenarios import SCENARIOS, _future, PREDICT_SEED
from api import predict  # noqa: E402

INTERVALS = ["yhat_lower", "yhat_upper", "trend_lower", "trend_upper"]


@pytest.mark.parametrize("name", [n for n in SCENARIOS if n != "linear_no_uncertainty"])
def test_future_intervals_equal_future_only_predict(name):
    m, df, fit_kwargs, *opts = SCENARIOS[name](Prophet)
    opts = opts[0] if opts else {}
    predict_kwargs = {k: v for k, v in opts.items() if k == "vectorized"}
    m.fit(df, **fit_kwargs)
    full = _future(m, include_history=True, df=df, future_x=opts.get("future_x", True))
    np.random.seed(PREDICT_SEED)
    fc = predict(m, full, history_uncertainty=False, **predict_kwargs)

    hist = fc["ds"] <= m.history["ds"].max()
    assert fc.loc[hist, INTERVALS].isna().all().all()
    # Point forecasts and component columns are unaffected.
    np.random.seed(PREDICT_SEED)
    ref_all = m.predict(full, **predict_kwargs)
    other = [c for c in fc.columns if c not in INTERVALS]
    pd.testing.assert_frame_equal(fc[other], ref_all[other])
    # Future intervals: exactly those of predicting the future rows alone, same seed.
    future = full[full["ds"] > m.history["ds"].max()].reset_index(drop=True)
    np.random.seed(PREDICT_SEED)
    ref_future = m.predict(future, **predict_kwargs)
    np.testing.assert_array_equal(fc.loc[~hist, INTERVALS].to_numpy(), ref_future[INTERVALS].to_numpy())


def test_history_only_frame_has_no_intervals():
    from synthetic import make_series

    df = make_series(n_days=300)
    m = Prophet().fit(df)
    fc = predict(m, df[["ds"]], history_uncertainty=False)
    assert fc[INTERVALS].isna().all().all() and fc["yhat"].notna().all()


# D2: history_uncertainty='analytic'.

def _half_width(m):
    from statistics import NormalDist

    z = NormalDist().inv_cdf((1 + m.interval_width) / 2)
    return z * float(np.ravel(m.params["sigma_obs"])[0]) * m.y_scale


def _check_analytic_history(m, fc, hist):
    """History bounds are yhat ± z·sigma·y_scale, trend bounds are the trend."""
    hw = _half_width(m)
    yhat = fc.loc[hist, "yhat"].to_numpy()
    np.testing.assert_allclose(fc.loc[hist, "yhat_lower"].to_numpy(), yhat - hw, rtol=1e-14, atol=0)
    np.testing.assert_allclose(fc.loc[hist, "yhat_upper"].to_numpy(), yhat + hw, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(fc.loc[hist, "trend_lower"].to_numpy(), fc.loc[hist, "trend"].to_numpy())
    np.testing.assert_array_equal(fc.loc[hist, "trend_upper"].to_numpy(), fc.loc[hist, "trend"].to_numpy())


@pytest.mark.parametrize("name", [n for n in SCENARIOS if n not in ("linear_no_uncertainty", "mcmc")])
def test_analytic_history_and_future_equal_false(name):
    m, df, fit_kwargs, *opts = SCENARIOS[name](Prophet)
    opts = opts[0] if opts else {}
    predict_kwargs = {k: v for k, v in opts.items() if k == "vectorized"}
    m.fit(df, **fit_kwargs)
    full = _future(m, include_history=True, df=df, future_x=opts.get("future_x", True))
    fc = predict(m, full, history_uncertainty="analytic", rng=np.random.default_rng(3), **predict_kwargs)
    ref = predict(m, full, history_uncertainty=False, rng=np.random.default_rng(3), **predict_kwargs)

    hist = (fc["ds"] <= m.history["ds"].max()).to_numpy()
    _check_analytic_history(m, fc, hist)
    # Future rows and every other column: exactly those of history_uncertainty=False.
    np.testing.assert_array_equal(fc.loc[~hist, INTERVALS].to_numpy(), ref.loc[~hist, INTERVALS].to_numpy())
    other = [c for c in fc.columns if c not in INTERVALS]
    pd.testing.assert_frame_equal(fc[other], ref[other])


def test_analytic_is_the_limit_of_sampling():
    from statistics import NormalDist
    from synthetic import make_series

    df = make_series(n_days=120)
    m = Prophet(uncertainty_samples=50_000).fit(df)
    fut = m.make_future_dataframe(5)
    analytic = predict(m, fut, history_uncertainty="analytic", rng=np.random.default_rng(0))
    sampled = predict(m, fut, history_uncertainty="sampled", rng=np.random.default_rng(0))
    hist = (fut["ds"] <= df["ds"].max()).to_numpy()
    # Monte Carlo standard error of the 10% / 90% sample quantiles, in units of sigma·y_scale.
    p, z = 0.1, NormalDist().inv_cdf(0.9)
    se = np.sqrt(p * (1 - p) / m.uncertainty_samples) / NormalDist().pdf(z) * _half_width(m) / z
    for col in ("yhat_lower", "yhat_upper"):
        err = sampled.loc[hist, col].to_numpy() - analytic.loc[hist, col].to_numpy()
        assert np.max(np.abs(err)) < 5 * se
    np.testing.assert_array_equal(sampled.loc[hist, "trend_lower"], analytic.loc[hist, "trend_lower"])


def test_analytic_history_only_frame():
    from synthetic import make_series

    df = make_series(n_days=300)
    m = Prophet().fit(df)
    fc = predict(m, history_uncertainty="analytic")
    assert fc[INTERVALS].notna().all().all()
    _check_analytic_history(m, fc, np.ones(len(fc), dtype=bool))


def test_analytic_without_uncertainty_samples():
    from synthetic import make_series

    df = make_series(n_days=300)
    m = Prophet(uncertainty_samples=0).fit(df)
    fut = m.make_future_dataframe(10)
    pd.testing.assert_frame_equal(predict(m, fut, history_uncertainty="analytic"), predict(m, fut))


@pytest.mark.parametrize("value", ["sample", 1, None])
def test_invalid_history_uncertainty(value):
    from synthetic import make_series

    m = Prophet().fit(make_series(n_days=200))
    with pytest.raises(ValueError, match="history_uncertainty must be"):
        predict(m, history_uncertainty=value)


def test_analytic_rejects_mcmc():
    from synthetic import make_series

    m = Prophet(mcmc_samples=20).fit(make_series(n_days=100), chains=1, seed=1, show_progress=False)
    with pytest.raises(ValueError, match="requires a MAP fit"):
        predict(m, history_uncertainty="analytic")


# N3.2 (0.1.3): history_uncertainty=True with an rng draws history bounds from order statistics.

OLD_PATH = ("not_vectorized", "regressor_predictor", "mcmc")  # sampled as before (not eligible)


def _fitted(name):
    m, df, fit_kwargs, *opts = SCENARIOS[name](Prophet)
    opts = opts[0] if opts else {}
    m.fit(df, **fit_kwargs)
    full = _future(m, include_history=True, df=df, future_x=opts.get("future_x", True))
    return m, full, {k: v for k, v in opts.items() if k == "vectorized"}


@pytest.mark.parametrize("name", [n for n in SCENARIOS if n not in ("linear_no_uncertainty",) + OLD_PATH])
def test_order_statistics_history_and_future_equal_false(name):
    m, full, predict_kwargs = _fitted(name)
    fc = predict(m, full, rng=np.random.default_rng(3), **predict_kwargs)
    ref = predict(m, full, history_uncertainty=False, rng=np.random.default_rng(3), **predict_kwargs)
    hist = (fc["ds"] <= m.history["ds"].max()).to_numpy()
    # Future rows and every other column: exactly those of history_uncertainty=False.
    np.testing.assert_array_equal(fc.loc[~hist, INTERVALS].to_numpy(), ref.loc[~hist, INTERVALS].to_numpy())
    other = [c for c in fc.columns if c not in INTERVALS]
    pd.testing.assert_frame_equal(fc[other], ref[other])
    # History: trend bounds are the trend (no trend uncertainty in the history), yhat bounds around yhat.
    np.testing.assert_array_equal(fc.loc[hist, "trend_lower"].to_numpy(), fc.loc[hist, "trend"].to_numpy())
    np.testing.assert_array_equal(fc.loc[hist, "trend_upper"].to_numpy(), fc.loc[hist, "trend"].to_numpy())
    yhat = fc.loc[hist, "yhat"].to_numpy()
    assert (fc.loc[hist, "yhat_lower"].to_numpy() < yhat).all() and (fc.loc[hist, "yhat_upper"].to_numpy() > yhat).all()


@pytest.mark.parametrize("name", OLD_PATH)
def test_not_eligible_models_sample_every_path(name):
    m, full, predict_kwargs = _fitted(name)
    fc = predict(m, full, rng=np.random.default_rng(3), **predict_kwargs)
    ref = predict(m, full, history_uncertainty="sampled", rng=np.random.default_rng(3), **predict_kwargs)
    pd.testing.assert_frame_equal(fc, ref)


def _ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                         - np.searchsorted(b, grid, side="right") / len(b)))


@pytest.mark.parametrize("width", [0.8, 0.95])
def test_order_statistics_have_the_sampled_distribution(width):
    """History bounds, standardized by sigma·y_scale, have the distribution of sampling every path."""
    from synthetic import make_series

    df = make_series(n_days=3000)
    m = Prophet(interval_width=width, uncertainty_samples=1000).fit(df)
    hist = df[["ds"]]
    scale = float(np.ravel(m.params["sigma_obs"])[0]) * m.y_scale
    new = predict(m, hist, rng=np.random.default_rng(1))
    old = predict(m, hist, history_uncertainty="sampled", rng=np.random.default_rng(2))
    n = len(hist)
    for col in ("yhat_lower", "yhat_upper"):
        a = (new[col] - new["yhat"]).to_numpy() / scale
        b = (old[col] - old["yhat"]).to_numpy() / scale
        # KS at significance 0.001; means within 4 standard errors (seeded: deterministic).
        assert _ks_statistic(a, b) < 1.95 * np.sqrt(2 / n), col
        assert abs(a.mean() - b.mean()) < 4 * np.sqrt((a.var() + b.var()) / n), col


def test_order_statistics_rng_none_is_stock():
    from synthetic import make_series

    m = Prophet().fit(make_series(n_days=300))
    fut = m.make_future_dataframe(10)
    np.random.seed(5)
    ref = m.predict(fut)
    np.random.seed(5)
    pd.testing.assert_frame_equal(predict(m, fut), ref)
