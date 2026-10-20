"""Scenarios for exact-parity tests. Each covers a different Prophet code path.

No pytest dependency, so make_baseline.py can import it.
"""

import numpy as np
import pandas as pd

from synthetic import make_series

PREDICT_SEED = 123


def _future(m, periods=30, include_history=True, df=None, future_x=True):
    fut = m.make_future_dataframe(periods=periods, include_history=include_history)
    if "cap" in m.history:
        fut["cap"] = m.history["cap"].iloc[0]
    if future_x and df is not None and "x" in df:
        rng = np.random.default_rng(5)
        fut["x"] = rng.normal(size=len(fut))
    return fut


def linear(Prophet):
    df = make_series()
    return Prophet(), df, {}


def linear_no_uncertainty(Prophet):
    df = make_series()
    return Prophet(uncertainty_samples=0), df, {}


def multiplicative(Prophet):
    df = make_series()
    df["y"] = df["y"] + 20
    return Prophet(seasonality_mode="multiplicative"), df, {}


def logistic(Prophet):
    df = make_series()
    df["cap"] = df["y"].max() * 1.2
    return Prophet(growth="logistic"), df, {}


def flat(Prophet):
    df = make_series()
    return Prophet(growth="flat"), df, {}


def holidays_regressor(Prophet):
    df = make_series()
    rng = np.random.default_rng(4)
    df["x"] = rng.normal(size=len(df))
    df["y"] = df["y"] + 0.8 * df["x"]
    holidays = pd.DataFrame({
        "holiday": "promo",
        "ds": pd.date_range("2020-02-01", periods=12, freq="90D"),
        "lower_window": 0,
        "upper_window": 1,
    })
    m = Prophet(holidays=holidays)
    m.add_regressor("x")
    m.add_country_holidays("US")
    return m, df, {}


def short_newton(Prophet):
    df = make_series(n_days=80)
    return Prophet(), df, {}


def not_vectorized(Prophet):
    df = make_series(n_days=400)
    return Prophet(uncertainty_samples=200), df, {}, {"vectorized": False}


def regressor_predictor(Prophet):
    # The regressor gets its own Prophet model; its samples go through the non-vectorized path.
    df = make_series(n_days=400)
    t = np.arange(len(df))
    df["x"] = 5 + 0.01 * t + np.sin(2 * np.pi * t / 7)
    df["y"] = df["y"] + 0.8 * df["x"]
    m = Prophet(uncertainty_samples=200)
    m.add_regressor("x", regressor_predictor=True)
    return m, df, {}, {"future_x": False}


def mcmc(Prophet):
    df = make_series(n_days=400)
    return Prophet(mcmc_samples=40), df, {"seed": 1, "chains": 2, "show_progress": False}


SCENARIOS = {
    f.__name__: f
    for f in [linear, linear_no_uncertainty, multiplicative, logistic, flat,
              holidays_regressor, short_newton, not_vectorized, regressor_predictor, mcmc]
}


def run(name, Prophet):
    """Fit the scenario and return a flat dict of float64 arrays."""
    m, df, fit_kwargs, *opts = SCENARIOS[name](Prophet)
    opts = opts[0] if opts else {}
    predict_kwargs = {k: v for k, v in opts.items() if k == "vectorized"}
    m.fit(df, **fit_kwargs)
    out = {f"params.{k}": np.asarray(v, dtype=np.float64) for k, v in m.params.items()}
    for include_history in (True, False):
        np.random.seed(PREDICT_SEED)
        fut = _future(m, include_history=include_history, df=df, future_x=opts.get("future_x", True))
        fc = m.predict(fut, **predict_kwargs)
        tag = "hist" if include_history else "future"
        for col in fc.columns:
            if col == "ds":
                out[f"{tag}.ds"] = fc["ds"].to_numpy().astype("datetime64[ns]").astype(np.int64)
            else:
                out[f"{tag}.{col}"] = fc[col].to_numpy(dtype=np.float64)
    return out
