"""E1.2.1: the stock Prophet API (group A of docs/parity-matrix.md) on fit_many models.

For each model kind we build three fitted models from the same data and configuration:
  stock: Prophet.fit;  ours: fit_many;  twin: ours with stock's params copied in.
Each check runs an API call on all three. It must run on `ours`; on `twin` its output must be
bit-identical to `stock`'s ('equal' checks), so any difference comes from model state that the
batched path sets differently, not from the different (better) MAP. 'runs' checks (plots) only
need to run. No pytest dependency: tests/test_api_parity.py (and the workspace's experiments/e1_parity.py) use it.
"""

from __future__ import annotations

import copy
import importlib.util
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from synthetic import make_series  # noqa: E402

HAS_PLOTLY = importlib.util.find_spec("plotly") is not None
NA = object()  # a check that doesn't apply to this model kind (e.g. no regressor)
HOLIDAYS = pd.DataFrame({"holiday": "promo", "ds": pd.date_range("2020-02-01", periods=12, freq="90D"),
                         "lower_window": 0, "upper_window": 1})
CV = dict(initial="600 days", period="120 days", horizon="30 days")


def _with_x(df, seed):
    df = df.copy()
    df["x"] = np.random.default_rng(seed).normal(size=len(df))
    df["y"] = df["y"] + 0.8 * df["x"]
    return df


def _additive(Prophet):
    m = Prophet(holidays=HOLIDAYS)
    m.add_regressor("x")
    return m


def _multiplicative(Prophet):
    m = Prophet(seasonality_mode="multiplicative", holidays=HOLIDAYS)
    m.add_regressor("x", mode="additive")
    return m


def _predictor(Prophet):
    m = Prophet()
    m.add_regressor("x", regressor_predictor=True)
    return m


# name -> (make model, history frame, has regressor x, x needed in the future frame)
KINDS = {
    "additive": (_additive, lambda: _with_x(make_series(), 1), True, True),
    "multiplicative": (_multiplicative, lambda: _with_x(make_series().assign(y=lambda d: d["y"] + 20), 2),
                       True, True),
    "flat": (lambda P: P(growth="flat"), lambda: make_series(seed=3), False, False),
    "constant": (lambda P: P(), lambda: make_series(seed=4).assign(y=5.0), False, False),
    "fallback: logistic": (lambda P: P(growth="logistic"),
                           lambda: make_series(seed=5).assign(cap=lambda d: d["y"].max() * 1.2), False, False),
    "fallback: regressor predictor": (_predictor, lambda: _with_x(make_series(seed=6), 6), True, False),
}


def models(kind):
    """(stock, ours, twin, df, future) for a model kind."""
    from prophet import Prophet
    from api import fit_many

    make, frame, _, future_x = KINDS[kind]
    df = frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stock = make(Prophet).fit(df)
        ours = fit_many(df.assign(unique_id="s"), make_model=lambda sid: make(Prophet))["s"]
    twin = copy.deepcopy(ours)
    twin.params = copy.deepcopy(stock.params)
    future = stock.make_future_dataframe(30)
    if "cap" in df:
        future["cap"] = df["cap"].iloc[0]
    if future_x:
        future["x"] = np.random.default_rng(99).normal(size=len(future))
    return stock, ours, twin, df, future


def _seeded(fn):
    def run(m, ctx):
        np.random.seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(m, ctx)
    return run


def _fig(fn):
    def run(m, ctx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fn(m, ctx)
        plt.close("all")
        return out
    return run


def _cv(m, ctx):
    from prophet.diagnostics import cross_validation

    return cross_validation(m, **CV, disable_tqdm=True)


def _setup(m, ctx):
    return m.setup_dataframe(ctx["future"].copy())


def _fcst(m, ctx):
    np.random.seed(0)
    return m.predict(ctx["future"])


def _vars(obj):
    """Model state, without handles to the compiled Stan model and its output files."""
    return {k: v for k, v in vars(obj).items() if k not in ("stan_backend", "stan_fit")}


def _plotly(fn):
    return None if not HAS_PLOTLY else _fig(fn)


def _checks():
    from prophet import diagnostics, metrics, plot, serialize, utilities

    c = [
        ("Prophet", "predict", "equal", _seeded(lambda m, x: m.predict(x["future"]))),
        ("Prophet", "predict (vectorized=False: sample_model, sample_predictive_trend)", "equal",
         _seeded(lambda m, x: m.predict(x["future"], vectorized=False))),
        ("Prophet", "predictive_samples", "equal", _seeded(lambda m, x: m.predictive_samples(x["future"]))),
        ("Prophet", "make_future_dataframe", "equal", lambda m, x: m.make_future_dataframe(30)),
        ("Prophet", "predict_trend", "equal", lambda m, x: m.predict_trend(_setup(m, x))),
        ("Prophet", "predict_seasonal_components", "equal",
         lambda m, x: m.predict_seasonal_components(_setup(m, x))),
        ("Prophet", "predict_uncertainty (sample_posterior_predictive, *_vectorized)", "equal",
         _seeded(lambda m, x: m.predict_uncertainty(_setup(m, x), vectorized=True))),
        ("Prophet", "fit (again)", "equal", lambda m, x: m.fit(x["df"])),
        ("Prophet", "plot", "runs", _fig(lambda m, x: m.plot(_fcst(m, x)))),
        ("Prophet", "plot_components", "runs", _fig(lambda m, x: m.plot_components(_fcst(m, x)))),
        ("prophet.diagnostics", "cross_validation", "equal", _seeded(_cv)),
        ("prophet.diagnostics", "generate_cutoffs", "equal",
         lambda m, x: diagnostics.generate_cutoffs(x["df"], pd.Timedelta("30 days"), pd.Timedelta("600 days"),
                                                   pd.Timedelta("120 days"))),
        ("prophet.diagnostics", "single_cutoff_forecast", "equal",
         _seeded(lambda m, x: diagnostics.single_cutoff_forecast(
             x["df"], m, pd.Timestamp(x["df"]["ds"].iloc[700]), pd.Timedelta("30 days"),
             ["ds", "yhat", "yhat_lower", "yhat_upper"]))),
        ("prophet.diagnostics", "prophet_copy", "equal",
         lambda m, x: _vars(diagnostics.prophet_copy(m, pd.Timestamp(x["df"]["ds"].iloc[700])))),
        ("prophet.metrics", "performance_metrics", "equal",
         _seeded(lambda m, x: metrics.performance_metrics(_cv(m, x)))),
        ("prophet.plot", "plot", "runs", _fig(lambda m, x: plot.plot(m, _fcst(m, x)))),
        ("prophet.plot", "plot_components", "runs", _fig(lambda m, x: plot.plot_components(m, _fcst(m, x)))),
        ("prophet.plot", "plot_forecast_component", "runs",
         _fig(lambda m, x: plot.plot_forecast_component(m, _fcst(m, x), "trend"))),
        ("prophet.plot", "seasonality_plot_df", "equal",
         lambda m, x: plot.seasonality_plot_df(m, pd.date_range("2021-01-01", periods=7))),
        ("prophet.plot", "plot_weekly", "runs", _fig(lambda m, x: plot.plot_weekly(m))),
        ("prophet.plot", "plot_yearly", "runs", _fig(lambda m, x: plot.plot_yearly(m))),
        ("prophet.plot", "plot_seasonality", "runs", _fig(lambda m, x: plot.plot_seasonality(m, "weekly"))),
        ("prophet.plot", "add_changepoints_to_plot", "runs",
         _fig(lambda m, x: plot.add_changepoints_to_plot(m.plot(_fcst(m, x)).gca(), m, _fcst(m, x)))),
        ("prophet.plot", "plot_cross_validation_metric", "runs",
         _fig(lambda m, x: plot.plot_cross_validation_metric(_seeded(_cv)(m, x), "mape"))),
        ("prophet.plot", "plot_plotly", "runs", _plotly(lambda m, x: plot.plot_plotly(m, _fcst(m, x)))),
        ("prophet.plot", "plot_components_plotly", "runs",
         _plotly(lambda m, x: plot.plot_components_plotly(m, _fcst(m, x)))),
        ("prophet.plot", "plot_forecast_component_plotly", "runs",
         _plotly(lambda m, x: plot.plot_forecast_component_plotly(m, _fcst(m, x), "trend"))),
        ("prophet.plot", "plot_seasonality_plotly", "runs",
         _plotly(lambda m, x: plot.plot_seasonality_plotly(m, "weekly"))),
        ("prophet.plot", "get_forecast_component_plotly_props", "runs",
         _plotly(lambda m, x: plot.get_forecast_component_plotly_props(m, _fcst(m, x), "trend"))),
        ("prophet.plot", "get_seasonality_plotly_props", "runs",
         _plotly(lambda m, x: plot.get_seasonality_plotly_props(m, "weekly"))),
        ("prophet.serialize", "model_to_dict", "equal", lambda m, x: serialize.model_to_dict(m)),
        ("prophet.serialize", "model_to_json", "equal", lambda m, x: serialize.model_to_json(m)),
        ("prophet.serialize", "model_from_dict (round trip, then predict)", "equal",
         _seeded(lambda m, x: serialize.model_from_dict(serialize.model_to_dict(m)).predict(x["future"]))),
        ("prophet.serialize", "model_from_json (round trip, then predict)", "equal",
         _seeded(lambda m, x: serialize.model_from_json(serialize.model_to_json(m)).predict(x["future"]))),
        ("prophet.utilities", "regressor_index", "equal",
         lambda m, x: utilities.regressor_index(m, "x") if x["has_x"] else NA),
        ("prophet.utilities", "regressor_coefficients", "equal",
         lambda m, x: utilities.regressor_coefficients(m) if x["has_x"] else NA),
        ("prophet.utilities", "warm_start_params", "equal", lambda m, x: utilities.warm_start_params(m)),
    ]
    return c


def _same(a, b):
    """Bit-identical comparison for the output types the API returns."""
    if isinstance(a, pd.DataFrame):
        return isinstance(b, pd.DataFrame) and a.equals(b) and list(a.columns) == list(b.columns)
    if isinstance(a, pd.Series):
        return isinstance(b, pd.Series) and a.equals(b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if type(a).__name__ == "Prophet":  # e.g. a regressor's fitted predictor model
        return type(b) is type(a) and _same(_vars(a), _vars(b))
    try:
        return bool(a == b)
    except Exception:
        return False


def _call(fn, m, ctx):
    try:
        return fn(m, ctx), None
    except Exception as e:
        return None, (type(e).__name__, str(e).split("\n")[0][:120])


def detects_difference(kind="additive"):
    """Self-test: comparing ours (a different MAP) with stock must report a difference."""
    stock, ours, _, df, future = models(kind)
    ctx = {"df": df, "future": future, "has_x": KINDS[kind][2]}
    fn = dict(((w, n), f) for w, n, _, f in _checks())[("Prophet", "predict")]
    return not _same(fn(ours, ctx), fn(stock, ctx))


def run(kind):
    """{(where, name): (status, detail)} for one model kind; status '=', 'ok', '✗' or 'skip'."""
    stock, ours, twin, df, future = models(kind)
    ctx = {"df": df, "future": future, "has_x": KINDS[kind][2]}
    out = {}
    for where, name, how, fn in _checks():
        if fn is None:
            out[(where, name)] = ("skip", "plotly not installed")
            continue
        o_ours, e_ours = _call(fn, copy.deepcopy(ours), ctx)
        o_stock, e_stock = _call(fn, copy.deepcopy(stock), ctx)
        o_twin, e_twin = _call(fn, copy.deepcopy(twin), ctx)
        if o_stock is NA:
            out[(where, name)] = ("–", "not applicable to this model kind")
        elif e_stock and e_ours == e_stock and e_twin == e_stock:
            out[(where, name)] = ("=", f"same error as stock: {e_stock[0]}")
        elif e_ours or e_twin or e_stock:
            out[(where, name)] = ("✗", f"stock {e_stock}, ours {e_ours}, twin {e_twin}")
        elif how == "runs":
            out[(where, name)] = ("ok", "")
        elif _same(o_twin, o_stock):
            out[(where, name)] = ("=", "")
        else:
            out[(where, name)] = ("✗", "twin output differs from stock")
    return out
