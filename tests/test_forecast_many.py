"""0.1.2: forecast_many and Pool: stock-format forecasts in one step, equal to fit_many + predict_many."""

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from prophet_batch import Pool, fit_many, forecast_many, predict_many

from synthetic import make_series


def make_model(sid):
    """Module-level (picklable): the config depends on the series id."""
    if sid.startswith("logistic"):
        return Prophet(growth="logistic")
    if sid.startswith("mult"):
        return Prophet(seasonality_mode="multiplicative")
    if sid.startswith("reg"):
        m = Prophet()
        m.add_regressor("promo")
        return m
    return Prophet()


def _frames(with_futures=False):
    frames = {}
    for i in range(4):
        frames[f"add{i}"] = make_series(seed=i)                       # shared dates
    frames["mult0"] = make_series(n_days=600, seed=10).assign(y=lambda d: d["y"] + 20)
    frames["const"] = make_series(n_days=300).assign(y=3.0)
    frames["short"] = make_series(n_days=250, seed=21)
    if with_futures:
        lg = make_series(n_days=500, seed=20)
        frames["logistic0"] = lg.assign(cap=lg["y"].max() * 1.2)
        rg = make_series(n_days=400, seed=30)
        frames["reg0"] = rg.assign(promo=(np.arange(len(rg)) % 7 == 0).astype(float))
    return frames


def _long(frames):
    return pd.concat([f.assign(unique_id=k) for k, f in frames.items()], ignore_index=True)


def _futures(models, periods):
    out = {}
    for sid, m in models.items():
        fut = m.make_future_dataframe(periods)
        if sid.startswith("logistic"):
            fut["cap"] = m.history["cap"].iloc[0]
        if sid.startswith("reg"):
            fut["promo"] = (np.arange(len(fut)) % 7 == 0).astype(float)
        out[sid] = fut
    return out


def _by_series(long_fc, id_col="unique_id"):
    return {sid: g.drop(columns=id_col).reset_index(drop=True) for sid, g in long_fc.groupby(id_col, sort=False)}


def _assert_same(long_fc, per_series):
    got = _by_series(long_fc)
    assert list(got) == list(per_series)
    for sid, fc in per_series.items():
        # one long frame holds the union of every model's columns: the ones a model doesn't have are all NaN
        extra = [c for c in got[sid].columns if c not in fc.columns]
        assert got[sid][extra].isna().all().all(), (sid, extra)
        pd.testing.assert_frame_equal(got[sid][list(fc.columns)], fc.reset_index(drop=True), check_exact=True,
                                      obj=str(sid))


@pytest.fixture(scope="module")
def two_step():
    df = _long(_frames())
    return df, predict_many(fit_many(df, make_model=make_model), periods=10, seed=3)


@pytest.mark.parametrize("kwargs", [{}, {"n_jobs": 3}, {"batch_size": 2}, {"n_jobs": 2, "batch_size": 1}])
def test_equals_fit_many_then_predict_many(two_step, kwargs):
    df, ref = two_step
    _assert_same(forecast_many(df, periods=10, seed=3, make_model=make_model, **kwargs), ref)


def test_default_is_stock_format_and_values(two_step):
    df, _ = two_step
    fc = forecast_many(df, periods=10, seed=3, make_model=make_model)
    models = fit_many(df, make_model=make_model)
    m = models["add1"]
    stock = m.predict(m.make_future_dataframe(10))
    got = _by_series(fc)["add1"]
    assert list(fc.columns) == ["unique_id", *stock.columns]
    assert len(got) == len(stock)
    for col in ("ds", "yhat", "trend", "weekly", "yearly", "additive_terms"):
        np.testing.assert_array_equal(got[col].to_numpy(), stock[col].to_numpy(), err_msg=col)
    assert fc.attrs["seed"] == 3


def test_independent_of_input_split(two_step):
    df, ref = two_step
    ids = list(df["unique_id"].unique())
    halves = [forecast_many(df[df["unique_id"].isin(part)], periods=10, seed=3, make_model=make_model)
              for part in (ids[:3], ids[3:])]
    _assert_same(pd.concat(halves, ignore_index=True), ref)


def test_futures_long_frame_and_dict():
    df = _long(_frames(with_futures=True))
    models = fit_many(df, make_model=make_model)
    futures = _futures(models, 14)
    ref = predict_many(models, futures=futures, seed=5)
    long_futures = pd.concat([f.assign(unique_id=k) for k, f in futures.items()], ignore_index=True)
    _assert_same(forecast_many(df, futures=futures, seed=5, make_model=make_model), ref)
    _assert_same(forecast_many(df, futures=long_futures, seed=5, make_model=make_model, n_jobs=2), ref)


def test_opt_outs_change_no_values(two_step):
    df, ref = two_step
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    trimmed = forecast_many(df, periods=10, seed=3, make_model=make_model, columns=cols)
    assert list(trimmed.columns) == ["unique_id", *cols]
    _assert_same(trimmed, {sid: fc[cols] for sid, fc in ref.items()})
    future_only = forecast_many(df, periods=10, seed=3, make_model=make_model, include_history=False)
    models = fit_many(df, make_model=make_model)
    _assert_same(future_only, predict_many(models, periods=10, seed=3, include_history=False))
    analytic = forecast_many(df, periods=10, seed=3, make_model=make_model, history_uncertainty="analytic")
    _assert_same(analytic, predict_many(models, periods=10, seed=3, history_uncertainty="analytic"))


def test_return_params(two_step):
    df, _ = two_step
    fc, params = forecast_many(df, periods=5, seed=0, make_model=make_model, return_params=True, n_jobs=2)
    models = fit_many(df, make_model=make_model)
    assert list(params) == list(models)
    for sid, m in models.items():
        assert params[sid].keys() == m.params.keys()
        for k in m.params:
            np.testing.assert_array_equal(params[sid][k], m.params[k])
    assert isinstance(fc, pd.DataFrame)


def test_skip_and_order():
    frames = _frames()
    frames = {**dict(list(frames.items())[:2]), "bad": make_series(n_days=1), **dict(list(frames.items())[2:])}
    df = _long(frames)
    fc = forecast_many(df, periods=3, seed=1, make_model=make_model, on_error="skip", batch_size=2)
    assert list(fc["unique_id"].unique()) == [k for k in frames if k != "bad"]


def test_validation():
    df = _long({"a": make_series()})
    with pytest.raises(ValueError, match="exactly one of futures or periods"):
        forecast_many(df)
    with pytest.raises(ValueError, match="exactly one of futures or periods"):
        forecast_many(df, periods=3, futures={})
    with pytest.raises(ValueError, match="no future for series"):
        forecast_many(df, futures={"b": pd.DataFrame({"ds": []})})
    with pytest.raises(ValueError, match="columns"):
        forecast_many(df, periods=3, columns=[])
    with pytest.raises(ValueError, match="picklable"):
        forecast_many(df, periods=3, make_model=lambda sid: Prophet(), n_jobs=2)
    with pytest.raises(ValueError, match="batch_size"):
        forecast_many(df, periods=3, batch_size=0)


def test_series_ids_checked():
    """0.1.3 (M8d): rows with a missing id raise as in fit_many (they were dropped silently)."""
    df = _long({"a": make_series(), "b": make_series(seed=1)})
    holes = df.assign(unique_id=df["unique_id"].where(df.index % 50 != 0))
    for fn in (fit_many, lambda d: forecast_many(d, periods=3, seed=0)):
        with pytest.raises(ValueError, match=r"\d+ rows have a missing unique_id;"):
            fn(holes)
        with pytest.raises(ValueError, match="no id column 'unique_id'"):
            fn(df.drop(columns="unique_id"))
    future = df[["unique_id", "ds"]]
    with pytest.raises(ValueError, match="rows have a missing unique_id in futures"):
        forecast_many(df, futures=future.assign(unique_id=holes["unique_id"]), seed=0)
    with pytest.raises(ValueError, match="futures has no id column"):
        forecast_many(df, futures=future.drop(columns="unique_id"), seed=0)


def test_categorical_ids_with_unused_categories():
    """0.1.3 (M8d): unused categories of a categorical id are not series (pandas < 3 grouped them as empty)."""
    df = _long({"a": make_series(), "b": make_series(seed=1)})
    cat = df.assign(unique_id=pd.Categorical(df["unique_id"], categories=["a", "b", "unused"]))
    ref = forecast_many(df, periods=3, seed=0)
    for n_jobs in (1, 2):
        out = forecast_many(cat, periods=3, seed=0, n_jobs=n_jobs)
        pd.testing.assert_frame_equal(out.assign(unique_id=out["unique_id"].astype(str)), ref)
    assert list(fit_many(cat)) == ["a", "b"]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_predict_error_names_the_series(n_jobs):
    """0.1.3 (M8d): an error while predicting one series says which series it was."""
    df = _long(_frames(with_futures=True))
    models = fit_many(df, make_model=make_model)
    futures = _futures(models, 14)
    futures["reg0"].loc[len(futures["reg0"]) - 1, "promo"] = np.nan
    with pytest.raises(ValueError, match=r"series 'reg0': Found NaN in column 'promo'"):
        forecast_many(df, futures=futures, seed=0, make_model=make_model, n_jobs=n_jobs)
    with pytest.raises(ValueError, match=r"series 'reg0': Found NaN in column 'promo'"):
        predict_many(models, futures=futures, seed=0, n_jobs=n_jobs)


def test_assembler_unexpected_row_counts_and_missing_series():
    """The output stays right when a series' row count differs from the expected one, or a series is missing."""
    from prophet_batch._forecast import _Assembler, _Part

    ds = pd.date_range("2024-01-01", periods=5)
    part = _Part(["a", "b"], [3, 5], {"ds": np.concatenate([ds[:3], ds]), "yhat": np.arange(8.0)})
    asm = _Assembler(["a", "b", "c"], {"a": 3, "b": 4, "c": 2}, "unique_id")   # b: 5 rows, not 4; c: never comes
    asm.add(part)
    out = asm.finish()
    assert list(out["unique_id"]) == ["a"] * 3 + ["b"] * 5
    np.testing.assert_array_equal(out["yhat"].to_numpy(), np.arange(8.0))
    np.testing.assert_array_equal(out["ds"].to_numpy(), np.concatenate([ds[:3], ds]).astype(out["ds"].dtype))


def test_repeated_dates_in_history():
    """Duplicate ds rows: make_future_dataframe keeps the distinct dates; the output follows it."""
    frames = _frames()
    dup = frames["add1"]
    frames["add1"] = pd.concat([dup, dup.iloc[:5]], ignore_index=True).sort_values("ds", kind="stable")
    df = _long(frames)
    ref = predict_many(fit_many(df, make_model=make_model), periods=7, seed=2)
    _assert_same(forecast_many(df, periods=7, seed=2, make_model=make_model, n_jobs=2), ref)


class TestPool:
    def test_reuse_submit_and_concurrent_calls(self, two_step):
        df, ref = two_step
        ids = list(df["unique_id"].unique())
        parts = [df[df["unique_id"].isin(ids[:3])], df[df["unique_id"].isin(ids[3:])]]
        with Pool(n_jobs=2) as pool:
            first = pool.forecast_many(df, periods=10, seed=3, make_model=make_model)
            futures = [pool.submit(p, periods=10, seed=3, make_model=make_model) for p in parts]
            again = pool.forecast_many(df, periods=10, seed=3, make_model=make_model, batch_size=1)
            split = pd.concat([f.result() for f in futures], ignore_index=True)
        _assert_same(first, ref)
        _assert_same(again, ref)
        _assert_same(split, ref)

    def test_one_process_pool_and_closed(self, two_step):
        df, ref = two_step
        pool = Pool(n_jobs=1)
        _assert_same(pool.submit(df, periods=10, seed=3, make_model=make_model).result(), ref)
        pool.close()
        pool.close()                                      # idempotent
        with pytest.raises(RuntimeError, match="closed"):
            pool.forecast_many(df, periods=10, seed=3, make_model=make_model)
