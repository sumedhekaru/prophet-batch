"""C7.3: predict_many gives the same forecasts regardless of n_jobs and order."""

import random

import numpy as np
import pandas as pd
import pytest
from api import fit_many, predict_many, series_rng

from synthetic import make_series
from api import predict  # noqa: E402


@pytest.fixture(scope="module")
def models():
    frames = {f"s{i}": make_series(n_days=400 + 50 * i, seed=i) for i in range(12)}
    long = pd.concat([f.assign(unique_id=k) for k, f in frames.items()], ignore_index=True)
    return fit_many(long)


def _equal(a, b):
    assert list(a) == list(b)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_same_results_for_any_n_jobs(models):
    serial = predict_many(models, periods=14, seed=42)
    parallel = predict_many(models, periods=14, seed=42, n_jobs=4)
    _equal(serial, parallel)


def test_independent_of_order_and_subset(models):
    full = predict_many(models, periods=14, seed=42)
    ids = list(models)
    random.Random(0).shuffle(ids)
    shuffled = predict_many({k: models[k] for k in ids}, periods=14, seed=42)
    for k in ids:
        pd.testing.assert_frame_equal(full[k], shuffled[k])
    one = predict_many({"s5": models["s5"]}, periods=14, seed=42)
    pd.testing.assert_frame_equal(full["s5"], one["s5"])


def test_matches_single_model_predict(models):
    fc = predict_many(models, periods=14, seed=7, history_uncertainty=False)
    m = models["s3"]
    ref = predict(m, m.make_future_dataframe(14), history_uncertainty=False, rng=series_rng(7, "s3"))
    pd.testing.assert_frame_equal(fc["s3"], ref)


def test_seed_none_is_recorded_and_reproducible(models):
    fc = predict_many(models, periods=7)
    seed = fc["s0"].attrs["seed"]
    again = predict_many(models, periods=7, seed=seed)
    _equal(fc, again)


def test_explicit_futures(models):
    futures = {k: m.make_future_dataframe(10, include_history=False) for k, m in models.items()}
    a = predict_many(models, futures=futures, seed=1)
    b = predict_many(models, periods=10, include_history=False, seed=1)
    _equal(a, b)
    with pytest.raises(ValueError):
        predict_many(models, futures=futures, periods=10)


def test_n_jobs_1_runs_in_process(models, monkeypatch):
    from api import MODULE as pb

    calls = []
    original = pb._predict_chunk

    def spy(items, seed, hu):
        calls.append(len(items))
        return original(items, seed, hu)

    monkeypatch.setattr(pb, "_predict_chunk", spy)
    pb.predict_many(models, periods=7, seed=1, n_jobs=1)
    assert calls == [len(models)]  # one in-process call: no worker processes


def test_analytic_matches_single_model_predict_for_any_n_jobs(models):
    serial = predict_many(models, periods=14, seed=7, history_uncertainty="analytic")
    _equal(serial, predict_many(models, periods=14, seed=7, n_jobs=3, history_uncertainty="analytic"))
    m = models["s3"]
    ref = predict(m, m.make_future_dataframe(14), history_uncertainty="analytic", rng=series_rng(7, "s3"))
    pd.testing.assert_frame_equal(serial["s3"], ref)


# F3.4: models with the same configuration and future share their feature matrices.

FC_HOLIDAYS = pd.DataFrame({"holiday": "promo", "ds": pd.date_range("2020-02-01", periods=8, freq="90D"),
                            "lower_window": 0, "upper_window": 1})


def _feature_cache_model(sid):
    from prophet import Prophet

    kind = sid.split("_")[0]
    if kind == "mult":
        return Prophet(seasonality_mode="multiplicative", holidays=FC_HOLIDAYS)
    if kind == "otherhol":
        return Prophet(holidays=FC_HOLIDAYS.assign(ds=FC_HOLIDAYS["ds"] + pd.Timedelta(days=3)))
    if kind == "cond":
        m = Prophet(weekly_seasonality=False)
        m.add_seasonality("weekly_on", period=7, fourier_order=3, condition_name="on")
        return m
    if kind == "pred":
        m = Prophet(uncertainty_samples=100)
        m.add_regressor("x", regressor_predictor=True)
        return m
    m = Prophet(holidays=FC_HOLIDAYS)  # "common" (same x for all) and "own" (own x)
    m.add_country_holidays("US")
    m.add_regressor("x")
    return m


def _feature_cache_columns(sid, ds):
    kind, i = sid.split("_")
    t = np.arange(len(ds))
    if kind in ("common", "pred"):
        return {"x": 5 + np.sin(2 * np.pi * t / 7) + 0.001 * t}
    if kind == "own":
        return {"x": np.random.default_rng(int(i)).normal(size=len(ds))}
    if kind == "cond":
        return {"on": (ds.dt.month < 7).to_numpy() if i == "0" else (ds.dt.dayofweek < 5).to_numpy()}
    return {}


@pytest.fixture(scope="module")
def mixed():
    ids = ["common_0", "common_1", "common_2", "own_0", "own_1", "cond_0", "cond_1", "cond_2", "mult_0",
           "mult_1", "otherhol_0", "pred_0"]
    frames = {}
    for i, sid in enumerate(ids):
        f = make_series(n_days=500 if sid != "mult_1" else 450, seed=i)
        if sid.startswith("mult"):
            f["y"] += 30
        frames[sid] = f.assign(**_feature_cache_columns(sid, f["ds"]))
    long = pd.concat([f.assign(unique_id=k) for k, f in frames.items()], ignore_index=True)
    models = fit_many(long, make_model=_feature_cache_model)
    futures = {}
    for sid, m in models.items():
        fut = m.make_future_dataframe(21)
        futures[sid] = fut.assign(**_feature_cache_columns(sid, fut["ds"]))
    return models, futures


class TestFeatureCache:
    @pytest.mark.parametrize("hu", [True, "analytic", False])
    def test_same_forecasts_as_without_cache(self, mixed, monkeypatch, hu):
        import api

        models, futures = mixed
        cached = predict_many(models, futures=futures, seed=3, history_uncertainty=hu)
        monkeypatch.setattr(api.MODULE, "_MAX_FEATURE_ENTRIES", 0)
        plain = predict_many(models, futures=futures, seed=3, history_uncertainty=hu)
        for sid in models:
            pd.testing.assert_frame_equal(cached[sid], plain[sid], check_exact=True)

    def test_parallel_same_as_serial(self, mixed):
        models, futures = mixed
        _equal(predict_many(models, futures=futures, seed=3),
               predict_many(models, futures=futures, seed=3, n_jobs=3))

    def test_features_are_reused(self, mixed, monkeypatch):
        import api

        models, futures = mixed
        built = []
        original = api.MODULE._build_features
        monkeypatch.setattr(api.MODULE, "_build_features", lambda m, df: built.append(1) or original(m, df))
        predict_many(models, futures=futures, seed=3)
        # two builds per series without reuse; the 3 "common" series share theirs
        assert 0 < len(built) <= 2 * len(models) - 4

    def test_models_left_alone(self, mixed):
        from prophet import Prophet

        models, futures = mixed
        before = {sid: (type(m), set(vars(m))) for sid, m in models.items()}
        predict_many(models, futures=futures, seed=3)
        assert all(type(m) is Prophet for m in models.values())
        assert {sid: (type(m), set(vars(m))) for sid, m in models.items()} == before

    def test_subclass_with_own_features_is_not_cached(self, monkeypatch):
        import api
        from prophet import Prophet

        class Custom(Prophet):
            def make_holiday_features(self, dates, holidays):
                return super().make_holiday_features(dates, holidays)

        cache = api.MODULE._FeatureCache()
        m = Custom()
        assert cache.attach(m) is m
        assert type(cache.attach(Prophet())) is not Prophet
