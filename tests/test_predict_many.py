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
