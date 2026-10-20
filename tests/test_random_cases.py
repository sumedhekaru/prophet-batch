"""D1.1.1: the seeded case generator is deterministic and its specs are plain JSON."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from conftest import machine_baseline
from random_cases import build, coverage, make_spec, spec_json


@pytest.mark.parametrize("seed", range(10))
def test_same_seed_same_case(seed):
    assert spec_json(seed) == spec_json(seed)
    _, df1, fut1 = build(make_spec(seed), Prophet)
    _, df2, fut2 = build(make_spec(seed), Prophet)
    pd.testing.assert_frame_equal(df1, df2, check_exact=True)
    pd.testing.assert_frame_equal(fut1, fut2, check_exact=True)


def test_spec_round_trips_through_json():
    for seed in range(50):
        spec = make_spec(seed)
        assert json.loads(json.dumps(spec)) == spec


def test_different_seeds_differ():
    assert len({spec_json(s) for s in range(50)}) == 50


def test_every_option_value_is_generated():
    c = coverage([make_spec(s) for s in range(200)])
    expected = [
        ("growth", "linear"), ("growth", "flat"), ("growth", "logistic"),
        ("seasonality_mode", "additive"), ("seasonality_mode", "multiplicative"),
        ("changepoints", "n"), ("changepoints", "explicit"), ("changepoints", "none"),
        ("scaling", "absmax"), ("scaling", "minmax"), ("conditional_seasonality", True),
        ("regressor_predictor", True), ("constant", True), ("negative", True), ("nans", True),
        ("outliers", True), ("scale", "tiny"), ("scale", "huge"), ("length", "short"),
        ("holiday_prior_scale", True), ("country_holidays", "US"), ("daily_seasonality", "int"),
    ]
    assert not [k for k in expected if c[k] == 0]


def test_cases_fit_and_predict():
    for seed in (0, 1, 2):
        model, df, future = build(make_spec(seed), Prophet)
        fc = model.fit(df).predict(future)
        assert len(fc) == len(future) and fc["yhat"].notna().all()


# D1.2.1: other frequencies, without changing the daily cases.

def _frame_bytes(df):
    out = b""
    for c in df.columns:
        v = (df[c].to_numpy(dtype="datetime64[ns]").astype(np.int64) if c == "ds"
             else np.asarray(df[c].to_numpy(), dtype=float))
        out += c.encode() + v.tobytes()
    return out


@machine_baseline
def test_daily_cases_unchanged_since_d1_1():
    """Pinned before frequencies were added: existing seeds keep their meaning.

    The specs and data go through libm (log, exp, sin), so the hash is per platform.
    """
    specs = hashlib.sha256("".join(spec_json(s) for s in range(200)).encode()).hexdigest()
    assert specs == "e0bb4363bdfaae9be14d24f6679d4892512ba818731d0a454ba02e065f364f53"
    data = hashlib.sha256()
    for s in range(20):
        _, df, fut = build(make_spec(s), Prophet)
        data.update(_frame_bytes(df))
        data.update(_frame_bytes(fut))
    assert data.hexdigest() == "a96552c45aaef071ad9dae14ce64a922c88e9a9f4ff773793bb382ed829cd2b3"


@pytest.mark.parametrize("freq,step", [("H", "1h"), ("W", "7D")])
def test_other_frequencies_are_deterministic_and_regular(freq, step):
    for seed in range(5):
        assert spec_json(seed, freq) == spec_json(seed, freq)
        _, df, fut = build(make_spec(seed, freq), Prophet)
        _, df2, _ = build(make_spec(seed, freq), Prophet)
        pd.testing.assert_frame_equal(df, df2, check_exact=True)
        assert (df["ds"].diff().dropna() == pd.Timedelta(step)).all()
        assert fut["ds"].max() > df["ds"].max()


def test_monthly_dates_are_month_starts():
    _, df, fut = build(make_spec(3, "MS"), Prophet)
    assert (df["ds"].dt.day == 1).all() and (fut["ds"].dt.day == 1).all()
