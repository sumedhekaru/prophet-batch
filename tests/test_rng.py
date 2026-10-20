"""C7.1: predict(rng=...) uses only the given Generator."""

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from scenarios import SCENARIOS, _future
from api import predict, predictive_samples  # noqa: E402

UNCERTAIN = [n for n in SCENARIOS if n != "linear_no_uncertainty"]


def _fitted(name):
    m, df, fit_kwargs, *opts = SCENARIOS[name](Prophet)
    opts = opts[0] if opts else {}
    m.fit(df, **fit_kwargs)
    fut = _future(m, include_history=True, df=df, future_x=opts.get("future_x", True))
    kw = {k: v for k, v in opts.items() if k == "vectorized"}
    return m, fut, kw


@pytest.mark.parametrize("name", UNCERTAIN)
def test_same_seed_same_forecast_and_global_state_untouched(name):
    m, fut, kw = _fitted(name)
    np.random.seed(7)
    before = np.random.get_state()
    f1 = predict(m, fut, rng=np.random.default_rng(123), **kw)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:], \
        "global RNG was used"
    f2 = predict(m, fut, rng=np.random.default_rng(123), **kw)
    pd.testing.assert_frame_equal(f1, f2)
    f3 = predict(m, fut, rng=np.random.default_rng(124), **kw)
    assert not np.array_equal(f1["yhat_upper"].to_numpy(), f3["yhat_upper"].to_numpy())
    # Point forecasts don't depend on the RNG (except regressor-model point forecasts, which don't either).
    np.testing.assert_array_equal(f1["yhat"].to_numpy(), f3["yhat"].to_numpy())


def test_predictive_samples_accepts_rng():
    m, fut, _ = _fitted("linear")
    a = predictive_samples(m, fut, rng=np.random.default_rng(5))
    b = predictive_samples(m, fut, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(a["yhat"], b["yhat"])
