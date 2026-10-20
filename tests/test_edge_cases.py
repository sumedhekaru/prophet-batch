"""D1.3.1: degenerate series behave in fit_many exactly as Prophet.fit allows.

If Prophet.fit raises, fit_many raises the same error (naming the series). Otherwise fit_many
fits the series: either exactly as Prophet.fit (fallback, constant rule) or batched, with finite
forecasts and a log posterior at least Stan's.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from api import fit_many

DAYS = pd.date_range("2020-01-01", periods=400, freq="D")


def _frame(y, n=None):
    n = len(y) if n is None else n
    return pd.DataFrame({"ds": DAYS[:n], "y": np.asarray(y, dtype=float)})


def _cases():
    rng = np.random.default_rng(0)
    spike = np.zeros(300)
    spike[150] = 100.0
    mostly_nan = np.full(300, np.nan)
    mostly_nan[[10, 120, 250]] = [1.0, 4.0, 2.0]
    return {
        "all_zero": _frame(np.zeros(200)),
        "constant": _frame(np.full(200, 7.0)),
        "negative_constant": _frame(np.full(200, -3.0)),
        "one_row": _frame([5.0]),
        "two_rows": _frame([1.0, 2.0]),
        "three_rows": _frame([1.0, 3.0, 2.0]),
        "five_rows": _frame([1.0, 0.0, 4.0, 2.0, 3.0]),
        "two_non_nan": _frame([np.nan, 1.0, np.nan, np.nan, 2.0, np.nan]),
        "mostly_nan": _frame(mostly_nan),
        "single_spike": _frame(spike),
        "sparse_counts": _frame(rng.poisson(0.2, 400)),
        "zeros_then_sales": _frame(np.r_[np.zeros(300), rng.poisson(3, 100)]),
    }


CASES = _cases()


def _outcome(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fn(), None
        except Exception as e:  # compare errors, not just success
            return None, (type(e), str(e))


@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
@pytest.mark.parametrize("name", list(CASES))
def test_same_behavior_as_prophet_fit(name, mode):
    df = CASES[name]
    stock, stock_err = _outcome(lambda: Prophet(seasonality_mode=mode).fit(df))
    ours, ours_err = _outcome(
        lambda: fit_many(df.assign(unique_id="a"), seasonality_mode=mode)["a"])
    if stock_err:  # same error, with the series named (fit_many on_error='raise', D1.4.6)
        assert ours_err == (stock_err[0], f"series 'a': {stock_err[1]}")
        return
    assert ours_err is None
    if ours.stan_fit is not None or ours.history["y"].nunique() == 1:
        for k in stock.params:  # fallback or constant rule: exactly Prophet.fit
            np.testing.assert_array_equal(ours.params[k], stock.params[k], err_msg=k)
    else:
        assert ours.params["lp__"][0, 0] >= stock.params["lp__"][0, 0] - 1e-4 * max(
            1.0, abs(stock.params["lp__"][0, 0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = ours.predict(ours.make_future_dataframe(14))
    assert np.isfinite(fc[["yhat", "yhat_lower", "yhat_upper"]].to_numpy()).all()


@pytest.mark.parametrize("mode", ["additive", "multiplicative"])
def test_valid_edge_cases_together_in_one_batch(mode):
    """All edge cases Prophet accepts, in one call: same results as fitting each alone."""
    valid = {k: v for k, v in CASES.items() if k != "one_row"}
    long = pd.concat([df.assign(unique_id=k) for k, df in valid.items()], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        together = fit_many(long, seasonality_mode=mode)
        for k, df in valid.items():
            alone = fit_many(df.assign(unique_id=k), seasonality_mode=mode)[k]
            for p in alone.params:
                np.testing.assert_array_equal(together[k].params[p], alone.params[p], err_msg=f"{k}.{p}")
