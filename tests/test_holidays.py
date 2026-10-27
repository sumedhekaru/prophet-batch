"""0.1.3: prophet-batch's vectorized make_holiday_features equals Prophet's, bit for bit."""

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from prophet_batch import _holidays
from prophet_batch._holidays import fast_holidays, make_holiday_features

ORIGINAL = Prophet.make_holiday_features


def _both(model_kwargs, dates, holidays, train_names=None):
    out = []
    for fn in (ORIGINAL, make_holiday_features):
        m = Prophet(**model_kwargs)
        if train_names is not None:
            m.train_holiday_names = pd.Series(train_names)
        hol = m.construct_holiday_dataframe(dates) if holidays is None else holidays
        try:
            res = fn(m, dates, hol)
        except Exception as e:  # the same error, or the same result
            out.append((type(e), str(e)))
            continue
        out.append((res, m.train_holiday_names))
    return out


def _assert_same(a, b):
    if isinstance(a[0], type):
        assert a == b
        return
    (fa, pa, na), ta = a
    (fb, pb, nb), tb = b
    pd.testing.assert_frame_equal(fa, fb, check_exact=True)
    assert list(fa.columns) == list(fb.columns)
    assert pa == pb and na == nb
    pd.testing.assert_series_equal(ta, tb)


def _random_holidays(rng, years, n_names):
    rows = []
    for i in range(n_names):
        lw, uw = -int(rng.integers(0, 16)), int(rng.integers(0, 16))
        for y in years:
            rows.append({"holiday": f"h{i}", "ds": pd.Timestamp(y, int(rng.integers(1, 13)), int(rng.integers(1, 29))),
                         "lower_window": lw, "upper_window": uw})
    return pd.DataFrame(rows)


class TestMakeHolidayFeatures:
    @pytest.mark.parametrize("seed", range(20))
    def test_random_weekly_and_daily(self, seed):
        rng = np.random.default_rng(seed)
        freq = ["W-SAT", "D", "6h", "MS"][seed % 4]
        start = ["1965-03-01", "2010-01-01", "2019-06-15", "1999-12-25"][seed % 4]
        dates = pd.Series(pd.date_range(start, periods=int(rng.integers(20, 400)), freq=freq))
        hol = _random_holidays(rng, range(1960, 2030), int(rng.integers(1, 7)))
        if seed % 3 == 0:
            hol["prior_scale"] = rng.choice([2.0, np.nan], len(hol))
            hol["prior_scale"] = hol.groupby("holiday")["prior_scale"].transform("first")
        _assert_same(*_both({}, dates, hol))

    def test_unsorted_and_duplicate_dates(self):
        dates = pd.Series(pd.to_datetime(["2020-12-26 00:00", "2020-12-25 06:00", "2020-12-25 00:00", "2020-12-24 00:00", "2021-01-01 00:00"]))
        hol = pd.DataFrame({"holiday": ["xmas", "ny"], "ds": pd.to_datetime(["2020-12-25", "2021-01-01"]),
                            "lower_window": [-1, 0], "upper_window": [1, 0]})
        _assert_same(*_both({}, dates, hol))

    def test_missing_and_nan_windows(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=60))
        hol = pd.DataFrame({"holiday": ["a", "b"], "ds": pd.to_datetime(["2020-01-10", "2020-02-01"])})
        _assert_same(*_both({}, dates, hol))
        hol["lower_window"] = [np.nan, -2]
        hol["upper_window"] = [1, np.nan]
        _assert_same(*_both({}, dates, hol))

    def test_predict_path_with_holidays_not_in_future(self):
        # construct_holiday_dataframe adds fit-time holidays absent now with ds = NaT
        dates = pd.Series(pd.date_range("2030-01-01", periods=30))
        hol = pd.DataFrame({"holiday": ["a"], "ds": pd.to_datetime(["2030-01-05"]),
                            "lower_window": [-1], "upper_window": [1]})
        _assert_same(*_both({"holidays": hol}, dates, None, train_names=["a", "gone"]))

    def test_country_holidays(self):
        dates = pd.Series(pd.date_range("2018-01-06", periods=150, freq="W-SAT"))
        a, b = [], []
        for fn, out in ((ORIGINAL, a), (make_holiday_features, b)):
            m = Prophet()
            m.add_country_holidays("US")
            out.append((fn(m, dates, m.construct_holiday_dataframe(dates)), m.train_holiday_names))
        _assert_same(a[0], b[0])

    def test_errors(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=10))
        bad_scale = pd.DataFrame({"holiday": ["a", "a"], "ds": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                                  "prior_scale": [1.0, 2.0]})
        _assert_same(*_both({}, dates, bad_scale))
        nonpositive = pd.DataFrame({"holiday": ["a"], "ds": pd.to_datetime(["2020-01-02"]), "prior_scale": [0.0]})
        _assert_same(*_both({}, dates, nonpositive))


class TestFastHolidays:
    def test_patches_only_while_active_and_nests(self):
        assert Prophet.make_holiday_features is ORIGINAL
        with fast_holidays():
            assert Prophet.make_holiday_features is make_holiday_features
            with fast_holidays():
                pass
            assert Prophet.make_holiday_features is make_holiday_features
        assert Prophet.make_holiday_features is ORIGINAL

    def test_restores_on_error(self):
        with pytest.raises(RuntimeError):
            with fast_holidays():
                raise RuntimeError
        assert Prophet.make_holiday_features is ORIGINAL

    def test_unknown_source_keeps_prophets(self, monkeypatch):
        monkeypatch.setattr(_holidays, "_KNOWN_SOURCES", frozenset())
        _holidays._source_known.cache_clear()
        try:
            with fast_holidays():
                assert Prophet.make_holiday_features is ORIGINAL
        finally:
            _holidays._source_known.cache_clear()

    def test_subclass_override_kept(self):
        class Mine(Prophet):
            def make_holiday_features(self, dates, holidays):
                return ORIGINAL(self, dates, holidays)

        with fast_holidays():
            assert Mine.make_holiday_features is not make_holiday_features

    def test_installed_source_is_known(self):
        # the installed Prophet is one this mirrors (else the speedup silently turns off)
        assert _holidays._source_known(ORIGINAL)
