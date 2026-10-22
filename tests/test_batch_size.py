"""0.1.2: fit_many(batch_size=...): results bit-identical for any batch size, automatic sizing, memory reader."""

import logging

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from api import BATCH_LOGGER, cross_validation_many, fit_many

from synthetic import make_series


def make_model(sid):
    """Module-level (picklable): the config depends on the series id."""
    if sid.startswith("logistic"):
        return Prophet(growth="logistic")
    if sid.startswith("mult"):
        return Prophet(seasonality_mode="multiplicative")
    return Prophet()


def _frames():
    frames = {}
    for i in range(5):
        frames[f"add{i}"] = make_series(seed=i)                       # shared dates
    frames["mult0"] = make_series(n_days=600, seed=10).assign(y=lambda d: d["y"] + 20)
    frames["const"] = make_series(n_days=300).assign(y=3.0)
    lg = make_series(n_days=500, seed=20)
    frames["logistic0"] = lg.assign(cap=lg["y"].max() * 1.2)
    frames["short"] = make_series(n_days=250, seed=21)
    frames["add5"] = make_series(n_days=420, seed=22)
    return frames


def _long(frames):
    return pd.concat([f.assign(unique_id=k) for k, f in frames.items()], ignore_index=True)


def _assert_identical(a, b):
    assert list(a) == list(b)
    for sid in a:
        assert a[sid].params.keys() == b[sid].params.keys(), sid
        for k in a[sid].params:
            np.testing.assert_array_equal(a[sid].params[k], b[sid].params[k], err_msg=f"{sid}.{k}")


@pytest.fixture(scope="module")
def reference():
    """Every series in one batch, in this process (0.1.1's behaviour)."""
    df = _long(_frames())
    return df, fit_many(df, make_model=make_model, batch_size=10**6)


@pytest.mark.parametrize("batch_size", [1, 2, 5, None])
def test_in_process_identical_for_any_batch_size(reference, batch_size):
    df, ref = reference
    _assert_identical(fit_many(df, make_model=make_model, batch_size=batch_size), ref)


@pytest.mark.parametrize("batch_size", [1, 3, None])
def test_parallel_identical_for_any_batch_size(reference, batch_size):
    df, ref = reference
    _assert_identical(fit_many(df, make_model=make_model, n_jobs=3, batch_size=batch_size), ref)


def test_prophet_kwargs_path_identical():
    frames = {k: v for k, v in _frames().items() if k.startswith("add")}
    df = _long(frames)
    ref = fit_many(df, yearly_seasonality=3, batch_size=10**6)
    _assert_identical(fit_many(df, yearly_seasonality=3, batch_size=2), ref)
    _assert_identical(fit_many(df, yearly_seasonality=3, n_jobs=2, batch_size=2), ref)


def test_skipped_series_and_order_preserved(caplog):
    frames = _frames()
    frames = {**dict(list(frames.items())[:3]), "bad": make_series(n_days=1), **dict(list(frames.items())[3:])}
    df = _long(frames)
    ref = fit_many(df, make_model=make_model, on_error="skip", batch_size=10**6)
    assert "bad" not in ref
    for kwargs in ({"batch_size": 2}, {"batch_size": 1, "n_jobs": 2}):
        _assert_identical(fit_many(df, make_model=make_model, on_error="skip", **kwargs), ref)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "5"])
def test_invalid_batch_size(bad):
    df = _long({"a": make_series()})
    with pytest.raises(ValueError, match="batch_size"):
        fit_many(df, batch_size=bad)
    with pytest.raises(ValueError, match="batch_size"):
        cross_validation_many(fit_many(df), horizon="30 days", batch_size=bad, seed=0)


def test_cross_validation_identical_for_batch_size():
    frames = {k: v for k, v in _frames().items() if k.startswith("add")}
    models = fit_many(_long(frames))
    kw = dict(horizon="30 days", period="90 days", initial="300 days", seed=1)
    ref = cross_validation_many(models, batch_size=10**6, **kw)
    for batch_size in (1, None):
        got = cross_validation_many(models, batch_size=batch_size, **kw)
        assert list(got) == list(ref)
        for sid in ref:
            pd.testing.assert_frame_equal(got[sid], ref[sid], check_exact=True)


def test_batches_logged_once(caplog):
    df = _long({k: v for k, v in _frames().items() if k.startswith("add")})
    with caplog.at_level(logging.INFO, logger=BATCH_LOGGER):
        fit_many(df, batch_size=2)
    lines = [r.getMessage() for r in caplog.records if r.name == BATCH_LOGGER and r.levelno >= logging.INFO]
    assert lines == ["fit_many: 6 series in 3 batches in this process (batch_size=2)"]


class TestPlan:
    """How fit_many sizes its batches."""

    def test_count_cap_and_order(self):
        from api import _plan_batches

        ids = list("abcdefg")
        rows = dict.fromkeys(ids, 10)
        assert _plan_batches(ids, rows, 3, None, None) == [list("abc"), list("def"), ["g"]]

    def test_memory_budget(self):
        from api import _plan_batches

        ids = list("abcde")
        rows = {"a": 10, "b": 10, "c": 50, "d": 10, "e": 10}
        # 8 bytes per row, budget 200 bytes: a+b fit (160), c alone (400 > budget, still one series)
        assert _plan_batches(ids, rows, 100, 200.0, 8.0) == [["a", "b"], ["c"], ["d", "e"]]

    def test_automatic_count_and_worker_rule(self, monkeypatch):
        from api import MODULE, _batch_plan

        monkeypatch.setattr(MODULE, "_available_memory", lambda: None)
        frames = {f"s{i}": make_series(n_days=200, seed=i) for i in range(30)}
        df = _long(frames)
        rows_by_id = df.groupby("unique_id", sort=False).indices
        ids = list(rows_by_id)
        args = (df, "unique_id", ids, rows_by_id, lambda sid: Prophet(), lambda sid: {})
        batches, note = _batch_plan(*args, None, 2)        # 2 workers: at least 8 tasks
        assert [len(b) for b in batches] == [4] * 7 + [2] and "count-limited" in note
        batches, _ = _batch_plan(*args, None, 1)           # one process: the cap (100)
        assert [len(b) for b in batches] == [30]
        batches, note = _batch_plan(*args, 7, 2)           # given: exactly that
        assert [len(b) for b in batches] == [7, 7, 7, 7, 2] and note == "batch_size=7"

    def test_automatic_memory_limited(self, monkeypatch, caplog):
        from api import MODULE, _batch_plan

        frames = {f"s{i}": make_series(n_days=400, seed=i) for i in range(12)}
        df = _long(frames)
        rows_by_id = df.groupby("unique_id", sort=False).indices
        ids = list(rows_by_id)
        # 400 rows × (6 feature + 33 design + 8) columns × 8 bytes ≈ 150 KB per series; half of 700 KB on one
        # process ≈ 2 series per batch
        monkeypatch.setattr(MODULE, "_available_memory", lambda: 700_000)
        batches, note = _batch_plan(df, "unique_id", ids, rows_by_id, lambda sid: Prophet(), lambda sid: {},
                                    None, 1)
        assert [len(b) for b in batches] == [2] * 6 and "memory-limited" in note
        assert [s for b in batches for s in b] == ids
        ref = fit_many(df, batch_size=10**6)
        _assert_identical(fit_many(df), ref)            # memory-limited automatic batches: same fits


class TestAvailableMemory:
    @staticmethod
    def _available(tmp_path, files):
        from api import _cgroup_memory_available

        for name, text in files.items():
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(text)
        return _cgroup_memory_available(str(tmp_path))

    @pytest.mark.parametrize("files, expected", [
        ({"memory.max": "4294967296\n", "memory.current": "1073741824\n"}, 3 << 30),      # cgroup v2
        ({"memory.max": "max\n", "memory.current": "1073741824\n"}, None),               # no limit
        ({"memory/memory.limit_in_bytes": "2147483648\n", "memory/memory.usage_in_bytes": "536870912\n"},
         3 << 29),                                                                        # cgroup v1
        ({"memory/memory.limit_in_bytes": "9223372036854771712\n", "memory/memory.usage_in_bytes": "1\n"},
         None),                                                                           # v1, no limit
        ({"memory.max": "1000\n", "memory.current": "2000\n"}, 0),                        # over the limit
        ({}, None),
    ])
    def test_cgroup(self, tmp_path, files, expected):
        assert self._available(tmp_path, files) == expected

    def test_this_machine(self):
        """Linux, macOS and Windows (CI) all report a positive amount."""
        from api import _available_memory

        avail = _available_memory()
        assert avail is not None and avail > 0


def test_stan_classification_budget():
    """Scheduling only: with the time budget used up, the remaining series are scheduled with the rest."""
    from api import _split_stan_series

    ids = ["logistic0", "add0", "logistic1", "add1"]
    assert _split_stan_series(ids, make_model, {}, budget=None) == (["logistic0", "logistic1"], ["add0", "add1"])
    assert _split_stan_series(ids, make_model, {}, budget=-1.0) == ([], ids)        # no time: nothing classified
