"""C7.2: series_rng(seed, series_id) gives a reproducible stream per series."""

import subprocess
import sys

import numpy as np
import pytest
from api import MODULE_NAME, series_rng, stable_id_hash


def draws(seed, sid, n=5):
    return series_rng(seed, sid).normal(size=n)


def test_same_seed_and_id_same_stream():
    np.testing.assert_array_equal(draws(1, "FOODS_1_001"), draws(1, "FOODS_1_001"))


def test_different_ids_and_seeds_differ():
    a = draws(1, "a")
    assert not np.array_equal(a, draws(1, "b"))
    assert not np.array_equal(a, draws(2, "a"))


def test_order_and_context_independent():
    ids = [f"s{i}" for i in range(50)]
    forward = {i: draws(3, i) for i in ids}
    backward = {i: draws(3, i) for i in reversed(ids)}
    for i in ids:
        np.testing.assert_array_equal(forward[i], backward[i])


def test_same_in_another_process():
    code = f"from {MODULE_NAME} import series_rng; print(repr(series_rng(9, ('store', 7)).random()))"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert float(out) == series_rng(9, ("store", 7)).random()
    assert stable_id_hash("x") == stable_id_hash("x")


def test_id_types():
    assert stable_id_hash(5) == stable_id_hash(np.int64(5))
    assert stable_id_hash(5) != stable_id_hash("5")
    assert stable_id_hash(("a", 1)) != stable_id_hash(("a1",))
    with pytest.raises(TypeError):
        stable_id_hash(object())
