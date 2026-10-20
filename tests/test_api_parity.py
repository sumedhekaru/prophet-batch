"""E1.2.1: the stock API works on fit_many models and matches stock (see api_parity.py).

Plotly checks are skipped unless plotly is importable: `uv run --with plotly pytest tests/test_api_parity.py`.
"""

import logging

import pytest

import api_parity


def test_harness_detects_a_real_difference():
    assert api_parity.detects_difference()


@pytest.mark.parametrize("kind", list(api_parity.KINDS))
def test_api_matches_stock(kind):
    logging.getLogger("cmdstanpy").disabled = True
    gaps = {f"{w}.{n}": d for (w, n), (status, d) in api_parity.run(kind).items() if status == "✗"}
    assert not gaps, gaps
