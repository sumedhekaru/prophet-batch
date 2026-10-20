"""D4.1.2: prophet_batch.predict on stock Prophet equals the fork's seeded predict."""

import copy
import inspect
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from prophet_batch import predict
from prophet_batch._seeded import seeded

sys.path.insert(0, str(Path(__file__).parent))
from scenarios import SCENARIOS, build, params_owners  # noqa: E402

logging.getLogger('cmdstanpy').disabled = True
FORK = 'rng' in inspect.signature(Prophet.predict).parameters
FIXTURES = np.load(Path(__file__).parent / 'fixtures' / 'seeded.npz')
MODES = {'true': True, 'false': False, 'analytic': 'analytic'}


@pytest.fixture(scope='module', params=list(SCENARIOS))
def fitted(request):
    return (request.param, *build(request.param, Prophet))


def test_default_is_stock_predict(fitted):
    name, m, future, kwargs = fitted
    np.random.seed(3)
    ref = m.predict(future, **kwargs)
    np.random.seed(3)
    ours = predict(m, future, rng=None, **kwargs)
    pd.testing.assert_frame_equal(ours, ref)
    np.random.seed(3)
    pd.testing.assert_frame_equal(seeded(m).predict(future, **kwargs), ref)


def _with_fixture_params(name, m):
    """A copy of m with the fork's fitted params (Stan's fits differ across platforms)."""
    m = copy.deepcopy(m)
    for key, model in params_owners(m).items():
        prefix = f'{name}/params/{key}/'
        model.params = {k[len(prefix):]: FIXTURES[k] for k in FIXTURES.files if k.startswith(prefix)}
    return m


@pytest.mark.parametrize('mode', list(MODES))
def test_seeded_matches_fork(fitted, mode):
    name, m, future, kwargs = fitted
    if FORK:  # bit-identical to the fork's own predict, same model
        fc = seeded(m).predict(future, history_uncertainty=MODES[mode], rng=np.random.default_rng(11), **kwargs)
        ref = m.predict(future, history_uncertainty=MODES[mode], rng=np.random.default_rng(11), **kwargs)
        pd.testing.assert_frame_equal(fc, ref)
    # Against the fork's saved forecasts, with the fork's params: equal up to floating-point rounding
    # (bit-identical on the machine that made the fixtures; BLAS/libm may differ in the last bits elsewhere).
    m = _with_fixture_params(name, m)
    fc = seeded(m).predict(future, history_uncertainty=MODES[mode], rng=np.random.default_rng(11), **kwargs)
    for col in ['yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper']:
        np.testing.assert_allclose(fc[col].to_numpy(dtype=float), FIXTURES[f'{name}/{mode}/{col}'],
                                   rtol=1e-9, atol=1e-9, err_msg=f'{name}/{mode}/{col}')
    # Everything else does not depend on the draws: equal to the default (stock) predict.
    other = [c for c in fc.columns if c not in ('yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper')]
    pd.testing.assert_frame_equal(fc[other], m.predict(future, **kwargs)[other])


def test_seeded_view_shares_state_and_leaves_model_alone():
    m, future, _ = build('linear', Prophet)
    view = seeded(m)
    assert type(m) is Prophet and view.params is m.params and view.history is m.history
    assert seeded(view) is view
    a = predict(m, future, rng=np.random.default_rng(1))
    b = predict(m, future, rng=np.random.default_rng(1))
    pd.testing.assert_frame_equal(a, b)


def test_errors():
    m, future, _ = build('short', Prophet)
    with pytest.raises(ValueError, match='history_uncertainty must be'):
        seeded(m).predict(future, history_uncertainty='x')


FORK_SOURCE = Path(__file__).parents[3] / 'prophet-dev' / 'python' / 'prophet' / 'forecaster.py'


@pytest.mark.skipif(not FORK_SOURCE.exists(), reason='prophet-dev checkout not available')
def test_generated_module_is_in_sync_with_fork():
    script = Path(__file__).parents[1] / 'tools' / 'sync_seeded.py'
    out = subprocess.run([sys.executable, str(script), '--check'], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
