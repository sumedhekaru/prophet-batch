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
from predict_scenarios import SCENARIOS, build, params_owners  # noqa: E402

logging.getLogger('cmdstanpy').disabled = True
FORK = 'rng' in inspect.signature(Prophet.predict).parameters
FIXTURES = np.load(Path(__file__).parent / 'fixtures' / 'seeded.npz')
MODES = {'true': True, 'sampled': 'sampled', 'false': False, 'analytic': 'analytic'}


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


FORK_SOURCE = Path(__file__).resolve().parents[2] / 'prophet-dev' / 'python' / 'prophet' / 'forecaster.py'


def test_logistic_paths_match_prophet():
    """The seeded copy computes logistic trend paths only where a slope changes: the
    same draws as the installed Prophet's _logistic_uncertainty (every step)."""
    rng = np.random.default_rng(1)
    view = seeded(Prophet(growth='logistic'))
    for _ in range(40):
        hist, horizon, n_cp = (int(rng.choice(c)) for c in ([30, 365, 1500], [1, 28, 200], [0, 5, 25]))
        sd = 1.0 / hist
        view.changepoints_t = np.sort(rng.uniform(0.02, 0.8, n_cp))
        deltas = rng.laplace(0, 0.05, n_cp)
        deltas[rng.random(n_cp) < 0.3] = 0.0
        mat = view._make_trend_shift_matrix(np.mean(np.abs(deltas)) + 1e-8 if n_cp else 1e-8, n_cp * sd,
                                            horizon, n_samples=200, rng=rng)
        args = dict(mat=mat, deltas=deltas, k=float(rng.normal(0, 1)), m=float(rng.uniform(-0.5, 1)),
                    cap=rng.uniform(0.5, 3) * np.ones(horizon), t_time=1 + np.arange(1, horizon + 1) * sd,
                    n_length=horizon, single_diff=sd)
        expected = Prophet._logistic_uncertainty(view, **args)
        assert np.array_equal(view._logistic_uncertainty(**args), expected)


@pytest.mark.skipif(not FORK_SOURCE.exists(), reason='prophet-dev checkout not available')
def test_generated_module_is_in_sync_with_fork():
    script = Path(__file__).parents[1] / 'tools' / 'sync_seeded.py'
    out = subprocess.run([sys.executable, str(script), '--check'], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


class _ClippedProphet(Prophet):
    def predict(self, df=None, vectorized=True):
        fc = super().predict(df, vectorized)
        fc['yhat'] = fc['yhat'].clip(lower=0)
        return fc


def test_subclass_predict_override_warns():
    m, df, _ = SCENARIOS['short'](_ClippedProphet)
    m.fit(df)
    with pytest.warns(UserWarning, match='_ClippedProphet overrides predict'):
        predict(m, m.make_future_dataframe(5), rng=np.random.default_rng(0))
