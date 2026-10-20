"""Save the fork's seeded forecasts as fixtures (run in the dev env, where prophet is the prophet-dev fork).

    PYTHONPATH=packages/prophet-batch/src uv run python packages/prophet-batch/tests/make_fixtures.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
from prophet import Prophet

sys.path.insert(0, str(Path(__file__).parent))
from scenarios import SCENARIOS, build, params_owners  # noqa: E402

MODES = {'true': True, 'false': False, 'analytic': 'analytic'}
# The columns that depend on the random draws (the rest is checked against stock predict directly).
RANDOM_COLUMNS = ['yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper']
OUT = Path(__file__).parent / 'fixtures' / 'seeded.npz'


def main():
    logging.getLogger('cmdstanpy').disabled = True
    import inspect

    assert 'rng' in inspect.signature(Prophet.predict).parameters, 'needs the prophet-dev fork'
    arrays = {}
    for name in SCENARIOS:
        m, future, kwargs = build(name, Prophet)
        # Fitted params, so the test can reproduce the forecasts without Stan (whose fits differ across platforms).
        for key, model in params_owners(m).items():
            for k, v in model.params.items():
                arrays[f'{name}/params/{key}/{k}'] = np.asarray(v)
        for key, mode in MODES.items():
            fc = m.predict(future, history_uncertainty=mode, rng=np.random.default_rng(11), **kwargs)
            for col in RANDOM_COLUMNS:
                arrays[f'{name}/{key}/{col}'] = fc[col].to_numpy(dtype=float)
    np.savez_compressed(OUT, **arrays)
    print(f'wrote {OUT}: {len(arrays)} arrays')


if __name__ == '__main__':
    main()
