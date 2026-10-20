"""Smoke test for an installed prophet-batch (run outside the repo, e.g. against a built wheel).

    uv run --isolated --no-project --python 3.10 --with prophet==1.5.0 --with dist/<wheel> python tools/smoke.py
"""

import logging

import numpy as np
import pandas as pd
import prophet
import prophet_batch
from prophet_batch import cross_validation_many, fit_many, predict, predict_many


def series(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 10 + 0.02 * t + 2 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, n)
    return pd.DataFrame({'ds': pd.date_range('2022-01-01', periods=n), 'y': y})


def main():
    logging.getLogger('cmdstanpy').disabled = True
    assert 'site-packages' in prophet_batch.__file__, f'not the installed package: {prophet_batch.__file__}'
    long_df = pd.concat([series(400 + 20 * i, i).assign(unique_id=f's{i}') for i in range(6)], ignore_index=True)
    models = fit_many(long_df, n_jobs=2)
    fc = predict_many(models, periods=14, seed=1, n_jobs=2, history_uncertainty='analytic')
    again = predict_many(models, periods=14, seed=1, history_uncertainty='analytic')
    for sid in models:
        pd.testing.assert_frame_equal(fc[sid], again[sid])
        assert fc[sid][['yhat', 'yhat_lower', 'yhat_upper']].notna().all().all()
    one = predict(models['s0'], models['s0'].make_future_dataframe(14), rng=np.random.default_rng(0))
    cv = cross_validation_many({k: models[k] for k in ['s0', 's1']}, horizon='14 days', period='60 days',
                               initial='250 days', seed=1)
    assert all(len(df) > 0 for df in cv.values())
    print(f'ok: prophet {prophet.__version__}, prophet-batch {prophet_batch.__version__}, '
          f'{len(models)} series, forecast rows {len(one)}, cv rows {sum(len(d) for d in cv.values())}')


if __name__ == '__main__':
    main()
