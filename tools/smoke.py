"""Smoke test for an installed prophet-batch (run outside the repo, e.g. against a built wheel).

    uv run --isolated --no-project --python 3.10 --with prophet==1.5.0 --with dist/<wheel> python tools/smoke.py
"""

import logging

import numpy as np
import pandas as pd
import prophet
import prophet_batch
from prophet_batch import Pool, cross_validation_many, fit_many, forecast_many, predict, predict_many


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
    # forecast_many (0.1.2): equal to fit_many + predict_many per series; a Pool gives the same
    cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    long_fc = forecast_many(long_df, periods=14, seed=1, n_jobs=2, history_uncertainty='analytic', columns=cols)
    with Pool(n_jobs=2) as pool:
        pooled = pool.submit(long_df, periods=14, seed=1, history_uncertainty='analytic', columns=cols).result()
    for sid, g in long_fc.groupby('unique_id', sort=False):
        pd.testing.assert_frame_equal(g[cols].reset_index(drop=True), fc[sid][cols].reset_index(drop=True))
    pd.testing.assert_frame_equal(pooled, long_fc)
    # solver='stan' (0.1.3): Prophet.fit's result, Stan model loaded once (reuse_stan_model public)
    stan = fit_many(long_df[long_df['unique_id'].isin(['s0', 's1'])], solver='stan')
    with prophet_batch.reuse_stan_model():
        own = prophet.Prophet().fit(series(400, 0))
    np.testing.assert_array_equal(stan['s0'].params['k'], own.params['k'])
    # history intervals (0.1.3): order statistics by default, 'sampled' every path; same yhat, similar widths
    default = predict_many(models, periods=14, seed=1)
    sampled = predict_many(models, periods=14, seed=1, history_uncertainty='sampled')
    for sid in models:
        d, s = default[sid], sampled[sid]
        pd.testing.assert_series_equal(d['yhat'], s['yhat'])
        width = [(f['yhat_upper'] - f['yhat_lower']).iloc[:-14].mean() for f in (d, s)]
        assert 0.9 < width[0] / width[1] < 1.1, (sid, width)
    print(f'ok: prophet {prophet.__version__}, prophet-batch {prophet_batch.__version__}, '
          f'{len(models)} series, forecast rows {len(one)}, cv rows {sum(len(d) for d in cv.values())}, '
          f'forecast_many rows {len(long_fc)}')


if __name__ == '__main__':
    main()
