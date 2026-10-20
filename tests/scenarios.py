"""Prophet configurations for the package tests (fit with whichever prophet is installed)."""

import numpy as np
import pandas as pd


def make_series(n_days=600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    y = (10 + 0.01 * t + 2 * np.sin(2 * np.pi * t / 7) + 3 * np.sin(2 * np.pi * t / 365.25)
         + rng.normal(0, 0.5, n_days))
    return pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=n_days), 'y': y})


def _linear(Prophet):
    return Prophet(), make_series(), {}


def _multiplicative(Prophet):
    df = make_series()
    df['y'] += 20
    return Prophet(seasonality_mode='multiplicative'), df, {}


def _logistic(Prophet):
    df = make_series()
    df['cap'] = df['y'].max() * 1.2
    return Prophet(growth='logistic'), df, {'cap': df['cap'].iloc[0]}


def _flat(Prophet):
    return Prophet(growth='flat'), make_series(), {}


def _holidays_regressor(Prophet):
    df = make_series()
    df['x'] = np.random.default_rng(4).normal(size=len(df))
    df['y'] += 0.8 * df['x']
    holidays = pd.DataFrame({'holiday': 'promo', 'ds': pd.date_range('2020-02-01', periods=8, freq='90D'),
                             'lower_window': 0, 'upper_window': 1})
    m = Prophet(holidays=holidays)
    m.add_regressor('x')
    return m, df, {'x': True}


def _not_vectorized(Prophet):
    return Prophet(uncertainty_samples=100), make_series(n_days=300), {'vectorized': False}


def _regressor_predictor(Prophet):
    df = make_series(n_days=300)
    t = np.arange(len(df))
    df['x'] = 5 + 0.01 * t + np.sin(2 * np.pi * t / 7)
    df['y'] += 0.8 * df['x']
    m = Prophet(uncertainty_samples=100)
    m.add_regressor('x', regressor_predictor=True)
    return m, df, {}


def _short(Prophet):
    return Prophet(), make_series(n_days=60), {}


SCENARIOS = {f.__name__[1:]: f for f in [_linear, _multiplicative, _logistic, _flat, _holidays_regressor,
                                         _not_vectorized, _regressor_predictor, _short]}


def build(name, Prophet):
    """Fit scenario `name`; return (model, future frame incl. history + 20 periods, predict kwargs)."""
    m, df, opts = SCENARIOS[name](Prophet)
    m.fit(df)
    future = m.make_future_dataframe(20)
    if 'cap' in opts:
        future['cap'] = opts['cap']
    if opts.get('x'):
        future['x'] = np.random.default_rng(5).normal(size=len(future))
    kwargs = {'vectorized': opts['vectorized']} if 'vectorized' in opts else {}
    return m, future, kwargs


def params_owners(m):
    """The fitted models whose params determine m's forecasts: m and any regressor predictors."""
    owners = {'model': m}
    for name, props in m.extra_regressors.items():
        if props.get('predictor') is not None:
            owners[f'predictor_{name}'] = props['predictor']
    return owners
