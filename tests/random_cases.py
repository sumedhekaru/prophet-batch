"""D1.1: seeded random Prophet cases (config + data) for property-based tests.

Each case is one integer seed. `make_spec(seed)` returns a JSON-able dict describing the
Prophet configuration and the data recipe; `build(spec, Prophet)` turns it into an unfitted
model, a history DataFrame and a future DataFrame. Same seed, same case, in any process.

Frequencies (D1.2): `make_spec(seed)` is daily, unchanged since D1.1 (seeds keep their meaning);
`make_spec(seed, freq)` with freq 'H' (hourly, with a daily cycle), 'W' (weekly) or 'MS' (month
start) adjusts that spec using a separate random stream. MCMC is not generated (too slow).
Combinations Prophet itself rejects are excluded by construction:
  - fewer than 2 non-NaN y rows (n_days >= 30, at most 20% NaN);
  - explicit changepoints outside the history (they are drawn from non-NaN interior dates);
  - NaN or constant regressors (regressors never have NaN and always take two or more values);
  - logistic growth without cap, or with cap <= floor (cap is set above the data; no floor column).
Avoided because stock Prophet 1.5.0 fails on it (upstream bug, journal 2026-10-19): a regressor_predictor
regressor left out of a future that includes history, when training y has NaN rows. For predictor
regressors the future omits the column only when y has no NaN; otherwise it carries the history values
with NaN for the future rows (the other valid usage).
No pytest dependency, so scripts and the isolated stock-Prophet env can import it.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

_START = pd.Timestamp('2015-01-01')
# Spec frequency -> pandas frequency, and the code for its separate random stream.
FREQS = {'D': ('D', 0), 'H': ('h', 1), 'W': ('W-SUN', 2), 'MS': ('MS', 3)}


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _choice(rng: np.random.Generator, options: list, p: list[float] | None = None) -> Any:
    return options[int(rng.choice(len(options), p=p))]


def _seasonality_setting(rng: np.random.Generator, p_int: float) -> str | int | bool:
    """'auto', False, or a Fourier order."""
    u = rng.uniform()
    if u < p_int:
        return int(rng.integers(1, 11))
    return 'auto' if u < p_int + (1 - p_int) * 0.7 else False


def make_spec(seed: int, freq: str = 'D') -> dict[str, Any]:
    """The case for this seed and frequency, as a JSON-able dict."""
    spec = _daily_spec(seed)
    if freq != 'D':
        _adapt_frequency(spec, freq)
    return spec


def _adapt_frequency(spec: dict[str, Any], freq: str) -> None:
    """Turn a daily spec into one for another frequency (own random stream, so daily specs
    don't change)."""
    rng = np.random.default_rng([20261019, spec['seed'], FREQS[freq][1]])
    d = spec['data']
    spec['freq'] = freq
    if freq == 'H':  # 2 to 60 days of hourly data, with a daily cycle
        d['n_days'] = int(rng.integers(48, 24 * 60))
        d['amp_daily'] = float(abs(rng.normal(0, 0.4)))
        periods = [0.5, 3.5, float(rng.uniform(0.2, 10))]
        spec['future']['periods'] = int(rng.integers(1, 24 * 7))
    elif freq == 'W':  # 20 to 400 weeks
        d['n_days'] = int(rng.integers(20, 400))
        periods = [91.3, 182.6, float(rng.uniform(20, 400))]
        spec['future']['periods'] = int(rng.integers(1, 53))
    else:  # 'MS': 15 to 150 months
        d['n_days'] = int(rng.integers(15, 150))
        periods = [91.3, 182.6, 730.5]
        spec['future']['periods'] = int(rng.integers(1, 25))
    for s in spec['seasonalities']:
        s['period'] = float(periods[int(rng.integers(0, len(periods)))])


def _daily_spec(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng([20261019, seed])
    growth = _choice(rng, ['linear', 'flat', 'logistic'], [0.55, 0.3, 0.15])
    mode = _choice(rng, ['additive', 'multiplicative'], [0.6, 0.4])
    n_days = int(_choice(rng, [int(rng.integers(30, 120)), int(rng.integers(120, 800)),
                               int(rng.integers(800, 1500))], [0.2, 0.4, 0.4]))

    cp_kind = _choice(rng, ['n', 'explicit', 'none'], [0.6, 0.2, 0.2])
    changepoints: dict[str, Any] = {'kind': cp_kind}
    if cp_kind == 'n':
        changepoints['n'] = int(rng.integers(1, 41))
    elif cp_kind == 'explicit':
        changepoints['fracs'] = sorted(float(f) for f in rng.uniform(0.05, 0.95, int(rng.integers(1, 8))))

    holidays = []
    for h in range(int(_choice(rng, [0, 1, 2, 3], [0.5, 0.25, 0.15, 0.1]))):
        holidays.append({
            'name': f'h{h}',
            'n_dates': int(rng.integers(1, 15)),
            'lower': -int(rng.integers(0, 3)),
            'upper': int(rng.integers(0, 3)),
            'prior_scale': _loguniform(rng, 0.01, 10) if rng.uniform() < 0.4 else None,
        })

    regressors = []
    for r in range(int(_choice(rng, [0, 1, 2], [0.5, 0.3, 0.2]))):
        regressors.append({
            'name': f'x{r}',
            'kind': _choice(rng, ['normal', 'binary'], [0.7, 0.3]),
            'standardize': _choice(rng, ['auto', True, False]),
            'mode': _choice(rng, [None, 'additive', 'multiplicative'], [0.5, 0.25, 0.25]),
            'prior_scale': _loguniform(rng, 0.01, 10) if rng.uniform() < 0.4 else None,
            'predictor': bool(rng.uniform() < 0.1),
            'coef': float(rng.normal(0, 1)),
        })

    seasonalities = []
    for s in range(int(_choice(rng, [0, 1, 2], [0.6, 0.3, 0.1]))):
        seasonalities.append({
            'name': f's{s}',
            'period': float(_choice(rng, [30.5, 91.3, float(rng.uniform(3, 60))])),
            'fourier_order': int(rng.integers(1, 6)),
            'prior_scale': _loguniform(rng, 0.01, 10) if rng.uniform() < 0.4 else None,
            'mode': _choice(rng, [None, 'additive', 'multiplicative'], [0.5, 0.25, 0.25]),
            'conditional': bool(rng.uniform() < 0.4),
        })

    data = {
        'n_days': n_days,
        'start_offset': int(rng.integers(0, 2000)),
        'scale': float(10 ** rng.uniform(-6, 9)),
        'negative': bool(growth != 'logistic' and rng.uniform() < 0.2),
        'constant': bool(rng.uniform() < 0.05),
        'slope': float(rng.normal(0, 1)),
        'n_kinks': int(rng.integers(0, 4)),
        'amp_weekly': float(abs(rng.normal(0, 0.3))),
        'amp_yearly': float(abs(rng.normal(0, 0.5))),
        'noise': float(_loguniform(rng, 1e-4, 0.5)),
        'nan_frac': float(_choice(rng, [0.0, float(rng.uniform(0, 0.2))], [0.6, 0.4])),
        'n_outliers': int(_choice(rng, [0, int(rng.integers(1, 6))], [0.7, 0.3])),
        'data_seed': int(rng.integers(0, 2 ** 31)),
    }
    return {
        'seed': seed,
        'growth': growth,
        'seasonality_mode': mode,
        'holidays_mode': _choice(rng, [None, 'additive', 'multiplicative'], [0.7, 0.15, 0.15]),
        'changepoints': changepoints,
        'changepoint_range': float(rng.uniform(0.5, 1.0)),
        'changepoint_prior_scale': _loguniform(rng, 0.001, 0.5),
        'seasonality_prior_scale': _loguniform(rng, 0.01, 10),
        'holidays_prior_scale': _loguniform(rng, 0.01, 10),
        'yearly_seasonality': _seasonality_setting(rng, 0.2),
        'weekly_seasonality': _seasonality_setting(rng, 0.2),
        'daily_seasonality': _seasonality_setting(rng, 0.1),
        'scaling': _choice(rng, ['absmax', 'minmax'], [0.7, 0.3]),
        'interval_width': float(rng.uniform(0.5, 0.95)),
        'uncertainty_samples': int(_choice(rng, [0, int(rng.integers(50, 301))], [0.2, 0.8])),
        'country_holidays': _choice(rng, [None, 'US'], [0.85, 0.15]),
        'holidays': holidays,
        'regressors': regressors,
        'seasonalities': seasonalities,
        'data': data,
        'future': {'periods': int(rng.integers(1, 61)), 'include_history': bool(rng.uniform() < 0.7)},
    }


def _regressor_values(rng: np.random.Generator, kind: str, n: int) -> np.ndarray:
    if kind == 'binary':
        x = (rng.uniform(size=n) < 0.3).astype(float)
        x[:2] = [0.0, 1.0]  # always two distinct values
        return x
    return rng.normal(size=n)


def _condition(ds: pd.Series, i: int) -> pd.Series:
    """Boolean condition column from the date alone, so history and future agree."""
    return (ds.dt.month % (i + 2)) == 0


def build(spec: dict[str, Any], Prophet: type) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """Unfitted model, history DataFrame and future DataFrame for a spec."""
    d = spec['data']
    rng = np.random.default_rng(d['data_seed'])
    n = d['n_days']  # number of rows (days only for daily data)
    periods = spec['future']['periods']
    ds_all = pd.Series(pd.date_range(_START + pd.Timedelta(days=d['start_offset']),
                                     periods=n + periods, freq=FREQS[spec.get('freq', 'D')][0]))
    ds = ds_all.iloc[:n].reset_index(drop=True)

    # y: trend with kinks + seasonality + noise, then outliers, NaNs, sign and scale.
    t = np.arange(n) / max(n - 1, 1)
    trend = 1.0 + d['slope'] * t
    for _ in range(d['n_kinks']):
        at = rng.uniform(0.1, 0.9)
        trend += rng.normal(0, 1) * np.maximum(t - at, 0)
    if spec.get('freq', 'D') == 'D':
        days = np.arange(n)
    else:
        days = ((ds - ds[0]) / pd.Timedelta(days=1)).to_numpy()
    season = (d['amp_weekly'] * np.sin(2 * np.pi * days / 7 + rng.uniform(0, 6.3))
              + d['amp_yearly'] * np.sin(2 * np.pi * days / 365.25 + rng.uniform(0, 6.3)))
    if 'amp_daily' in d:
        season = season + d['amp_daily'] * np.sin(2 * np.pi * days + rng.uniform(0, 6.3))
    if spec['seasonality_mode'] == 'multiplicative':
        y = trend * (1 + 0.5 * season)
    else:
        y = trend + season
    y = y + d['noise'] * rng.normal(size=n)
    if d['constant']:
        y = np.full(n, 1.5)
    df = pd.DataFrame({'ds': ds})

    future_x = {}
    for r in spec['regressors']:
        x_all = _regressor_values(rng, r['kind'], n + periods)
        df[r['name']] = x_all[:n]
        future_x[r['name']] = x_all
        if not d['constant']:
            y = y + r['coef'] * x_all[:n] * 0.3
    for i, s in enumerate(spec['seasonalities']):
        if s['conditional']:
            df[f"cond_{s['name']}"] = _condition(ds, i)

    if not d['constant']:
        for idx in rng.choice(n, size=min(d['n_outliers'], n), replace=False):
            y[idx] += rng.choice([-1, 1]) * 10 * max(d['noise'], 0.05)
    if spec['growth'] == 'logistic':
        y = y - min(y.min(), 0) + 0.1  # positive, so cap > floor = 0 is meaningful
    elif d['negative']:
        y = y - y.mean() - 0.5
    y = y * d['scale']
    if d['nan_frac'] > 0:
        nan_idx = rng.choice(n, size=int(d['nan_frac'] * n), replace=False)
        y[nan_idx] = np.nan
    df['y'] = y
    cap = None
    if spec['growth'] == 'logistic':
        cap = float(np.nanmax(y) * 1.2)
        df['cap'] = cap

    model = Prophet(
        growth=spec['growth'],
        changepoints=_changepoints(spec, df),
        n_changepoints=spec['changepoints'].get('n', 0) if spec['changepoints']['kind'] != 'explicit' else 25,
        changepoint_range=spec['changepoint_range'],
        yearly_seasonality=spec['yearly_seasonality'],
        weekly_seasonality=spec['weekly_seasonality'],
        daily_seasonality=spec['daily_seasonality'],
        holidays=_holidays(spec, ds_all, rng),
        seasonality_mode=spec['seasonality_mode'],
        seasonality_prior_scale=spec['seasonality_prior_scale'],
        holidays_prior_scale=spec['holidays_prior_scale'],
        changepoint_prior_scale=spec['changepoint_prior_scale'],
        interval_width=spec['interval_width'],
        uncertainty_samples=spec['uncertainty_samples'],
        scaling=spec['scaling'],
        holidays_mode=spec['holidays_mode'],
    )
    if spec['country_holidays']:
        model.add_country_holidays(spec['country_holidays'])
    for r in spec['regressors']:
        model.add_regressor(r['name'], prior_scale=r['prior_scale'], standardize=r['standardize'],
                            mode=r['mode'], regressor_predictor=True if r['predictor'] else None)
    for s in spec['seasonalities']:
        model.add_seasonality(s['name'], period=s['period'], fourier_order=s['fourier_order'],
                              prior_scale=s['prior_scale'], mode=s['mode'],
                              condition_name=f"cond_{s['name']}" if s['conditional'] else None)

    future = pd.DataFrame({'ds': ds_all if spec['future']['include_history']
                           else ds_all.iloc[n:].reset_index(drop=True)})
    offset = 0 if spec['future']['include_history'] else n
    for r in spec['regressors']:
        if not r['predictor']:
            future[r['name']] = future_x[r['name']][offset:]
        elif d['nan_frac'] > 0:
            x = future_x[r['name']].copy()
            x[n:] = np.nan  # predicted by the regressor's own model
            future[r['name']] = x[offset:]
    for i, s in enumerate(spec['seasonalities']):
        if s['conditional']:
            future[f"cond_{s['name']}"] = _condition(future['ds'], i).to_numpy()
    if cap is not None:
        future['cap'] = cap
    return model, df, future


def _changepoints(spec: dict[str, Any], df: pd.DataFrame) -> list[pd.Timestamp] | None:
    if spec['changepoints']['kind'] != 'explicit':
        return None
    hist = df['ds'][df['y'].notna()].reset_index(drop=True)
    idx = sorted({min(max(int(f * len(hist)), 1), len(hist) - 2)
                  for f in spec['changepoints']['fracs']})
    return [hist[i] for i in idx]


def _holidays(spec: dict[str, Any], ds_all: pd.Series, rng: np.random.Generator) -> pd.DataFrame | None:
    if not spec['holidays']:
        return None
    rows = []
    for h in spec['holidays']:
        for i in sorted(rng.choice(len(ds_all), size=min(h['n_dates'], len(ds_all)), replace=False)):
            rows.append({'holiday': h['name'], 'ds': ds_all[i], 'lower_window': h['lower'],
                         'upper_window': h['upper'], 'prior_scale': h['prior_scale']})
    hdf = pd.DataFrame(rows)
    if hdf['prior_scale'].isna().all():
        hdf = hdf.drop(columns='prior_scale')
    return hdf


def coverage(specs: list[dict[str, Any]]) -> Counter:
    """How often each option value occurs across specs (for checking the generator's reach)."""
    c: Counter = Counter()
    for s in specs:
        for key in ('growth', 'seasonality_mode', 'holidays_mode', 'scaling', 'country_holidays',
                    'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality'):
            v = s[key]
            c[(key, 'int' if isinstance(v, int) and not isinstance(v, bool) else v)] += 1
        c[('changepoints', s['changepoints']['kind'])] += 1
        c[('n_holidays', len(s['holidays']))] += 1
        c[('holiday_prior_scale', any(h['prior_scale'] is not None for h in s['holidays']))] += 1
        c[('n_regressors', len(s['regressors']))] += 1
        for r in s['regressors']:
            c[('regressor', r['kind'], f"standardize={r['standardize']}", f"mode={r['mode']}")] += 1
            c[('regressor_predictor', r['predictor'])] += 1
        c[('n_seasonalities', len(s['seasonalities']))] += 1
        c[('conditional_seasonality', any(x['conditional'] for x in s['seasonalities']))] += 1
        d = s['data']
        c[('freq', s.get('freq', 'D'))] += 1
        c[('length', 'short' if d['n_days'] < 120 else 'mid' if d['n_days'] < 800 else 'long')] += 1
        c[('scale', 'tiny' if d['scale'] < 1e-3 else 'huge' if d['scale'] > 1e6 else 'normal')] += 1
        for flag in ('negative', 'constant'):
            c[(flag, d[flag])] += 1
        c[('nans', d['nan_frac'] > 0)] += 1
        c[('outliers', d['n_outliers'] > 0)] += 1
        c[('uncertainty_samples', s['uncertainty_samples'] > 0)] += 1
        c[('include_history', s['future']['include_history'])] += 1
    return c


def case_id(seed: int, freq: str = 'D') -> str:
    """A case's name, e.g. 'H-5' (frequency, seed)."""
    return f'{freq}-{seed}'


def spec_for(case: str) -> dict[str, Any]:
    """The spec for a case id from `case_id`."""
    freq, seed = case.split('-')
    return make_spec(int(seed), freq)


def select_cases(text: str, seed0: int = 0) -> list[str]:
    """Case ids from a selection like 'D:24,H:3' (the first n seeds per frequency from seed0)."""
    out = []
    for part in text.split(','):
        freq, n = part.split(':')
        out += [case_id(s, freq) for s in range(seed0, seed0 + int(n))]
    return out


def spec_json(seed: int, freq: str = 'D') -> str:
    """The spec as a stable JSON string (for logs and repros)."""
    return json.dumps(make_spec(seed, freq), sort_keys=True)
