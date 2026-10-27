# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""A faster, bit-identical Prophet.make_holiday_features, used inside prophet-batch's calls.

Prophet looks up every holiday date and window offset one at a time (`pd.to_datetime` and
`DatetimeIndex.get_loc`: ~55 ms for 6 holidays over 22 years with ±15-day windows). This
version keeps Prophet's per-row logic (windows, prior scales, errors, column names and
order) and finds the rows of each occurrence with one sorted search per holiday row.
"""

from __future__ import annotations

import hashlib
import inspect
import textwrap
import threading
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date
from functools import lru_cache
from inspect import getattr_static
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from prophet import Prophet

# sha256 of the source of Prophet.make_holiday_features this mirrors (prophet 1.5.0). Any
# other source (another version, a patched install) keeps Prophet's own function.
_KNOWN_SOURCES = frozenset({
    '6ae13164dcb91a7af2e191aead7607090d811c42011febde82fc8c53c758f9a5',
})
# False turns the replacement off (for tests and comparisons).
_FAST_HOLIDAYS = True
_lock = threading.Lock()
_depth = 0
_original: Any = None
_EPOCH = date(1970, 1, 1)


def make_holiday_features(
    self: Prophet,
    dates: pd.Series,
    holidays: pd.DataFrame,
) -> tuple[pd.DataFrame, list[float], list[str]]:
    """Prophet.make_holiday_features (1.5.0) with vectorized date lookups; same results.

    Parameters
    ----------
    dates: pd.Series containing timestamps used for computing seasonality.
    holidays: pd.Dataframe containing holidays, as returned by
        construct_holiday_dataframe.

    Returns
    -------
    holiday_features: pd.DataFrame with a column for each holiday.
    prior_scale_list: List of prior scales for each holiday column.
    holiday_names: List of names of holidays
    """
    n = dates.shape[0]
    expanded_holidays: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(n))
    prior_scales: dict[Any, float] = {}
    # Prophet matches on the date part of each timestamp; several rows may share a date.
    row_index = pd.DatetimeIndex(dates.dt.date)
    days = row_index.to_numpy().astype('datetime64[D]').astype(np.int64)  # exact: midnights
    order = np.argsort(days, kind='stable')
    sorted_days = days[order]

    for row in holidays.itertuples():
        ds = cast(pd.Timestamp, row.ds)
        dt = ds.date()
        try:
            lw = int(getattr(row, 'lower_window', 0))
            uw = int(getattr(row, 'upper_window', 0))
        except ValueError:
            lw = 0
            uw = 0
        ps = float(getattr(row, 'prior_scale', self.holidays_prior_scale))
        if np.isnan(ps):
            ps = float(self.holidays_prior_scale)
        if row.holiday in prior_scales and prior_scales[row.holiday] != ps:
            raise ValueError(
                'Holiday {holiday!r} does not have consistent prior '
                'scale specification.'.format(holiday=row.holiday)
            )
        if ps <= 0:
            raise ValueError('Prior scale must be > 0')
        prior_scales[row.holiday] = ps

        offsets = range(lw, uw + 1)
        if dt is not pd.NaT:
            occ = (dt - _EPOCH).days + np.arange(lw, uw + 1)
            lo = np.searchsorted(sorted_days, occ, 'left')
            hi = np.searchsorted(sorted_days, occ, 'right')
        else:  # NaT (a holiday seen in fit but absent here): columns of zeros
            lo = hi = np.zeros(len(offsets), dtype=int)
        for offset, a, b in zip(offsets, lo, hi):
            key = '{}_delim_{}{}'.format(
                row.holiday,
                '+' if offset >= 0 else '-',
                abs(offset)
            )
            col = expanded_holidays[key]  # Access key to generate value
            if b > a:
                col[order[a:b]] = 1.
    holiday_features = pd.DataFrame(expanded_holidays)
    # Make sure column order is consistent
    holiday_features = holiday_features[sorted(holiday_features.columns
                                                               .tolist())]
    prior_scale_list = [
        prior_scales[h.split('_delim_')[0]]
        for h in holiday_features.columns
    ]
    holiday_names = list(prior_scales.keys())
    # Store holiday names used in fit
    if self.train_holiday_names is None:
        self.train_holiday_names = pd.Series(holiday_names)
    return holiday_features, prior_scale_list, holiday_names


@lru_cache(maxsize=8)
def _source_known(function: Any) -> bool:
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError):
        return False
    return hashlib.sha256(source.encode()).hexdigest() in _KNOWN_SOURCES


@contextmanager
def fast_holidays() -> Generator[None, None, None]:
    """While active, Prophet builds holiday features with `make_holiday_features` above.

    Patches Prophet.make_holiday_features (subclasses that override it keep theirs) and
    restores it on exit, also on errors; nested uses restore at the outermost exit. Does
    nothing unless the installed function's source is one this mirrors.
    """
    global _depth, _original
    from prophet import Prophet

    with _lock:
        if _depth == 0:
            current = getattr_static(Prophet, 'make_holiday_features', None)
            if _FAST_HOLIDAYS and inspect.isfunction(current) and _source_known(current):
                _original = current
                setattr(Prophet, 'make_holiday_features', make_holiday_features)
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0 and _original is not None:
                setattr(Prophet, 'make_holiday_features', _original)
                _original = None
