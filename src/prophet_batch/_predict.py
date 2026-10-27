# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Seeded predict for any fitted Prophet model (stock or with the upstream hooks)."""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any, Literal

from prophet_batch._holidays import fast_holidays
from prophet_batch._seeded import seeded

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from prophet import Prophet


def has_native_hooks(model: Prophet) -> bool:
    """Whether the model's own predict accepts rng= and history_uncertainty=."""
    params = inspect.signature(type(model).predict).parameters
    return 'rng' in params and 'history_uncertainty' in params


@fast_holidays()
def predict(
    model: Prophet,
    df: pd.DataFrame | None = None,
    vectorized: bool = True,
    history_uncertainty: bool | Literal['analytic'] = True,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Prophet.predict with a per-call random generator and history interval modes.

    Parameters
    ----------
    model: A fitted Prophet model.
    df: As in Prophet.predict (None predicts on the history).
    vectorized: As in Prophet.predict.
    history_uncertainty: True (the default) samples intervals for every row, as stock
        Prophet does. False leaves history rows' yhat/trend intervals NaN and samples
        only future rows (exactly the intervals of predicting the future rows alone).
        'analytic' (MAP fits only) samples future rows as False does and gives history
        rows the exact limit of the sampled intervals, yhat +/- z * sigma_obs * y_scale.
    rng: Optional numpy Generator for all random draws. None uses NumPy's global RNG;
        with None and history_uncertainty=True the result is bit-identical to
        model.predict(df, vectorized).

    A Prophet subclass that overrides predict (without the rng and history_uncertainty
    arguments) gets Prophet's own predict logic here, not its override; a warning says so.

    Returns
    -------
    A pd.DataFrame with the forecast components, as Prophet.predict returns.
    """
    if has_native_hooks(model):
        native: Any = model  # stock Prophet's type has no rng / history_uncertainty
        return native.predict(df, vectorized, history_uncertainty=history_uncertainty, rng=rng)
    from prophet import Prophet

    if type(model).predict is not Prophet.predict:
        warnings.warn(
            f'{type(model).__name__} overrides predict; prophet_batch uses Prophet\'s predict '
            'logic (with seeded sampling) instead of the override.', stacklevel=2)
    target: Any = seeded(model)
    return target.predict(df, vectorized, history_uncertainty=history_uncertainty, rng=rng)
