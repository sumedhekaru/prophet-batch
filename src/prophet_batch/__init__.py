# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fit and forecast many Prophet models at once, on top of stock Prophet."""

from __future__ import annotations

__version__ = '0.1.1'

import logging  # noqa: E402
import multiprocessing  # noqa: E402


def _drop_repeated_plotly_error(record: logging.LogRecord) -> bool:
    return not str(record.msg).startswith('Importing plotly failed')


# Prophet logs a missing plotly when it is imported. Our worker processes import it
# again, so without this every worker repeats the parent's message. A spawned child
# imports us before parent_process() is set (re-running the main module, unpickling
# its work), while multiprocessing marks it as inheriting.
if (multiprocessing.parent_process() is not None
        or getattr(multiprocessing.current_process(), '_inheriting', False)):
    logging.getLogger('prophet.plot').addFilter(_drop_repeated_plotly_error)

from prophet_batch.batch import (  # noqa: E402
    cross_validation_many, fit_many, predict_many, series_rng,
)
from prophet_batch._predict import predict  # noqa: E402

__all__ = ['cross_validation_many', 'fit_many', 'predict', 'predict_many', 'series_rng']
