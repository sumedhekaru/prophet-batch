"""The batch API under test: prophet_batch.

Tests import batch functions from here (`from api import fit_many`); any name of the module resolves
(PEP 562), so worker processes that re-import a test module get the same module. Seeded predict calls
go through `predict` / `predictive_samples`: prophet_batch's seeded path, which delegates to the fork's
own predict when its hooks exist. To run the suite on stock PyPI prophet, see README (Commands).
"""

import importlib

MODULE_NAME = "prophet_batch.batch"
MODULE = importlib.import_module(MODULE_NAME)
BATCH_LOGGER = "prophet_batch"


def predict(model, df=None, **kwargs):
    import prophet_batch

    return prophet_batch.predict(model, df, **kwargs)


def predictive_samples(model, df, **kwargs):
    from prophet_batch._seeded import seeded

    return seeded(model).predictive_samples(df, **kwargs)


def __getattr__(name):
    return getattr(MODULE, name)
