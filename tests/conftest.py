import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synthetic import make_series  # noqa: E402

# The parallel-prophet workspace, when this repo is cloned inside it (as prophet-batch/): optional
# cross-checks against the prophet-dev fork, the M5 data and the Track B prototype. Skipped otherwise.
WORKSPACE = Path(__file__).resolve().parents[2]

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# Bit-for-bit fingerprints hold on the machine that made them: Stan's fits
# differ in the last bits across platforms and BLAS/libm builds. CI runners (CI=true) skip those tests.
machine_baseline = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="bit-for-bit baseline from the development machine; not comparable on CI runners")
