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

# Bit-for-bit fingerprints hold only on the machine that made them: Stan's fits and libm differ in the last bits
# across platforms and BLAS/libm builds. Opt-in (run_tests.sh sets it on the development machine); skipped
# everywhere else: CI, users running the suite from the sdist, other platforms.
machine_baseline = pytest.mark.skipif(
    os.environ.get("PROPHET_BATCH_MACHINE_BASELINES") != "1",
    reason="bit-for-bit baseline from the development machine; set PROPHET_BATCH_MACHINE_BASELINES=1 there")
