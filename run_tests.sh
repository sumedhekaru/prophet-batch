#!/bin/sh
# Run the tests on stock PyPI prophet (what CI does), and, when this repo is cloned inside the parallel-prophet
# workspace next to prophet-dev, also in the workspace env (prophet = the fork, with the native predict hooks).
#   sh run_tests.sh [pytest args]          PY=3.10 sh run_tests.sh   (Python for the PyPI run; default 3.12)
set -eu
PKG="$(cd "$(dirname "$0")" && pwd)"
cd "$PKG"
# This is the development machine: also run the bit-for-bit baseline tests (tests/conftest.py).
export PROPHET_BATCH_MACHINE_BASELINES="${PROPHET_BATCH_MACHINE_BASELINES:-1}"
WORKSPACE="$(dirname "$PKG")"
if [ -d "$WORKSPACE/prophet-dev" ] && [ -f "$WORKSPACE/pyproject.toml" ]; then
    echo "== workspace env (prophet-dev fork)"
    uv run --project "$WORKSPACE" pytest -q -p no:cacheprovider "$@"
fi
echo "== PyPI prophet==1.5.0, Python ${PY:-3.12}"
uv run --isolated --no-project --python "${PY:-3.12}" --with prophet==1.5.0 --with pytest --with pytest-timeout \
    --with-editable "$PKG" pytest -q -p no:cacheprovider "$@"
