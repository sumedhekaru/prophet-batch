#!/bin/sh
# Package tests in both environments:
#   1. the dev env (prophet = the prophet-dev fork, with the native hooks): the package path must equal the fork's;
#   2. an isolated env with PyPI prophet==1.5.0 (no hooks): the package's own seeded path.
#   sh packages/prophet-batch/run_tests.sh [pytest args]
set -eu
PKG="$(cd "$(dirname "$0")" && pwd)"
cd "$PKG"
echo "== dev env (prophet-dev fork)"
PYTHONPATH="$PKG/src" uv run --project "$PKG/../.." pytest -q -p no:cacheprovider "$@"
echo "== PyPI prophet==1.5.0, Python ${PY:-3.12}"
uv run --isolated --no-project --python "${PY:-3.12}" --with prophet==1.5.0 --with pytest --with-editable "$PKG" \
    pytest -q -p no:cacheprovider "$@"
