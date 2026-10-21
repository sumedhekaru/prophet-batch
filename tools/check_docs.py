"""Check the package docs: run every Python snippet, check that every referenced test exists, and that every
public function and class has a docstring.

Each Markdown file's ```python blocks run in order in one fresh namespace (so later snippets can use earlier
ones). Test references look like `tests/test_x.py::test_name` or `tests/test_x.py::TestClass`, relative to the
repository root; each file must exist and contain the test.

    uv run --isolated --no-project --with prophet==1.5.0 --with-editable . python tools/check_docs.py
"""

import logging
import re
import sys
import warnings
from pathlib import Path

PKG = Path(__file__).parents[1]
REPO = PKG
DOCS = [PKG / 'README.md', *sorted((PKG / 'docs').glob('*.md'))]
SNIPPET = re.compile(r'```python\n(.*?)```', re.S)
TEST_REF = re.compile(r'`(tests/[\w/]+\.py)(?:::(\w+))?`')
# A reference names a test function or a test class: `tests/test_x.py::test_y` or `...::TestY`.


def missing_docstrings():
    """Public names (in `__all__`, or module-level without a leading underscore) without a docstring."""
    import inspect

    import prophet_batch
    from prophet_batch import _predict, batch

    out = []
    for mod in (prophet_batch, batch, _predict):
        names = getattr(mod, '__all__', None) or [n for n in vars(mod) if not n.startswith('_')]
        for name in names:
            obj = getattr(mod, name)
            if (inspect.isfunction(obj) or inspect.isclass(obj)) and obj.__module__.startswith('prophet_batch'):
                if not (obj.__doc__ or '').strip():
                    out.append(f'{obj.__module__}.{name} has no docstring')
    return out


def main():
    logging.getLogger('cmdstanpy').disabled = True
    logging.getLogger('prophet').setLevel(logging.WARNING)
    warnings.simplefilter('ignore')
    problems = []
    for doc in DOCS:
        text = doc.read_text(encoding='utf-8')
        namespace = {'__name__': '__docs__'}
        for i, code in enumerate(SNIPPET.findall(text), 1):
            try:
                exec(compile(code, f'{doc.name}[{i}]', 'exec'), namespace)
            except Exception as e:  # noqa: BLE001
                problems.append(f'{doc.name} snippet {i}: {type(e).__name__}: {e}')
        for path, name in TEST_REF.findall(text):
            f = REPO / path
            if not f.exists():
                problems.append(f'{doc.name}: {path} does not exist')
            elif name:
                source = f.read_text(encoding='utf-8')
                if f'def {name}(' not in source and f'class {name}' not in source:
                    problems.append(f'{doc.name}: {path} has no {name}')
        print(f'{doc.relative_to(PKG)}: {len(SNIPPET.findall(text))} snippets, '
              f'{len(TEST_REF.findall(text))} test references')
    problems += missing_docstrings()
    for p in problems:
        print('FAIL', p)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
