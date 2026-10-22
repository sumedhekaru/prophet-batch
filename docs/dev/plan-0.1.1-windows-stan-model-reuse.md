# Plan: 0.1.1, reuse the loaded Stan model (Windows speed fix)

Status: approved (2026-10-21), to be implemented on branch `fix/windows-model-reuse`. Work one step at a time; tick
each step and add its result (numbers, commit) before the next.

## Problem and root cause (measured)

On a Windows machine (Intel i7-1260P, 16 logical CPUs, Python 3.12.10, prophet 1.5.0, cmdstanpy 1.3.0), `fit_many`
with `n_jobs=1` took **24.8 s for 200 M5 items** (default model), against 1.2 s on an Apple M4. Profile:

| where the time goes | time |
|---|---:|
| `Prophet.__init__` → `_load_stan_backend` → `CmdStanModel.__init__` → `do_command` (201 calls, ≈ 107 ms each) | 21.5 s (87%) |
| everything else (`_preprocess_shared` 1.6 s, `solve_map` 0.7 s, `to_datetime` 0.7 s, …) | ≈ 3.3 s |

- **cmdstanpy 1.3.0** (`cmdstanpy/model.py`, in `CmdStanModel.__init__`): on Windows, *every* `CmdStanModel(...)` runs
  `where.exe tbb.dll` as a separate process (≈ 111 ms on that machine; starting any process costs ≈ 30 ms there).
- **Prophet 1.5.0** (`prophet/models.py`): every `Prophet()` creates a `CmdStanPyBackend`, whose `__init__` calls
  `load_model()`, which builds a new `cmdstanpy.CmdStanModel(exe_file=...)`: always the same compiled model.
- **prophet-batch** creates one `Prophet()` per series (and `cross_validation_many` one per series × cutoff) and does
  only milliseconds of its own work per series, so this ≈ 100 ms dominates. Stock Prophet pays it too, behind ≈ 1 s
  of Stan per series.
- Measured `Prophet()` cost elsewhere: macOS 0.04 ms, Linux 0.03 ms. With cmdstanpy's Windows check bypassed (a
  test-only patch) `Prophet()` took 0.4 ms on the Windows machine: the extra ≈ 50× is entirely this check.
- Not the cause: BLAS threading (OPENBLAS/OMP threads = 1 changed little).
- Likely the same cause: `events_snap` (with `make_model`) getting slower with more workers (many processes launching
  `where.exe` at once); confirm with the measurements below.

## Decision (user, 2026-10-21)

- **A (this plan):** work around it in prophet-batch: while prophet-batch creates models, reuse **one loaded
  `CmdStanModel` per process** instead of letting every `Prophet()` reload it. Release as 0.1.1.
- **B (separate, not in this branch):** report upstream as drafts, held until the user decides to post: cmdstanpy
  ("run the tbb.dll PATH check once per process") and Prophet ("load the compiled model once per process").

## Design (A)

A private context manager in `src/prophet_batch/batch.py`, e.g. `_reuse_stan_model()`:

- While active, `prophet.models.CmdStanPyBackend.load_model` is replaced by a wrapper that returns a **per-process
  cached** model: the first call loads it with the original method; later calls return the same object.
- **Only the compiled-model handle is shared.** Every `Prophet` still gets its own `CmdStanPyBackend` instance (its
  own `stan_fit`, `newton_fallback`), so no fit state is shared. `CmdStanModel` runs (optimize/sample) write to new
  output directories each time, so sharing the handle between sequential fits is safe.
- **Scope:** active only inside `fit_many` (the in-process path and each worker's `_fit_chunk`) and
  `cross_validation_many` (which builds models via `prophet_copy`); restored on exit, also on exceptions. Nesting is
  allowed (a depth counter; only the outermost exit restores). Guard with a lock, since the patch is class-level.
- **Defensive:** if `prophet.models.CmdStanPyBackend` or its `load_model` is missing or has a different shape (future
  Prophet), do nothing. A subclass that overrides `load_model` is not affected (only the exact class attribute is
  patched).
- **Off switch:** a module constant (e.g. `_REUSE_STAN_MODEL = True`) so tests can compare with and without.
- Keep the change minimal; match the surrounding style (typed, single quotes, docstrings as in the file).

## Steps

- [x] 1. Baseline on the Windows machine, before any change (record numbers in this file):
      `fit_profile.py --model default --n 200`, `fit_profile.py --model events_snap --n 200`,
      `fit_scaling.py --model default --jobs 1,4,16 --n 400`, `fit_scaling.py --model events_snap --jobs 1,4,16 --n 400`
      (all from prophet-batch-examples), and the time of `[Prophet() for _ in range(100)]`.

      Result (2026-10-21, at d16496a, i7-1260P, Python 3.12.10, prophet 1.5.0, cmdstanpy 1.3.0, OpenBLAS):
      - `[Prophet() for _ in range(100)]`: **11.32 s** (113 ms each).
      - `fit_profile`, n_jobs=1, 200 items: default **31.7 s**, events_snap **32.4 s**. Of that,
        `cmdstanpy ... do_command` (201 / 200 calls) is 27.4 s (default) and 25.7 s (events_snap); `readline`
        (waiting for `where.exe`) 25.7 s / 24.0 s own time; `_winapi.CreateProcess` 1.5 s / 1.4 s.
      - `fit_scaling`, 400 items, fit_many seconds (BLAS threads default / 1):

        | model | n_jobs 1 | n_jobs 4 | n_jobs 16 |
        |---|---:|---:|---:|
        | default | 58.4 / 58.4 | 29.4 / 30.2 | 30.6 / 27.1 |
        | events_snap | 63.0 / 60.6 | 78.9 / 83.0 | 77.5 / 77.0 |

      - Same machine, earlier the same day (b00e7c9..5ee32a0): default 200 items 24.8 s; scaling default 83.3 / 38.0 /
        33.5 s, events_snap 88.9 / 113.0 / 105.7 s. Absolute times vary by ~30% between runs on this machine; the
        shape (events_snap slower with more workers, Prophet() ≈ 100+ ms) is stable.
- [x] 2. Implement the context manager and use it in `fit_many` / `_fit_chunk` / `cross_validation_many`.

      Result: `_reuse_stan_model()` in `batch.py` (after `_resolve_n_jobs`), applied as a decorator to `fit_many` and
      `cross_validation_many` (a `contextmanager` makes a fresh context per call). That covers `_fit_chunk` and the
      `cross_validation_many` workers (`_cv_group`), which both go through `fit_many`, and nested calls (depth counter).
      The cache is keyed by the original `load_model` function; the patch is skipped unless
      `getattr_static(CmdStanPyBackend, 'load_model')` is a plain function. Smoke check on Windows:
      `[Prophet() for _ in range(100)]` inside the scope **0.198 s** (vs 11.32 s); one shared `CmdStanModel`, separate
      backends; original restored after nested exit and after an exception.
- [x] 3. Tests (new, in `tests/`; they run on every OS, no Windows-only assumptions):
      - fitted parameters identical with and without reuse (`_REUSE_STAN_MODEL` False vs True), incl. a mixed batch
        (additive, multiplicative, logistic → Stan fallback, constant series);
      - Stan-fallback series give exactly `Prophet.fit`'s result inside the reuse scope;
      - `load_model` is called at most once per process during one `fit_many` of many series (count calls);
      - every model has its own backend object (no shared `stan_fit`);
      - after `fit_many` / `cross_validation_many` return (and after one raises), `CmdStanPyBackend.load_model` is the
        original function again;
      - `cross_validation_many` output unchanged with and without reuse.
      Full suite green on Windows (`pytest -q`, both with and without `-p no:faulthandler` noted).

      Result: `tests/test_stan_model_reuse.py`, 6 tests, one per bullet above (the mixed batch has additive,
      multiplicative, logistic and constant series; the CV test has additive, constant and logistic). The load-count
      test discriminates: on the same 20 series, `load_model` runs 21 times with `_REUSE_STAN_MODEL = False` and once
      with True. Full suite on Windows (Python 3.12.10, prophet 1.5.0):
      - `pytest -q`: **257 passed, 4 skipped** in 1028.99 s; 53 "Windows fatal exception: access violation" dumps
        (faulthandler, at `_winapi.CreateProcess` while worker processes start).
      - `pytest -q -p no:faulthandler`: **257 passed, 4 skipped** in 848.79 s; no dumps. Same results either way, so
        the dumps are most likely faulthandler reporting handled exceptions during process start, not crashes.
- [x] 4. Measure after the change on Windows (same commands as step 1) and one full
      `run.py --model default` on all 1,437 items; expected: `Prophet()` construction no longer dominates; speed-ups vs
      the stock pool similar to other platforms (macOS 10.2×, Linux container 8.6×, Cloud Run 10.6×).

      Result (2026-10-21, at 8dea93b, same machine and versions as step 1):
      - `[Prophet() for _ in range(100)]` (outside fit_many, unchanged): 15.22 s; inside `_reuse_stan_model()`:
        **0.151 s**.
      - `fit_profile`, n_jobs=1, 200 items: default **31.7 → 3.9 s** (8.1×), events_snap **32.4 → 7.5 s** (4.3×).
        `do_command` no longer appears; `readline` 0.15 s / 0.10 s (was 25.7 / 24.0 s). Default's phases now match the
        macOS profile's shape: `_preprocess_shared` 49%, `pandas to_datetime` 21%, `solve_map` 21%.
      - `fit_scaling`, 400 items, fit_many seconds (BLAS threads default / 1), before → after:

        | model | n_jobs 1 | n_jobs 4 | n_jobs 16 |
        |---|---:|---:|---:|
        | default | 58.4 → **6.1** / 58.4 → 6.1 | 29.4 → **5.8** / 30.2 → 6.1 | 30.6 → **11.0** / 27.1 → 11.0 |
        | events_snap | 63.0 → **10.6** / 60.6 → 12.5 | 78.9 → **9.2** / 83.0 → 9.3 | 77.5 → **12.8** / 77.0 → 13.2 |

        events_snap no longer gets much slower with more workers. At 400 items, 16 workers are slower than 4 for both
        models: the fit is now a few seconds, so starting 16 spawned processes (each importing prophet) costs more
        than it saves. Not measured further.
      - Full `run.py --model default --env windows`, 1,437 items (`windows-default-1437-20261021-224953`), before
        (b00e7c9, 2026-10-21 19:02) → after:

        | method | workers | fit s | total s | vs stock pool |
        |---|---:|---:|---:|---:|
        | stock_pool | 16 | 293.8 → 251.1 | 391.5 → 349.6 | 1.0× |
        | batch_analytic | 1 | 319.7 → **27.1** | 358.4 → 53.9 | 1.1× → **6.5×** |
        | batch_sampled | 16 | 104.3 → **15.2** | 196.1 → 87.6 | 2.0× → 4.0× |
        | batch_analytic | 16 | 104.3 → **15.2** | 125.8 → **30.3** | 3.1× → **11.5×** |

        Accuracy table identical to the run before the change (WAPE 0.3438, MASE mean 1.4261, Δ MASE −0.0021 ±
        0.0023, coverage 0.719 / 0.720); all checks pass (bit-identical across n_jobs, lp ≥ Stan's on 100% of series,
        0 fallbacks). 11.5× is in line with macOS 10.2×, Linux container 8.6×, Cloud Run 10.6×.
- [ ] 5. Push the branch (not `main`, no tag). Report: root cause recap, the change, before/after numbers, test
      summary. Then review from macOS: diff review, both test suites (`run_tests.sh`), CI on Linux/macOS/Windows,
      macOS numbers not worse.

      Windows side (2026-10-21): report sent; the first push was refused (credentials without write access), so the
      branch is pushed once the maintainer authenticates. Open: the push, the macOS review, `run_tests.sh`, CI, macOS
      numbers (maintainer).
- [ ] 6. Release prep (after review): version 0.1.1 in `pyproject.toml` and `__init__.py`; README: replace the
      "Windows speed (known issue in 0.1.0)" note with a short note that 0.1.1 fixed it, with the measured numbers.
      The release itself (tag `v0.1.1` → TestPyPI → maintainer approval → PyPI) is done by the maintainer.

## Rules for this work

- Commits: one-line messages, author Sumedhe Karunarathne <sumedhekaru@gmail.com>; no other trailers.
- Measure before changing; report numbers, not impressions; say plainly if something could not be verified.
- Do not merge to `main`, do not tag, do not publish.
- If the change turns out to need more than this design (e.g. Prophet internals differ), stop and report.
