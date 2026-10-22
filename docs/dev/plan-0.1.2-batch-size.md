# 0.1.2: memory-aware batch size in fit_many (design)

## Why (measured, prophet-batch 0.1.1, Apple M4)

- `fit_many(n_jobs=W)` splits the series into `min(n, 4·W)` tasks, and a task prepares all its series (features,
  design matrix) before solving. A worker therefore holds n / (4W) series: memory per worker grows with the dataset
  (30,490 series → 762 per task; 300,000 → ≈ 7,500). With `n_jobs=1` all series are prepared at once.
- Batch size does not buy speed (144 runs): one process is as fast or slightly faster at 25–200 series per batch
  than with everything in one batch (M5 item 11.8 vs 12.5 s; M4 Daily 24.9 vs 27.2 s; multiplicative flat); with
  workers, large batches only lose by idling workers. Worker memory ≈ 0.7–0.9 MB per series (1,913–2,371 rows).
- The parallel path also copies the whole input up front (every task's rows sliced before the pool starts): +1.45 GB
  on 30,490 series (input 1.40 GB).
- Results are bit-identical for every batch size (a series' fit depends only on its own data).

## Design (decided 2026-10-22)

**Parameter.** `fit_many(..., batch_size: int | None = None)` and `cross_validation_many(..., batch_size=None)` (its
refits go through fit_many). `batch_size` is the maximum number of series a worker prepares and solves at once (one
task); with `n_jobs=1`, the series are fit in consecutive batches of that size in this process. An int sets exactly
that cap (must be ≥ 1). `None` (default) chooses automatically:

- **Count cap 100** (the flat region of the measurements), lowered to `ceil(n / (4·W))` when that is smaller, so a
  small job still has at least 4·W tasks (today's load balance).
- **Memory cap**: tasks are built from consecutive series, adding series while the task's estimated bytes stay within
  `0.5 × available memory / W` (user: target 50% when speed is flat). Estimated bytes of a series =
  its rows × bytes per row, where bytes per row comes from preparing the first series once (design matrix + features
  + a few vectors, in float64). A task always holds at least one series. Long series (e.g. hourly data with 100,000
  rows) get small tasks; short ones are limited by the count cap.
- **Available memory** (read at each call; no new dependency): Linux `MemAvailable` (/proc/meminfo), capped by a
  cgroup limit (v2 `memory.max − memory.current`, v1 `memory.limit_in_bytes − memory.usage_in_bytes`); macOS `vm_stat`
  (free + inactive + speculative + purgeable pages); Windows `GlobalMemoryStatusEx` (ullAvailPhys) via ctypes. If it
  can't be read, only the count cap applies.

**Parent memory.** The parallel path submits tasks a few at a time (at most 2·W in flight) and slices each task's rows
when it is submitted, so the input is never copied whole. Results are gathered in task order (unchanged output order).

**Logging.** One info line: the series count, batch size (and whether count- or memory-limited), tasks, workers.

**Not changed** (measured, or API decisions for later): the returned models' memory (≈ 0.12 MB each at 1,913 rows,
mostly the history frame stock Prophet keeps); `predict_many`'s chunking (track I's `forecast_many` covers the
large-scale path); dropping Z early (shared design matrices already make it small for equal-dates series; no measured
need).

## Guarantees and tests

- Fits are bit-identical for any `batch_size` (1, 7, auto, n) and any `n_jobs`: new property test.
- Output order and skipped series unchanged; `on_error` behaves per series as before.
- `batch_size` validation: int ≥ 1 or None, else ValueError.
- The memory reader on each OS (CI: Linux, macOS, Windows), incl. a fake cgroup tree.
- Automatic size: count cap, the 4·W rule, the memory cap with a small fake budget (tasks shrink, results unchanged).
- Parent memory: no full copy (peak parent memory on a large synthetic input stays below input + a few tasks).

## Steps

- [x] 1. Memory reader + task builder + in-process batching + bounded submission; tests above.
      Done: 27 new tests (tests/test_batch_size.py); suites 288 + 288; pyrefly 0.
- [x] 2. Measure: M5 item × store at n_jobs=10 (parent and worker peaks, time vs 0.1.1, median of 3).
      Done (same isolated env for both, prophet 1.5.0 from PyPI): fit 31.3 → **23.2 s** (−26%) on 30,490 series;
      predict unchanged. Without memory pressure (15,000 series, no swapping): fit 13.7 → 11.6 s, parent peak during
      fit 6.80 → **4.42 GB**, whole-run parent peak 6.80 → 6.15 GB, worker peak 0.54 → 0.48 GB. At 30,490 series the
      16 GB Mac swaps, which distorts footprint comparisons. Known: the parent keeps ≈ 0.6 GB more right after fit
      (not at peak; not released by malloc pressure relief; likely Python's small-object arenas, as results now
      arrive while tasks are still being sent): for the code review.
- [x] 3. README (Limitations: memory; the parameter) and guarantees.md.
- Release: 0.1.2 together with `forecast_many` / `Pool`, after all experiments (maintainer).
