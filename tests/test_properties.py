"""D1.1.2: fit_many invariants on seeded random cases (see random_cases.py).

Tolerances (track D, D1.1/D1.4.1): KKT < 1e-8 additive (or 10x the float64 floor), block KKT <= 1e-3
multiplicative (fit_many's own acceptance rule; converged fits measure <= 1e-4);
lp >= Stan's - 1e-6 additive, - 1e-4 multiplicative; everything else bit-identical.
Cases: PP_CASES daily cases (default 24) plus PP_FREQS for other frequencies (default
'H:3,W:6,MS:6'), from seed PP_SEED0. Scale up with e.g. PP_CASES=300 or PP_CASES=0
PP_FREQS=H:100,W:100,MS:100. A failure lists case ids like 'H-5'; `spec_for(id)` reproduces one.
"""

import os
import warnings

import numpy as np
import pytest
from prophet import Prophet
from api import (
    _MultiplicativeBatch, design_matrix, fit_many, has_multiplicative_terms, is_constant,
    kkt_violation, series_rng, solve_map, stack_problems, unsupported_reason,
)

from random_cases import build, select_cases, spec_for
from test_multiplicative import _block_kkt
from api import predict  # noqa: E402

N_CASES = int(os.environ.get("PP_CASES", "24"))
SEED0 = int(os.environ.get("PP_SEED0", "0"))
FREQS = os.environ.get("PP_FREQS", "H:3,W:6,MS:6")
SEEDS = select_cases(f"D:{N_CASES}" + (f",{FREQS}" if FREQS else ""), SEED0)  # case ids

# Known failures, each tied to a D1.4 item in docs/plans/track-d-release.md: (test, seed) -> why.
# A listed case must keep failing (so the list stays current); any other failure fails the test.
KNOWN: dict = {}


def make_model(case):
    return build(spec_for(case), Prophet)[0]


def _frame(case):
    return build(spec_for(case), Prophet)[1]


def _signature(df):
    return tuple(df.columns)


@pytest.fixture(scope="module")
def fitted():
    """Per seed: the fit_many model, a plain Prophet.fit model, and what the solver saw."""
    frames = {s: _frame(s) for s in SEEDS}
    by_signature = {}
    for s, df in frames.items():
        by_signature.setdefault(_signature(df), []).append(s)
    ours, crashed = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seeds in by_signature.values():
            long = __import__("pandas").concat(
                [frames[s].assign(unique_id=s) for s in seeds], ignore_index=True)
            try:
                ours.update(fit_many(long, make_model=make_model))
            except Exception:  # one series broke the batch: find it, keep the others
                for s in seeds:
                    try:
                        ours.update(fit_many(frames[s].assign(unique_id=s), make_model=make_model))
                    except Exception as e:
                        crashed[s] = f"{type(e).__name__}: {e}"
        stan = {s: make_model(s).fit(frames[s]) for s in SEEDS}
    out = {}
    for s in SEEDS:
        if s in crashed:
            out[s] = dict(ours=None, stan=stan[s], kind="crashed", error=crashed[s], df=frames[s])
            continue
        probe = make_model(s)
        reason = unsupported_reason(probe)
        inputs = None if reason else probe.preprocess(frames[s])
        kind = ("fallback" if reason else "constant" if is_constant(probe)
                else "solver_fallback" if ours[s].stan_fit is not None
                else "multiplicative" if has_multiplicative_terms(inputs) else "additive")
        out[s] = dict(ours=ours[s], stan=stan[s], kind=kind, inputs=inputs, df=frames[s])
    return out


def _flat(model):
    return {k: np.asarray(v) for k, v in model.params.items()}


def _check(test, bad):
    """Fail on unknown failures, and on known failures that now pass."""
    failing = {b[0] if isinstance(b, tuple) else b for b in bad}
    new = [b for b in bad if (test, b[0] if isinstance(b, tuple) else b) not in KNOWN]
    fixed = [seed for (t, seed) in KNOWN if t == test and seed in SEEDS and seed not in failing]
    msg = []
    if new:
        msg.append("failing cases (reproduce with random_cases.spec_for(case)): "
                   + "; ".join(map(str, new)))
    if fixed:
        msg.append(f"known failures now pass, remove them from KNOWN: {fixed}")
    assert not msg, " | ".join(msg)


def test_fit_many_does_not_raise(fitted):
    bad = [(s, r["error"]) for s, r in fitted.items() if r["kind"] == "crashed"]
    _check("fit_many_does_not_raise", bad)


def test_solver_fallbacks_are_rare(fitted):
    """Eligible series the batched solver handed to Prophet.fit: at most 1% (one always allowed)."""
    eligible = [s for s, r in fitted.items() if r["kind"] not in ("fallback", "constant", "crashed")]
    used = [s for s in eligible if fitted[s]["kind"] == "solver_fallback"]
    allowed = max(1, int(0.01 * len(eligible)))
    assert len(used) <= allowed, f"solver fallbacks {used} of {len(eligible)} eligible"


def test_every_kind_is_covered(fitted):
    kinds = {r["kind"] for r in fitted.values()}
    if N_CASES >= 24:
        assert {"additive", "multiplicative", "fallback"} <= kinds, kinds


def test_eligible_series_skip_stan(fitted):
    uses_stan = ("fallback", "solver_fallback")
    bad = [s for s, r in fitted.items() if r["kind"] != "crashed"
           and (r["kind"] in uses_stan) == (r["ours"].stan_fit is None)]
    _check("eligible_series_skip_stan", bad)


def test_fallback_and_constant_match_prophet_fit(fitted):
    bad = []
    for s, r in fitted.items():
        if r["kind"] not in ("fallback", "solver_fallback", "constant"):
            continue
        a, b = _flat(r["ours"]), _flat(r["stan"])
        if a.keys() != b.keys() or any(not np.array_equal(a[k], b[k]) for k in a):
            bad.append((s, r["kind"]))
    _check("fallback_and_constant_match_prophet_fit", bad)


def _kkt_floor(G, C, w, sigma, tau):
    """KKT residual that float64 rounding alone can produce: eps * p * |terms| * tau / sigma²."""
    mag = max(np.abs(C).max(), np.abs(np.einsum("bpq,bq->bp", np.abs(G), np.abs(w))).max())
    return np.finfo(float).eps * C.shape[1] * mag * float(tau[0]) / float(sigma[0]) ** 2


def test_accepted_fits_have_sigma_above_floor(fitted):
    """At the sigma floor the KKT tolerance below is meaningless; such fits must go to Prophet.fit."""
    bad = [s for s, r in fitted.items() if r["kind"] in ("additive", "multiplicative")
           and not r["ours"].params["sigma_obs"][0, 0] > 1e-9]
    _check("accepted_fits_have_sigma_above_floor", bad)


def test_additive_is_exact_optimum(fitted):
    """KKT < 1e-8, or < 10x the float64 floor when tiny sigma makes 1e-8 unmeasurable."""
    bad = []
    for s, r in fitted.items():
        if r["kind"] != "additive":
            continue
        inputs = r["inputs"]
        _, G, C, yy, T, ridge, l1, tau = stack_problems(
            [(design_matrix(inputs), inputs.y, inputs.sigmas, inputs.tau)])
        if len(r["ours"].changepoints) == 0:
            # Stored params have delta folded into k (as Prophet.fit does), so check the
            # solver's own output, which fit_many stores before folding.
            w, sigma = solve_map(G, C, yy, T, ridge, l1, tau)
        else:
            p = {k: np.ravel(v) for k, v in r["ours"].params.items()}
            w = np.concatenate([p["k"], p["m"], p["delta"], p["beta"]])[None, :]
            sigma = p["sigma_obs"]
        kkt = kkt_violation(G, C, w, sigma, ridge, l1, tau)[0]
        tol = max(1e-8, 10 * _kkt_floor(G, C, w, sigma, tau))
        if not kkt < tol:
            bad.append((s, f"kkt={kkt:.2e} tol={tol:.1e}"))
    _check("additive_is_exact_optimum", bad)


def test_multiplicative_is_stationary(fitted):
    bad = []
    for s, r in fitted.items():
        if r["kind"] != "multiplicative":
            continue
        mb = _MultiplicativeBatch([r["inputs"]])
        if len(r["ours"].changepoints) == 0:
            theta, ba, bm, sigma, _, _ = mb.solve()  # see test_additive_is_exact_optimum
        else:
            part = mb.parts[0]
            p = {k: np.ravel(v) for k, v in r["ours"].params.items()}
            theta = np.concatenate([p["k"], p["m"], p["delta"]])[None, :]
            ba, bm = p["beta"][part.ia][None, :], p["beta"][part.im][None, :]
            sigma = p["sigma_obs"]
        kA, kB = _block_kkt(mb, theta, ba, bm, sigma)
        if not max(kA.max(), kB.max()) <= 1e-3:
            bad.append((s, f"kkt={kA.max():.2e},{kB.max():.2e}"))
    _check("multiplicative_is_stationary", bad)


def test_log_posterior_at_least_stan(fitted):
    bad = []
    for s, r in fitted.items():
        if r["kind"] not in ("additive", "multiplicative"):
            continue
        tol = 1e-6 if r["kind"] == "additive" else 1e-4
        ours, stan = r["ours"].params["lp__"][0, 0], r["stan"].params["lp__"][0, 0]
        if not ours >= stan - tol * max(1.0, abs(stan)):
            bad.append((s, r["kind"], f"ours={ours:.6f} stan={stan:.6f}"))
    _check("log_posterior_at_least_stan", bad)


def test_predict_works(fitted):
    bad = []
    for s, r in fitted.items():
        if r["kind"] == "crashed":
            continue
        future = build(spec_for(s), Prophet)[2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = predict(r["ours"], future, rng=series_rng(0, s))
        problems = []
        if len(fc) != len(future) or not np.isfinite(fc["yhat"]).all():
            problems.append("yhat")
        if r["ours"].uncertainty_samples:
            lo, hi = fc["yhat_lower"].to_numpy(), fc["yhat_upper"].to_numpy()
            if not (np.isfinite(lo).all() and np.isfinite(hi).all() and (lo <= hi).all()):
                problems.append("intervals")
        if problems:
            bad.append((s, problems))
    _check("predict_works", bad)


def test_analytic_history_intervals(fitted):
    """D2: 'analytic' = False on future rows; closed form on history rows."""
    from statistics import NormalDist

    cols = ["yhat_lower", "yhat_upper", "trend_lower", "trend_upper"]
    bad = []
    for s, r in fitted.items():
        if r["kind"] == "crashed":
            continue
        m = r["ours"]
        future = build(spec_for(s), Prophet)[2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = predict(m, future, history_uncertainty="analytic", rng=series_rng(0, s))
            ref = predict(m, future, history_uncertainty=False, rng=series_rng(0, s))
        if not m.uncertainty_samples:
            if not fc.equals(ref):
                bad.append((s, "no-uncertainty frame differs"))
            continue
        hist = (future["ds"] <= m.history["ds"].max()).to_numpy()
        problems = []
        if not np.array_equal(fc.loc[~hist, cols].to_numpy(), ref.loc[~hist, cols].to_numpy(), equal_nan=True):
            problems.append("future")
        hw = (NormalDist().inv_cdf((1 + m.interval_width) / 2)
              * float(np.ravel(m.params["sigma_obs"])[0]) * m.y_scale)
        yhat, trend = fc.loc[hist, "yhat"].to_numpy(), fc.loc[hist, "trend"].to_numpy()
        tol = 1e-14 * (np.abs(yhat) + hw)
        if not (np.all(np.abs(fc.loc[hist, "yhat_lower"].to_numpy() - (yhat - hw)) <= tol)
                and np.all(np.abs(fc.loc[hist, "yhat_upper"].to_numpy() - (yhat + hw)) <= tol)):
            problems.append("history yhat bounds")
        if not (np.array_equal(fc.loc[hist, "trend_lower"].to_numpy(), trend)
                and np.array_equal(fc.loc[hist, "trend_upper"].to_numpy(), trend)):
            problems.append("history trend bounds")
        if problems:
            bad.append((s, problems))
    _check("analytic_history_intervals", bad)


# D1.1.3: results don't depend on n_jobs, series order or batch composition (bit-identical).

def _groups(fitted):
    """Seeds grouped by column signature, as in the `fitted` fixture (crashed seeds left out)."""
    groups = {}
    for s, r in fitted.items():
        if r["kind"] == "crashed":
            continue
        groups.setdefault(_signature(r["df"]), []).append(s)
    return list(groups.values())


def _long(fitted, seeds):
    import pandas as pd

    return pd.concat([fitted[s]["df"].assign(unique_id=s) for s in seeds], ignore_index=True)


def _differs(a, b):
    pa, pb = _flat(a), _flat(b)
    return pa.keys() != pb.keys() or any(not np.array_equal(pa[k], pb[k]) for k in pa)


def _refit(fitted, seeds, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_many(_long(fitted, seeds), make_model=make_model, **kw)


def test_fit_independent_of_n_jobs(fitted):
    bad = []
    for seeds in _groups(fitted):
        if len(seeds) < 2:
            continue
        models = _refit(fitted, seeds, n_jobs=3)
        bad += [s for s in seeds if _differs(models[s], fitted[s]["ours"])]
    _check("fit_independent_of_n_jobs", bad)


def test_fit_independent_of_order(fitted):
    bad = []
    for seeds in _groups(fitted):
        if len(seeds) < 2:
            continue
        models = _refit(fitted, seeds[::-1])
        assert list(models) == seeds[::-1]
        bad += [s for s in seeds if _differs(models[s], fitted[s]["ours"])]
    _check("fit_independent_of_order", bad)


def test_fit_independent_of_batch_mates(fitted):
    bad = [s for s in SEEDS if fitted[s]["kind"] != "crashed"
           and _differs(_refit(fitted, [s])[s], fitted[s]["ours"])]
    _check("fit_independent_of_batch_mates", bad)


def test_predict_many_independent_of_n_jobs_and_order(fitted):
    from api import predict_many

    ok = [s for s in SEEDS if fitted[s]["kind"] != "crashed"]
    models = {s: fitted[s]["ours"] for s in ok}
    futures = {s: build(spec_for(s), Prophet)[2] for s in ok}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = predict_many(models, futures=futures, seed=7)
        par = predict_many(models, futures=futures, seed=7, n_jobs=3)
        rev = predict_many(dict(reversed(models.items())), futures=futures, seed=7)
    bad = []
    for s in ok:
        for other, label in ((par, "n_jobs=3"), (rev, "reversed")):
            if not base[s].equals(other[s]):
                bad.append((s, label))
    _check("predict_many_independent_of_n_jobs_and_order", bad)
