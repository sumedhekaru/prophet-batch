"""prophet_batch.batch.solve_map: the batched exact MAP solver (C1.1)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from api import kkt_violation, solve_map

# Optional cross-check against the Track B prototype in the parallel-prophet workspace.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "batched_fit"))


def synthetic_problem(B=20, T=600, S=10, K=8, seed=0):
    """Linear-trend design with hinge columns, Fourier-like features, and noisy targets."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)
    tc = np.linspace(0, 0.8, S + 1)[1:]
    hinge = (t[:, None] >= tc) * (t[:, None] - tc)
    X = np.column_stack([np.sin(2 * np.pi * (i + 1) * t * 7) for i in range(K)])
    Z = np.column_stack([t, np.ones(T), hinge, X])
    true = np.column_stack([
        rng.normal(0, 0.5, B), rng.normal(0.5, 0.1, B),
        rng.laplace(0, 0.3, (B, S)) * (rng.random((B, S)) < 0.3),
        rng.normal(0, 0.1, (B, K)),
    ])
    Y = Z @ true.T + rng.normal(0, 0.05, (T, B))
    p = Z.shape[1]
    ridge = np.tile(np.concatenate([[1 / 25, 1 / 25], np.zeros(S), np.full(K, 1 / 100)]), (B, 1))
    l1 = np.tile(np.concatenate([[False, False], np.ones(S, bool), np.zeros(K, bool)]), (B, 1))
    return Z, Y, ridge, l1, np.full(B, 0.05), p


def test_shared_gram_reaches_optimum():
    Z, Y, ridge, l1, tau, _ = synthetic_problem()
    G = Z.T @ Z
    C = (Z.T @ Y).T
    w, sigma = solve_map(G, C, (Y * Y).sum(0), np.full(Y.shape[1], len(Z)), ridge, l1, tau)
    assert kkt_violation(G, C, w, sigma, ridge, l1, tau).max() < 1e-9


def test_per_series_gram_matches_shared():
    Z, Y, ridge, l1, tau, p = synthetic_problem(B=8)
    G = Z.T @ Z
    C = (Z.T @ Y).T
    args = ((Y * Y).sum(0), np.full(Y.shape[1], len(Z)), ridge, l1, tau)
    w1, s1 = solve_map(G, C, *args)
    w2, s2 = solve_map(np.broadcast_to(G, (Y.shape[1], p, p)).copy(), C, *args)
    np.testing.assert_allclose(w1, w2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s1, s2, rtol=1e-12)


def test_padding_coordinates_solve_to_zero():
    Z, Y, ridge, l1, tau, p = synthetic_problem(B=6)
    B = Y.shape[1]
    G = np.zeros((B, p + 3, p + 3))
    G[:, :p, :p] = Z.T @ Z
    G[:, range(p, p + 3), range(p, p + 3)] = 1.0
    C = np.zeros((B, p + 3))
    C[:, :p] = (Z.T @ Y).T
    ridge_pad = np.concatenate([ridge, np.ones((B, 3))], axis=1)
    l1_pad = np.concatenate([l1, np.zeros((B, 3), bool)], axis=1)
    w, sigma = solve_map(G, C, (Y * Y).sum(0), np.full(B, len(Z)), ridge_pad, l1_pad, tau)
    assert np.all(w[:, p:] == 0)
    w_ref, s_ref = solve_map(Z.T @ Z, (Z.T @ Y).T, (Y * Y).sum(0), np.full(B, len(Z)), ridge, l1, tau)
    np.testing.assert_allclose(w[:, :p], w_ref, rtol=0, atol=1e-12)


def test_per_series_penalties():
    Z, Y, ridge, l1, tau, _ = synthetic_problem(B=10)
    tau = np.linspace(0.01, 0.5, len(tau))
    ridge = ridge * np.linspace(0.5, 2, len(tau))[:, None]
    G, C = Z.T @ Z, (Z.T @ Y).T
    w, sigma = solve_map(G, C, (Y * Y).sum(0), np.full(Y.shape[1], len(Z)), ridge, l1, tau)
    assert kkt_violation(G, C, w, sigma, ridge, l1, tau).max() < 1e-9
    # Stronger L1 (smaller tau) gives fewer nonzero deltas.
    nnz = (w[:, l1[0]] != 0).sum(axis=1)
    assert nnz[0] <= nnz[-1]


def test_agrees_with_prototype():
    proto = pytest.importorskip("lasso_fit")
    Z, Y, ridge, l1, tau, _ = synthetic_problem(B=12, seed=3)
    w, sigma = solve_map(Z.T @ Z, (Z.T @ Y).T, (Y * Y).sum(0), np.full(Y.shape[1], len(Z)), ridge, l1, tau)
    w_p, s_p, _ = proto.fit_batch_active(Z, Y, ridge[0], l1[0], tau[0])
    np.testing.assert_allclose(w, w_p, rtol=0, atol=1e-10)
    np.testing.assert_allclose(sigma, s_p, rtol=1e-10)


def test_warm_start_same_result_fewer_steps():
    Z, Y, ridge, l1, tau, _ = synthetic_problem(B=15, seed=5)
    G, C = Z.T @ Z, (Z.T @ Y).T
    args = ((Y * Y).sum(0), np.full(Y.shape[1], len(Z)), ridge, l1, tau)
    cold_stats, warm_stats = {}, {}
    w, sigma = solve_map(G, C, *args, stats=cold_stats)
    # Warm start from a perturbed solution (as between alternating blocks).
    rng = np.random.default_rng(0)
    w0 = w + 1e-3 * rng.normal(size=w.shape) * (w != 0)
    w2, sigma2 = solve_map(G, C, *args, w0=w0, sigma0=sigma * 1.01, stats=warm_stats)
    assert kkt_violation(G, C, w2, sigma2, ridge, l1, tau).max() < 1e-9
    np.testing.assert_allclose(w2, w, rtol=0, atol=1e-10)
    assert warm_stats["inner"] < cold_stats["inner"], (warm_stats, cold_stats)
