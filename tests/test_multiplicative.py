"""C6: batched MAP for multiplicative seasonality (alternating exact blocks)."""

import numpy as np
from prophet import Prophet
from api import _gram, _MultiplicativeBatch, kkt_violation

from synthetic import make_series


def _frames(n=4):
    out = []
    for i, T in enumerate([1000, 700, 1000, 450][:n]):
        df = make_series(n_days=T, seed=i)
        df["y"] = df["y"] + 20
        out.append(df)
    return out


def _block_kkt(mb, theta, ba, bm, sigma):
    """KKT violation of each block at the solution, given the other block."""
    B, P = theta.shape
    ridge_A = np.concatenate([np.full((B, 2), 1 / 25), np.where(mb.delta_mask, 0.0, 1.0), mb.ridge_a], axis=1)
    l1_A = np.concatenate([np.zeros((B, 2), bool), mb.delta_mask, np.zeros_like(mb.ridge_a, bool)], axis=1)
    wgt = 1 + np.matmul(mb.Xm, bm[:, :, None])[..., 0]
    G, C, _ = _gram(np.concatenate([wgt[:, :, None] * mb.ZT, mb.Xa], axis=2), mb.y)
    kA = kkt_violation(G, C, np.concatenate([theta, ba], axis=1), sigma, ridge_A, l1_A, mb.tau)
    trend = np.matmul(mb.ZT, theta[:, :, None])[..., 0]
    G, C, _ = _gram(np.concatenate([trend[:, :, None] * mb.Xm, mb.Xa], axis=2), mb.y - trend)
    ridge_B = np.concatenate([mb.ridge_m, mb.ridge_a], axis=1)
    kB = kkt_violation(G, C, np.concatenate([bm, ba], axis=1), sigma, ridge_B, np.zeros_like(ridge_B, bool), mb.tau)
    return kA, kB


def test_converges_to_stationary_point_better_than_stan():
    frames = _frames()
    inputs = [Prophet(seasonality_mode="multiplicative").preprocess(df) for df in frames]
    mb = _MultiplicativeBatch(inputs)
    theta, ba, bm, sigma, lp, iters = mb.solve()
    kA, kB = _block_kkt(mb, theta, ba, bm, sigma)
    assert kA.max() < 1e-5 and kB.max() < 1e-5, (kA, kB, iters)
    for b, df in enumerate(frames):
        ref = Prophet(seasonality_mode="multiplicative").fit(df)
        assert lp[b] >= ref.params["lp__"][0, 0] - 1e-4, (b, lp[b], ref.params["lp__"][0, 0])


def test_mixed_additive_and_multiplicative_features():
    frames = _frames(3)
    rng = np.random.default_rng(42)
    for df in frames:
        df["x"] = rng.normal(size=len(df))
        df["y"] = df["y"] + 0.5 * df["x"]

    def model():
        m = Prophet(seasonality_mode="multiplicative")
        m.add_regressor("x", mode="additive")
        return m

    inputs = [model().preprocess(df) for df in frames]
    mb = _MultiplicativeBatch(inputs)
    assert mb.Xa.shape[2] == 1 and mb.Xm.shape[2] == 26
    theta, ba, bm, sigma, lp, _ = mb.solve()
    kA, kB = _block_kkt(mb, theta, ba, bm, sigma)
    assert kA.max() < 1e-5 and kB.max() < 1e-5
    for b, df in enumerate(frames):
        assert lp[b] >= model().fit(df).params["lp__"][0, 0] - 1e-4


def test_gauss_newton_reaches_tight_kkt():
    frames = _frames()
    mb = _MultiplicativeBatch([Prophet(seasonality_mode="multiplicative").preprocess(df) for df in frames])
    theta, ba, bm, sigma, lp, iters = mb.solve(method="gauss-newton")
    kA, kB = _block_kkt(mb, theta, ba, bm, sigma)
    # 1e-8 was measured with the old inner sigma loop; since E6 (one w-step per subproblem, damped steps)
    # the stop rule fires at block KKT 4e-8 / 2.5e-7 with lp equal to the old, tighter point to 1e-12.
    # Still 1000x below fit_many's acceptance threshold (1e-3).
    assert kA.max() < 1e-6 and kB.max() < 1e-6 and iters <= 15


class TestFitManyMultiplicative:
    """C6.3: multiplicative models go through the batched path in fit_many."""

    def _check(self, frames, make_model):
        from api import fit_many
        import pandas as pd

        long = pd.concat([f.assign(unique_id=i) for i, f in enumerate(frames)], ignore_index=True)
        models = fit_many(long, make_model=lambda sid: make_model())
        for i, df in enumerate(frames):
            m = models[i]
            assert m.stan_fit is None, f"series {i} fell back"
            ref = make_model().fit(df)
            assert m.params["lp__"][0, 0] >= ref.params["lp__"][0, 0] - 1e-4
            for k in ref.params:
                assert m.params[k].shape == ref.params[k].shape, k
            fut = m.make_future_dataframe(14)
            if "x" in df:
                fut["x"] = np.concatenate([df["x"].to_numpy(), np.zeros(14)])
            fc = m.predict(fut)
            assert np.isfinite(fc[["yhat", "yhat_lower", "yhat_upper"]].to_numpy()).all()

    def test_multiplicative_seasonality(self):
        self._check(_frames(), lambda: Prophet(seasonality_mode="multiplicative"))

    def test_flat_growth_multiplicative(self):
        self._check(_frames(3), lambda: Prophet(growth="flat", seasonality_mode="multiplicative"))

    def test_additive_regressor_with_multiplicative_seasonality(self):
        frames = _frames(3)
        rng = np.random.default_rng(42)
        for df in frames:
            df["x"] = rng.normal(size=len(df))
            df["y"] = df["y"] + 0.5 * df["x"]

        def model():
            m = Prophet(seasonality_mode="multiplicative")
            m.add_regressor("x", mode="additive")
            return m

        self._check(frames, model)

    def test_shared_dates_multiplicative(self):
        from api import fit_many
        import pandas as pd

        frames = {f"s{i}": make_series(seed=i).assign(y=lambda d: d["y"] + 20) for i in range(4)}
        long = pd.concat([f.assign(unique_id=k) for k, f in frames.items()], ignore_index=True)
        shared = fit_many(long, seasonality_mode="multiplicative")
        unshared = fit_many(long, make_model=lambda sid: Prophet(seasonality_mode="multiplicative"))
        for k in frames:
            for p in shared[k].params:
                np.testing.assert_array_equal(shared[k].params[p], unshared[k].params[p])
