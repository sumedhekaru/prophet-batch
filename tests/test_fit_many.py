"""prophet_batch.fit_many and its helpers (C1)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from api import has_multiplicative_terms, unsupported_reason

from synthetic import make_series
from api import BATCH_LOGGER  # noqa: E402


class TestEligibility:
    def test_default_model_is_supported(self):
        assert unsupported_reason(Prophet()) is None

    def test_logistic_is_not(self):
        assert "growth" in unsupported_reason(Prophet(growth="logistic"))

    def test_flat_is_supported(self):
        assert unsupported_reason(Prophet(growth="flat")) is None

    def test_mcmc_is_not(self):
        assert "mcmc" in unsupported_reason(Prophet(mcmc_samples=100))

    def test_regressor_predictor_is_not(self):
        m = Prophet()
        m.add_regressor("x", regressor_predictor=True)
        assert "regressor" in unsupported_reason(m)

    def test_fitted_model_is_not(self):
        m = Prophet(uncertainty_samples=0).fit(make_series(n_days=200))
        assert "fitted" in unsupported_reason(m)

    def test_multiplicative_detected_after_preprocess(self):
        df = make_series()
        df["y"] += 20
        assert has_multiplicative_terms(Prophet(seasonality_mode="multiplicative").preprocess(df))
        assert not has_multiplicative_terms(Prophet().preprocess(df))

    def test_additive_regressor_with_multiplicative_seasonality(self):
        df = make_series()
        df["y"] += 20
        df["x"] = np.arange(len(df)) % 3
        m = Prophet(seasonality_mode="multiplicative")
        m.add_regressor("x", mode="additive")
        assert has_multiplicative_terms(m.preprocess(df))


class TestDesignAndLayout:
    def test_design_matrix_reproduces_prophet_fitted_values(self):
        from api import design_matrix

        df = make_series()
        m = Prophet(uncertainty_samples=0)
        inputs = m.preprocess(df)
        m2 = Prophet(uncertainty_samples=0).fit(df)
        p = {k: np.ravel(v) for k, v in m2.params.items()}
        w = np.concatenate([[p["k"][0], p["m"][0]], p["delta"], p["beta"]])
        yhat = design_matrix(inputs) @ w * m2.y_scale
        expected = m2.predict(m2.history[["ds"]])["yhat"].to_numpy()
        np.testing.assert_allclose(yhat, expected, rtol=1e-10, atol=1e-10)

    def test_layout_round_trip_with_different_sizes(self):
        from api import design_matrix, stack_problems

        configs = [
            (make_series(n_days=1000), {}),                         # weekly + yearly
            (make_series(n_days=300), {}),                          # weekly only
            (make_series(n_days=1000), {"n_changepoints": 5}),
            (make_series(n_days=1000), {"changepoints": []}),       # dummy changepoint
        ]
        problems = []
        for df, kw in configs:
            inputs = Prophet(**kw).preprocess(df)
            problems.append((design_matrix(inputs), inputs.y, inputs.sigmas, inputs.tau))
        layout, G, C, yy, T, ridge, l1, tau = stack_problems(problems)
        rng = np.random.default_rng(0)
        for b, (Z, y, sigmas, _) in enumerate(problems):
            w_nat = rng.normal(size=Z.shape[1])
            w_lay = np.zeros(layout.p)
            w_lay[layout.index(b)] = w_nat
            k, m, delta, beta = layout.unpack(w_lay, b)
            np.testing.assert_array_equal(np.concatenate([[k, m], delta, beta]), w_nat)
            # The padded Gram row block reproduces Zᵀ Z w for the series' own coordinates.
            idx = layout.index(b)
            np.testing.assert_allclose((G[b] @ w_lay)[idx], Z.T @ Z @ w_nat)
            assert l1[b].sum() == len(delta) and not l1[b, layout.index(b)[:2]].any()
        assert sorted(set(layout.S.tolist())) == [1, 5, 25]
        assert sorted(set(layout.K.tolist())) == [6, 26]


def _batch_fit_by_hand(models_and_dfs):
    """Preprocess → stack → solve → set_params, without the public API (C1.4)."""
    from api import design_matrix, log_posterior, set_params, solve_map, stack_problems

    problems = []
    for m, df in models_and_dfs:
        inputs = m.preprocess(df)
        problems.append((design_matrix(inputs), inputs.y, inputs.sigmas, inputs.tau))
    layout, G, C, yy, T, ridge, l1, tau = stack_problems(problems)
    w, sigma = solve_map(G, C, yy, T, ridge, l1, tau)
    lp = log_posterior(G, C, yy, T, w, sigma, ridge, l1, tau)
    for b, (m, _) in enumerate(models_and_dfs):
        k, m_, delta, beta = layout.unpack(w[b], b)
        set_params(m, k, m_, delta, beta, sigma[b], lp[b])
    return [m for m, _ in models_and_dfs]


class TestParams:
    def test_shapes_and_dtypes_match_stan_fit(self):
        df = make_series()
        (ours,) = _batch_fit_by_hand([(Prophet(), df)])
        ref = Prophet().fit(df)
        assert ours.params.keys() == ref.params.keys()
        for k in ref.params:
            assert ours.params[k].shape == ref.params[k].shape, k
            assert ours.params[k].dtype == ref.params[k].dtype, k
        assert ours.stan_fit is None
        # trend is computed like Stan's transformed parameter.
        np.testing.assert_allclose(ours.params["trend"], ref.params["trend"], atol=5e-2)
        # The exact optimum has at least Stan's log posterior.
        assert ours.params["lp__"][0, 0] >= ref.params["lp__"][0, 0] - 1e-4

    def test_predict_works_with_uncertainty(self):
        df = make_series()
        (m,) = _batch_fit_by_hand([(Prophet(), df)])
        fc = m.predict(m.make_future_dataframe(30))
        assert len(fc) == len(df) + 30
        assert (fc["yhat_lower"] <= fc["yhat"]).all() and (fc["yhat"] <= fc["yhat_upper"]).all()

    def test_constant_series_matches_fit(self):
        from api import is_constant, set_constant_params

        df = make_series(n_days=200)
        df["y"] = 3.0
        ref = Prophet().fit(df)
        m = Prophet()
        inputs = m.preprocess(df)
        assert is_constant(m)
        set_constant_params(m, inputs.K)
        for k in ref.params:
            np.testing.assert_array_equal(m.params[k], ref.params[k])

    def test_no_changepoints_folds_delta_into_k(self):
        df = make_series()
        (m,) = _batch_fit_by_hand([(Prophet(changepoints=[]), df)])
        ref = Prophet(changepoints=[]).fit(df)
        assert m.params["delta"].shape == ref.params["delta"].shape
        assert np.all(m.params["delta"] == 0)
        # No changepoint penalty in play: a smooth problem, so Stan should agree closely.
        np.testing.assert_allclose(m.params["k"], ref.params["k"], rtol=1e-4)


def _long(frames):
    return pd.concat([f.assign(unique_id=sid) for sid, f in frames.items()], ignore_index=True)


class TestFitMany:
    def test_mixed_batch(self, caplog):
        from api import fit_many

        flat = make_series(n_days=500, seed=2)
        const = make_series(n_days=300, seed=3)
        const["y"] = 4.0
        frames = {"a": make_series(seed=0), "b": make_series(n_days=400, seed=1),
                  "c_flat": flat, "d_const": const}
        configs = {"c_flat": dict(mcmc_samples=20)}

        def make_model(sid):
            return Prophet(**configs.get(sid, {}))

        with caplog.at_level("INFO", logger=BATCH_LOGGER):
            models = fit_many(_long(frames), make_model=make_model)
        assert list(models) == list(frames)
        assert "2 series batched (2 additive, 0 multiplicative), 1 constant, 1 fell back" in caplog.text
        for sid, m in models.items():
            fc = m.predict(m.make_future_dataframe(14))
            assert fc["yhat"].notna().all(), sid
        assert models["a"].stan_fit is None
        assert models["c_flat"].stan_fit is not None  # Stan fallback (MCMC)

    def test_rejects_kwargs_with_make_model(self):
        import pytest
        from api import fit_many

        with pytest.raises(ValueError):
            fit_many(_long({"a": make_series()}), make_model=lambda sid: Prophet(), growth="linear")


def _kkt_and_lp(model, df, make_model):
    """KKT violation and log posterior of a fitted model's params, recomputed from scratch."""
    from api import design_matrix, kkt_violation, log_posterior, stack_problems

    inputs = make_model().preprocess(df)
    layout, G, C, yy, T, ridge, l1, tau = stack_problems(
        [(design_matrix(inputs), inputs.y, inputs.sigmas, inputs.tau)])
    p = {k: np.ravel(v) for k, v in model.params.items()}
    if len(model.changepoints) == 0:
        return None, None  # delta was folded into k; the dummy-changepoint split isn't recoverable
    w = np.concatenate([[p["k"][0], p["m"][0]], p["delta"], p["beta"]])[None, :]
    sigma = p["sigma_obs"]
    return (kkt_violation(G, C, w, sigma, ridge, l1, tau)[0],
            log_posterior(G, C, yy, T, w, sigma, ridge, l1, tau)[0])


def _holiday_regressor_frame(seed):
    df = make_series(seed=seed)
    rng = np.random.default_rng(seed + 1000)  # independent of make_series' noise
    df["x"] = rng.normal(size=len(df))
    df["y"] = df["y"] + 0.8 * df["x"]
    return df


HOLIDAYS = pd.DataFrame({"holiday": "promo", "ds": pd.date_range("2020-02-01", periods=12, freq="90D"),
                         "lower_window": 0, "upper_window": 1})


def _holiday_regressor_model(sid=None):
    m = Prophet(holidays=HOLIDAYS)
    m.add_regressor("x")
    m.add_country_holidays("US")
    return m


class TestOptimality:
    """C1.6: batched fits are exact optima and at least as good as Stan's."""

    def _check(self, frames, make_model):
        from api import fit_many

        models = fit_many(_long(frames), make_model=lambda sid: make_model())
        for sid, df in frames.items():
            m = models[sid]
            assert m.stan_fit is None, f"{sid} was not batched"
            kkt, lp = _kkt_and_lp(m, df, make_model)
            ref = make_model().fit(df)
            _, lp_stan = _kkt_and_lp(ref, df, make_model)
            assert kkt < 1e-8, (sid, kkt)
            assert lp >= lp_stan - 1e-6, (sid, lp, lp_stan)
            np.testing.assert_allclose(m.params["lp__"][0, 0], lp, rtol=1e-12)

    def test_flat_growth(self):
        frames = {f"f{i}": make_series(n_days=n, seed=i) for i, n in enumerate([300, 700, 1000])}
        self._check(frames, lambda: Prophet(growth="flat"))
        models = __import__("api", fromlist=["fit_many"]).fit_many(_long(frames), growth="flat")
        for m in models.values():
            assert np.all(m.params["k"] == 0) and np.all(m.params["delta"] == 0)
            np.testing.assert_array_equal(m.params["trend"], np.full_like(m.params["trend"], m.params["m"][0, 0]))

    def test_minmax_scaling(self):
        frames = {f"m{i}": make_series(seed=i).assign(y=lambda d: d["y"] + 50) for i in range(3)}
        self._check(frames, lambda: Prophet(scaling="minmax"))

    def test_conditional_seasonality(self):
        def frame(i):
            df = make_series(seed=i)
            df["weekend"] = df["ds"].dt.dayofweek >= 5
            df["weekday"] = ~df["weekend"]
            return df

        def model():
            m = Prophet(weekly_seasonality=False)
            m.add_seasonality(name="weekly_on_weekend", period=7, fourier_order=3, condition_name="weekend")
            m.add_seasonality(name="weekly_on_weekday", period=7, fourier_order=3, condition_name="weekday")
            return m

        self._check({f"c{i}": frame(i) for i in range(3)}, model)

    def test_synthetic_different_lengths(self):
        frames = {f"s{i}": make_series(n_days=n, seed=i) for i, n in enumerate([200, 450, 800, 1000, 1300])}
        self._check(frames, Prophet)

    def test_holidays_and_additive_regressor(self):
        frames = {f"h{i}": _holiday_regressor_frame(i) for i in range(4)}
        self._check(frames, _holiday_regressor_model)

    def test_m5_sample(self):
        import pytest

        from conftest import WORKSPACE

        sys_path = WORKSPACE / "benchmarks"
        if not (WORKSPACE / "data" / "m5" / "calendar.csv").exists():
            pytest.skip("M5 data not downloaded")
        import sys
        sys.path.insert(0, str(sys_path))
        from m5 import load_series

        frames = dict(load_series(15, "item", seed=7))
        self._check(frames, Prophet)


class TestFallbackAndEdgeCases:
    def test_fallback_is_bit_identical_to_fit(self):
        from api import fit_many

        df = make_series()
        df["cap"] = df["y"].max() * 1.2
        models = fit_many(_long({"a": df}), growth="logistic")
        ref = Prophet(growth="logistic").fit(df)
        for k in ref.params:
            np.testing.assert_array_equal(models["a"].params[k], ref.params[k])
        fut = models["a"].make_future_dataframe(10)
        fut["cap"] = df["cap"].iloc[0]
        np.random.seed(0)
        f1 = models["a"].predict(fut)
        np.random.seed(0)
        f2 = ref.predict(fut)
        pd.testing.assert_frame_equal(f1, f2)

    def test_no_changepoints_in_fit_many(self):
        from api import fit_many

        df = make_series()
        models = fit_many(_long({"a": df}), changepoints=[])
        ref = Prophet(changepoints=[]).fit(df)
        assert models["a"].stan_fit is None
        assert np.all(models["a"].params["delta"] == 0)
        assert models["a"].params["lp__"][0, 0] >= ref.params["lp__"][0, 0] - 1e-4

    def test_perfect_fit_falls_back_to_prophet_fit(self):
        """No finite MAP when the data can be fit exactly (D1.4, MS-73): Prophet.fit takes over."""
        from api import fit_many

        df = make_series(n_days=400)
        t = np.arange(len(df))
        df["y"] = 5 + 0.01 * t + 2 * np.sin(2 * np.pi * t / 7)  # exactly representable, no noise
        m = fit_many(_long({"a": df}))["a"]
        assert m.stan_fit is not None
        assert_same(m.params, Prophet().fit(df).params)
        fc = m.predict(m.make_future_dataframe(7))
        assert np.isfinite(fc[["yhat", "yhat_lower", "yhat_upper"]].to_numpy()).all()


def assert_same(a, b, path="model"):
    """Deep equality for model state: DataFrames, Series, arrays, dicts, scalars."""
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, obj=path)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, obj=path)
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            assert_same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]")
    else:
        assert type(a) is type(b) and (a == b or (a is None and b is None)), (path, a, b)


class TestSharedDates:
    """C2.2: the shared-dates fast path produces exactly the per-series preprocess state."""

    def _check(self, frames, make_model=None, groups=None, **kw):
        """Shared preprocessing equals per-series preprocess; `groups`: expected distinct Zs."""
        import dataclasses

        from api import _prepare

        make_model = make_model or (lambda sid: Prophet(**kw))
        df = _long(frames)
        order, batch, mult, constant, fallback, skipped = _prepare(
            df, "unique_id", make_model, True)
        assert not fallback and not skipped and len(batch) == len(frames)
        n_z = len({id(Z) for _, _, _, Z, _ in batch})
        if groups is None:
            assert n_z < len(batch), "nothing was shared"
        else:
            assert n_z == groups, (n_z, groups)
        for sid, model, inputs, Z, _ in batch:
            ref = make_model(sid)
            ref_inputs = ref.preprocess(frames[sid])
            state, ref_state = vars(model), vars(ref)
            for key in set(state) | set(ref_state):
                if key == "stan_backend":
                    continue
                assert_same(state[key], ref_state[key], f"{sid}.{key}")
            for f in dataclasses.fields(ref_inputs):
                assert_same(getattr(inputs, f.name), getattr(ref_inputs, f.name), f"{sid}.inputs.{f.name}")

    def test_same_dates_and_mixed(self):
        frames = {f"s{i}": make_series(seed=i) for i in range(4)}
        with_nan = make_series(seed=9)
        with_nan.loc[[5, 50, 500], "y"] = np.nan
        frames["nan"] = with_nan
        frames["short"] = make_series(n_days=500, seed=11)
        self._check(frames)

    def test_holidays_and_minmax_scaling(self):
        frames = {f"s{i}": make_series(seed=i) for i in range(3)}
        self._check(frames, holidays=HOLIDAYS, scaling="minmax")

    def test_fit_results_same_as_without_sharing(self):
        from api import fit_many

        frames = {f"s{i}": make_series(seed=i) for i in range(5)}
        shared = fit_many(_long(frames))
        unshared = fit_many(_long(frames), make_model=lambda sid: Prophet())
        for sid in frames:
            for k in shared[sid].params:
                np.testing.assert_array_equal(shared[sid].params[k], unshared[sid].params[k])

    # F3.2: make_model series share when their models and regressor / condition data match.

    @staticmethod
    def _with_x(frames, x_for):
        return {sid: f.assign(x=x_for(sid, f)) for sid, f in frames.items()}

    def test_make_model_identical_config(self):
        frames = {f"s{i}": make_series(seed=i) for i in range(4)}
        self._check(frames, lambda sid: Prophet(holidays=HOLIDAYS), groups=1)

    def test_make_model_common_regressor(self):
        frames = self._with_x({f"s{i}": make_series(seed=i) for i in range(4)},
                              lambda sid, f: np.sin(np.arange(len(f)) / 7.0))
        self._check(frames, _holiday_regressor_model, groups=1)

    def test_make_model_per_series_regressor(self):
        frames = self._with_x({f"s{i}": make_series(seed=i) for i in range(4)},
                              lambda sid, f: np.random.default_rng(len(sid) + int(sid[1:])).normal(size=len(f)))
        self._check(frames, _holiday_regressor_model, groups=4)

    def test_make_model_regressor_groups(self):
        # two calendars, each shared by two series; standardization uses each group's own values
        frames = self._with_x({f"s{i}": make_series(seed=i) for i in range(4)},
                              lambda sid, f: (np.arange(len(f)) % (5 + int(sid[1:]) % 2)).astype(float))
        self._check(frames, _holiday_regressor_model, groups=2)

    def test_make_model_conditional_seasonality(self):
        def make_model(sid):
            m = Prophet(weekly_seasonality=False)
            m.add_seasonality("weekly_on", period=7, fourier_order=3, condition_name="on")
            return m

        frames = {f"s{i}": make_series(seed=i).assign(on=lambda f: f["ds"].dt.month < 7) for i in range(3)}
        frames["off"] = make_series(seed=7).assign(on=lambda f: f["ds"].dt.month >= 7)
        self._check(frames, make_model, groups=2)

    def test_make_model_mixed_configs(self):
        later = HOLIDAYS.assign(ds=HOLIDAYS["ds"] + pd.Timedelta(days=1))  # same shape, other dates

        def make_model(sid):
            i = int(sid[1:])
            if i % 3 == 0:
                return Prophet(holidays=HOLIDAYS)
            if i % 3 == 1:
                return Prophet(holidays=later)
            return Prophet(holidays=HOLIDAYS, holidays_prior_scale=5.0)

        frames = {f"s{i}": make_series(seed=i) for i in range(6)}
        self._check(frames, make_model, groups=3)

    def test_make_model_fit_same_as_without_sharing(self, monkeypatch):
        import api
        from api import fit_many

        frames = self._with_x({f"s{i}": make_series(seed=i) for i in range(5)},
                              lambda sid, f: np.cos(np.arange(len(f)) / 30.0))
        shared = fit_many(_long(frames), make_model=_holiday_regressor_model)
        monkeypatch.setattr(api.MODULE, "_MAX_CONFIGS_PER_KEY", 0)
        unshared = fit_many(_long(frames), make_model=_holiday_regressor_model)
        for sid in frames:
            assert shared[sid].extra_regressors == unshared[sid].extra_regressors
            for k in shared[sid].params:
                np.testing.assert_array_equal(shared[sid].params[k], unshared[sid].params[k])


class TestSolverFailures:
    """D1.4.3: a series the batched solver can't handle is fit by Prophet.fit; nothing else changes."""

    def test_degenerate_case_fits(self):
        """Seed 140 (57 rows, 83 parameters) made the solver singular before E6; now it is solved.
        Either way the result must be Prophet.fit's exactly or a batched fit at least as good as Stan's."""
        from api import fit_many
        from random_cases import build, make_spec

        spec = make_spec(140)
        _, df, _ = build(spec, Prophet)
        ours = fit_many(df.assign(unique_id="s"), make_model=lambda sid: build(spec, Prophet)[0])["s"]
        ref = build(spec, Prophet)[0].fit(df)
        if ours.stan_fit is not None:
            assert_same(ours.params, ref.params)
        else:
            assert ours.params["lp__"][0, 0] >= ref.params["lp__"][0, 0] - 1e-6

    def test_failure_is_isolated_within_its_chunk(self, monkeypatch, caplog):
        import logging

        from api import MODULE as batch

        caplog.set_level(logging.DEBUG, logger=BATCH_LOGGER)

        frames = {f"m{i}": make_series(n_days=600, seed=30 + i).assign(y=lambda d: d["y"] + 20)
                  for i in range(4)}
        df = _long(frames)
        make = lambda sid: Prophet(seasonality_mode="multiplicative")  # noqa: E731
        clean = batch.fit_many(df, make_model=make)

        bad_y = Prophet().setup_dataframe(frames["m2"].copy(), initialize_scales=True)["y_scaled"]
        original = batch._MultiplicativeBatch.solve

        def solve(self, **kw):
            if np.any(np.all(np.isclose(self.y, bad_y.to_numpy()[None, :]), axis=1)):
                raise np.linalg.LinAlgError("forced")
            return original(self, **kw)

        monkeypatch.setattr(batch._MultiplicativeBatch, "solve", solve)
        models = batch.fit_many(df, make_model=make)
        assert "Multiplicative chunk of 4 series failed (forced); solving them one at a time" in caplog.text
        assert list(models) == list(frames)
        for sid in ("m0", "m1", "m3"):
            assert models[sid].stan_fit is None
            assert_same(models[sid].params, clean[sid].params)
        assert models["m2"].stan_fit is not None
        assert_same(models["m2"].params, make("m2").fit(frames["m2"]).params)

    @staticmethod
    def _additive_frames():
        # Different lengths, one shape group: the patched solver recognises "a2" by its row count.
        return {f"a{i}": make_series(n_days=400 + 10 * i, seed=50 + i) for i in range(4)}

    def test_additive_failure_is_isolated_within_its_group(self, monkeypatch, caplog):
        """D4.4.1: like the multiplicative path, a singular additive solve no longer aborts the call."""
        import logging

        from api import MODULE as batch

        caplog.set_level(logging.DEBUG, logger=BATCH_LOGGER)
        frames = self._additive_frames()
        df = _long(frames)
        clean = batch.fit_many(df)
        original = batch.solve_map

        def solve_map(G, C, yy, T, *args, **kw):
            if np.any(np.asarray(T) == 420):
                raise np.linalg.LinAlgError("forced")
            return original(G, C, yy, T, *args, **kw)

        monkeypatch.setattr(batch, "solve_map", solve_map)
        models = batch.fit_many(df)
        assert "Additive group of 4 series failed (forced); solving them one at a time" in caplog.text
        assert list(models) == list(frames)
        for sid in ("a0", "a1", "a3"):
            assert models[sid].stan_fit is None
            assert_same(models[sid].params, clean[sid].params)
        assert models["a2"].stan_fit is not None
        assert_same(models["a2"].params, Prophet().fit(frames["a2"]).params)

    def test_additive_non_finite_solution_falls_back(self, monkeypatch):
        from api import MODULE as batch

        frames = self._additive_frames()
        original = batch.solve_map

        def solve_map(G, C, yy, T, *args, **kw):
            w, sigma = original(G, C, yy, T, *args, **kw)
            sigma = np.where(np.asarray(T) == 420, np.nan, sigma)
            return w, sigma

        monkeypatch.setattr(batch, "solve_map", solve_map)
        models = batch.fit_many(_long(frames))
        assert models["a2"].stan_fit is not None
        assert_same(models["a2"].params, Prophet().fit(frames["a2"]).params)
        assert all(models[sid].stan_fit is None for sid in ("a0", "a1", "a3"))


def logistic_without_cap(sid):
    """Module-level (picklable): 'nocap' is logistic but its data has no cap, so Prophet.fit raises."""
    return Prophet(growth="logistic") if sid == "nocap" else Prophet()


class TestInputValidation:
    """D4.4.1: input problems raise up front, the same way for any n_jobs."""

    def test_missing_ids_raise(self):
        from api import fit_many

        df = _long({"a": make_series(seed=0), "b": make_series(seed=1)})
        df.loc[5:9, "unique_id"] = None
        for n_jobs in (1, 2):
            with pytest.raises(ValueError, match="5 rows have a missing unique_id"):
                fit_many(df, n_jobs=n_jobs)

    def test_missing_y_column_raises_prophets_error(self):
        from api import fit_many

        df = _long({"a": make_series(seed=0)}).rename(columns={"y": "value"})
        with pytest.raises(ValueError, match='must have columns "ds" and "y"'):
            fit_many(df)

    def test_invalid_prophet_kwargs_raise_for_any_n_jobs(self):
        from api import fit_many

        df = _long({"a": make_series(seed=0), "b": make_series(seed=1)})
        for n_jobs in (1, 2):
            with pytest.raises(ValueError, match="Parameter \"growth\" should be"):
                fit_many(df, n_jobs=n_jobs, on_error="skip", growth="bogus")

    def test_structured_exception_args_are_kept(self):
        from api import MODULE as batch

        e = OSError(2, "No such file or directory")
        batch._series_failed("x", e, "raise")
        assert e.errno == 2 and e.strerror == "No such file or directory"
        k = KeyError("col")
        batch._series_failed("x", k, "raise")
        assert k.args == ("col",)
        v = ValueError("bad")
        batch._series_failed("x", v, "raise")
        assert v.args == ("series 'x': bad",)


class TestOnError:
    """D1.4.6: fit_many(on_error='raise'|'skip')."""

    def _frames(self):
        frames = {f"ok{i}": make_series(seed=i) for i in range(4)}
        frames["one_row"] = make_series().iloc[:1]  # preprocessing fails
        return frames

    def test_raise_names_the_series(self):
        from api import fit_many

        with pytest.raises(ValueError, match=r"^series 'one_row': Dataframe has less than 2 non-NaN rows"):
            fit_many(_long(self._frames()))

    def test_skip_leaves_out_only_the_bad_series(self, caplog):
        from api import fit_many

        frames = self._frames()
        with caplog.at_level(logging.WARNING, logger=BATCH_LOGGER):
            models = fit_many(_long(frames), on_error="skip")
        assert list(models) == ["ok0", "ok1", "ok2", "ok3"]
        assert "skipping series 'one_row': ValueError" in caplog.text
        clean = fit_many(_long({k: v for k, v in frames.items() if k != "one_row"}))
        for sid in clean:
            assert_same(models[sid].params, clean[sid].params)

    def test_fallback_errors_too(self):
        from api import fit_many

        frames = {"ok0": make_series(seed=0), "nocap": make_series(seed=1)}
        with pytest.raises(ValueError, match=r"^series 'nocap': "):
            fit_many(_long(frames), make_model=logistic_without_cap)
        models = fit_many(_long(frames), make_model=logistic_without_cap, on_error="skip")
        assert list(models) == ["ok0"]

    def test_parallel_matches_serial(self):
        from api import fit_many

        df = _long(self._frames())
        serial = fit_many(df, on_error="skip")
        parallel = fit_many(df, on_error="skip", n_jobs=3)
        assert list(parallel) == list(serial)
        for sid in serial:
            assert_same(parallel[sid].params, serial[sid].params)
        with pytest.raises(ValueError, match=r"^series 'one_row': "):
            fit_many(df, n_jobs=3)
        with pytest.raises(ValueError, match=r"^series 'nocap': "):
            fit_many(_long({"ok0": make_series(seed=0), "nocap": make_series(seed=1)}),
                     make_model=logistic_without_cap, n_jobs=2)

    def test_rejects_unknown_value(self):
        from api import fit_many

        with pytest.raises(ValueError, match="on_error must be"):
            fit_many(_long({"a": make_series()}), on_error="ignore")


def newton(sid):
    """Module-level (picklable) per-series fit kwargs."""
    return {"algorithm": "Newton"}


class TestFitKwargs:
    """E2.1: fit_many(fit_kwargs=dict | callable) passes Prophet.fit keyword arguments."""

    def _logistic(self, seed):
        df = make_series(seed=seed)
        return df.assign(cap=df["y"].max() * 1.2)

    def test_dict_reaches_stan_and_is_stored(self):
        from api import fit_many

        lg = self._logistic(0)
        models = fit_many(_long({"lg": lg}), make_model=lambda sid: Prophet(growth="logistic"),
                          fit_kwargs={"algorithm": "Newton"})
        ref = Prophet(growth="logistic").fit(lg, algorithm="Newton")
        assert_same(models["lg"].params, ref.params)
        assert models["lg"].fit_kwargs == {"algorithm": "Newton"}
        for kw in ({}, {"make_model": lambda sid: Prophet()}):  # shared-dates and make_model paths
            m = fit_many(_long({"a": make_series(seed=1), "b": make_series(seed=2)}),
                         fit_kwargs={"algorithm": "Newton"}, **kw)
            assert m["a"].stan_fit is None and m["a"].fit_kwargs == {"algorithm": "Newton"}

    def test_per_series_warm_start(self):
        from api import fit_many
        from prophet.utilities import warm_start_params

        old_df, new_df = self._logistic(3).iloc[:800], self._logistic(3)
        make = lambda sid: Prophet(growth="logistic")  # noqa: E731
        old = fit_many(_long({"lg": old_df}), make_model=make)
        new = fit_many(_long({"lg": new_df}), make_model=make,
                       fit_kwargs=lambda sid: {"init": warm_start_params(old[sid])})
        stock_old = Prophet(growth="logistic").fit(old_df)
        stock_new = Prophet(growth="logistic").fit(new_df, init=warm_start_params(stock_old))
        assert_same(new["lg"].params, stock_new.params)

    def test_cross_validation_reuses_them(self):
        from api import fit_many
        from prophet.diagnostics import cross_validation

        df = make_series(seed=4)
        ours = fit_many(_long({"a": df}), fit_kwargs={"algorithm": "Newton"})["a"]
        stock = Prophet().fit(df, algorithm="Newton")
        cv = dict(initial="700 days", period="150 days", horizon="30 days", disable_tqdm=True)
        np.random.seed(0)
        a = cross_validation(ours, **cv)
        np.random.seed(0)
        b = cross_validation(stock, **cv)
        pd.testing.assert_frame_equal(a, b)

    def test_parallel_and_validation(self):
        from api import fit_many

        df = _long({"a": make_series(seed=5), "lg": self._logistic(6)})
        make = logistic_without_cap  # logistic only for id 'nocap': plain Prophet here
        serial = fit_many(df, make_model=make, fit_kwargs=newton)
        parallel = fit_many(df, make_model=make, fit_kwargs=newton, n_jobs=2)
        for sid in serial:
            assert_same(parallel[sid].params, serial[sid].params)
            assert parallel[sid].fit_kwargs == {"algorithm": "Newton"}
        with pytest.raises(ValueError, match="fit_kwargs must be picklable"):
            fit_many(df, make_model=make, fit_kwargs=lambda sid: {}, n_jobs=2)
        with pytest.raises(ValueError, match="fit_kwargs must be a dict or a callable"):
            fit_many(df, fit_kwargs=["algorithm"])


def test_mcmc_matches_stock_with_a_seed():
    """E2.2: MCMC series go to Stan; with a sampler seed (via fit_kwargs) they match stock bit for bit."""
    from api import fit_many

    df = make_series(n_days=400)
    kw = {"seed": 1, "show_progress": False}
    ours = fit_many(_long({"s": df}), mcmc_samples=40, fit_kwargs=kw)["s"]
    stock = Prophet(mcmc_samples=40).fit(df, **kw)
    assert ours.stan_fit is not None
    assert_same(ours.params, stock.params)
    future = stock.make_future_dataframe(14)
    np.random.seed(0)
    a = ours.predict(future)
    np.random.seed(0)
    b = stock.predict(future)
    pd.testing.assert_frame_equal(a, b)
