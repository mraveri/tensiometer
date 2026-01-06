"""Tests for KDE-based MCMC tension estimators."""

#########################################################################################################
# Imports

import builtins
import importlib
import os
import types
import unittest
from unittest.mock import patch

import numpy as np
from getdist import MCSamples, loadMCSamples

import tensiometer.mcmc_tension.kde as mt
import tensiometer.mcmc_tension.param_diff as pd
import tensiometer.utilities.stats_utilities as stut

#########################################################################################################
# KDE shift tests


class TestMcmcKdeShift(unittest.TestCase):

    """MCMC KDE shift test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.here = os.path.dirname(os.path.abspath(__file__))
        self.chain_1 = loadMCSamples(self.here + "/../../test_chains/DES")
        self.chain_2 = loadMCSamples(self.here + "/../../test_chains/Planck18TTTEEE")
        self.chain_12 = loadMCSamples(self.here + "/../../test_chains/Planck18TTTEEE_DES")
        self.chain_prior = loadMCSamples(self.here + "/../../test_chains/prior")
        # keep joblib sequential for test environments that block semaphores
        mt.n_threads = 1
        # patch logsumexp to avoid SciPy array-api quirks in this environment
        self._orig_logsumexp = mt.scipy.special.logsumexp
        mt.scipy.special.logsumexp = lambda a, b=None, axis=None, return_sign=False, xp=None: np.log(
            np.sum(np.exp(a)*(1 if b is None else b))
        )
        # Disable numba JIT during tests to avoid dispatcher compilation in small fixtures.
        self._orig_jit = mt.jit
        mt.jit = lambda *args, **kwargs: (lambda f: f)
        # thin the chain:
        self.chain_1.getConvergeTests()
        self.chain_2.getConvergeTests()
        self.chain_12.getConvergeTests()
        self.chain_prior.getConvergeTests()
        self.chain_1.weighted_thin(int(self.chain_1.indep_thin))
        self.chain_2.weighted_thin(int(self.chain_2.indep_thin))
        self.chain_12.weighted_thin(int(self.chain_12.indep_thin))
        self.chain_prior.weighted_thin(int(self.chain_prior.indep_thin))
        # Keep numba helpers running in pure Python to avoid JIT compile errors in tests.
        self._orig_mise1d = mt._mise1d_optimizer
        self._orig_mise1d_jac = mt._mise1d_optimizer_jac
        self._orig_mise = mt._mise_optimizer
        self._orig_ell_helper = mt._ell_helper
        if hasattr(mt._mise1d_optimizer, "py_func"):
            mt._mise1d_optimizer = mt._mise1d_optimizer.py_func
        if hasattr(mt._mise1d_optimizer_jac, "py_func"):
            mt._mise1d_optimizer_jac = mt._mise1d_optimizer_jac.py_func
        if hasattr(mt._mise_optimizer, "py_func"):
            mt._mise_optimizer = mt._mise_optimizer.py_func
        if hasattr(mt._ell_helper, "py_func"):
            mt._ell_helper = mt._ell_helper.py_func
        self.diff_chain = pd.parameter_diff_chain(self.chain_1, self.chain_2, boost=1)

    def tearDown(self):
        """Clean up test fixtures."""
        mt.scipy.special.logsumexp = self._orig_logsumexp
        mt.jit = self._orig_jit
        mt._mise1d_optimizer = self._orig_mise1d
        mt._mise1d_optimizer_jac = self._orig_mise1d_jac
        mt._mise_optimizer = self._orig_mise
        mt._ell_helper = self._orig_ell_helper

    def test_kde_shift(self):
        """Test KDE shift consistency across methods."""
        res_1 = mt.kde_parameter_shift(self.diff_chain,
                                       method='brute_force',
                                       scale=0.5)
        res_2 = mt.kde_parameter_shift(self.diff_chain,
                                       method='neighbor_elimination',
                                       scale=0.5)
        print(res_1, res_2)
        assert np.allclose(res_1, res_2)
        res_3 = mt.kde_parameter_shift(self.diff_chain,
                                       method='brute_force',
                                       feedback=2,
                                       scale=0.5)
        assert np.allclose(res_1, res_3)

    def test_helper_branches(self):
        """Test helper branches and error paths."""
        samples = self.diff_chain.samples[:50, :2]
        weights = self.diff_chain.weights[:50]
        bw = mt.UCV_SP_bandwidth(samples, weights, near=2, feedback=0)
        self.assertTrue(hasattr(bw, "x"))
        res = mt.kde_parameter_shift(self.diff_chain,
                                     method='neighbor_elimination',
                                     scale=0.5,
                                     zero_prob=-1.0,
                                     num_samples=100,
                                     feedback=0)
        self.assertTrue(np.isfinite(res[0]))
        coords = np.random.rand(10, 2)
        dist_weights = np.ones(10)
        mt._gauss_ballkde_logpdf(coords[0], coords, weights[:10], dist_weights)
        ell_weights = np.array([np.eye(2) for _ in range(10)])
        mt._gauss_ellkde_logpdf(coords[0], coords, weights[:10], ell_weights)
        with self.assertRaises(ValueError):
            mt.kde_parameter_shift(self.diff_chain, method='invalid')
        res_4 = mt.kde_parameter_shift(self.diff_chain,
                                       method='neighbor_elimination',
                                       feedback=2,
                                       scale=0.5)
        self.assertTrue(np.isfinite(res_4[0]))
        param_names = ['delta_omegam', 'delta_sigma8']
        res_1 = mt.kde_parameter_shift(self.diff_chain,
                                       param_names=param_names,
                                       method='brute_force',
                                       scale=0.5)
        res_2 = mt.kde_parameter_shift(self.diff_chain,
                                       param_names=param_names,
                                       method='neighbor_elimination',
                                       scale=0.5)
        print(res_1, res_2)
        assert np.allclose(res_1, res_2)

    def test_band(self):
        """Test KDE bandwidth selectors."""
        n, d = self.diff_chain.samples.shape
        weights = self.diff_chain.weights
        wtot = np.sum(weights)
        neff = wtot**2 / np.sum(weights**2)
        mt.Scotts_bandwidth(d, neff)
        mt.AMISE_bandwidth(d, neff)
        mt.MAX_bandwidth(d, neff)
        mt.MISE_bandwidth_1d(d, neff)
        mt.MISE_bandwidth(d, neff)
        white_samples = stut.whiten_samples(self.diff_chain.samples, weights)
        mt.UCV_bandwidth(weights, white_samples, mode='1d', feedback=1)
        mt.UCV_SP_bandwidth(white_samples, weights, near=1, near_max=20, feedback=1)

    def test_fft_shift(self):
        """Test FFT KDE shift helpers."""
        param_names = ['delta_sigma8']
        mt.kde_parameter_shift_1D_fft(self.diff_chain, param_names=param_names, feedback=2)
        param_names = ['delta_omegam', 'delta_sigma8']
        mt.kde_parameter_shift_2D_fft(self.diff_chain, param_names=param_names, feedback=2)

    def test_ball_kde(self):
        """Test BALL and ELL KDE distance scales."""
        res_1 = mt.kde_parameter_shift(self.diff_chain,
                                       method='brute_force',
                                       feedback=2,
                                       scale='BALL')
        res_2 = mt.kde_parameter_shift(self.diff_chain,
                                       method='neighbor_elimination',
                                       feedback=2,
                                       scale='BALL')
        assert np.allclose(res_1, res_2)

        res_1 = mt.kde_parameter_shift(self.diff_chain,
                                       method='brute_force',
                                       feedback=2,
                                       scale='ELL')
        res_2 = mt.kde_parameter_shift(self.diff_chain,
                                       method='neighbor_elimination',
                                       feedback=2,
                                       scale='ELL')
        assert np.allclose(res_1, res_2)

#########################################################################################################
# Helper utilities


def make_chain_additional(dim=2, n=4):
    """Build a small MCSamples chain with simple names and weights.

    :param dim: number of parameters.
    :param n: number of samples.
    :returns: chain with ``n`` samples and ``dim`` parameters.
    """
    rng = np.linspace(0, 1, n * dim).reshape(n, dim)
    samples = rng + np.eye(n, dim, k=0) * 0.05
    weights = np.ones(n)
    names = [f"p{i}" for i in range(dim)]
    return MCSamples(samples=samples, weights=weights, names=names,
                     labels=names, sampler="uncorrelated", ignore_rows=0)


def make_chain_branch(dim=2, n=4):
    """Build a deterministic MCSamples chain.

    :param dim: number of parameters.
    :param n: number of samples.
    :returns: chain with evenly spaced samples.
    """
    data = np.linspace(0, 1, n * dim).reshape(n, dim)
    weights = np.ones(n)
    names = [f"p{i}" for i in range(dim)]
    chain = MCSamples(samples=data, weights=weights, names=names, labels=names,
                      sampler="uncorrelated", ignore_rows=0)
    return chain


class DummyParallel:
    """Replacement for joblib.Parallel that runs sequentially."""

    def __init__(self, *args, **kwargs):
        """Init."""
        pass

    def __enter__(self):
        """Enter."""
        return self

    def __exit__(self, *args):
        """Exit."""
        return False

    def __call__(self, tasks):
        """Call."""
        return np.array([t() if callable(t) else t for t in tasks])

#########################################################################################################
# Additional KDE tests


class TestKdeAdditional(unittest.TestCase):
    """KDE additional test suite."""
    def setUp(self):
        """Set up test fixtures."""
        # Disable numba to keep helpers in pure Python and reload module accordingly.
        self._orig_disable_jit = os.environ.get("NUMBA_DISABLE_JIT")
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        globals()["mt"] = importlib.reload(mt)
        mt.n_threads = 1
        self._orig_logsumexp = mt.scipy.special.logsumexp
        mt.scipy.special.logsumexp = lambda a, b=None, axis=None, return_sign=False, xp=None: np.log(
            np.sum(np.exp(a)*(1 if b is None else b))
        )
        # Replace numba.jit with a no-op to avoid generating Dispatchers at runtime.
        self._orig_jit = mt.jit
        mt.jit = lambda *args, **kwargs: (lambda f: f)
        # Force numba-jitted helpers to run in pure Python to avoid compile errors.
        self._orig_mise1d = mt._mise1d_optimizer
        self._orig_mise1d_jac = mt._mise1d_optimizer_jac
        self._orig_mise = mt._mise_optimizer
        self._orig_ell_helper = mt._ell_helper
        if hasattr(mt._mise1d_optimizer, "py_func"):
            mt._mise1d_optimizer = mt._mise1d_optimizer.py_func
        if hasattr(mt._mise1d_optimizer_jac, "py_func"):
            mt._mise1d_optimizer_jac = mt._mise1d_optimizer_jac.py_func
        if hasattr(mt._mise_optimizer, "py_func"):
            mt._mise_optimizer = mt._mise_optimizer.py_func
        if hasattr(mt._ell_helper, "py_func"):
            mt._ell_helper = mt._ell_helper.py_func

    def tearDown(self):
        """Clean up test fixtures."""
        mt.scipy.special.logsumexp = self._orig_logsumexp
        mt._mise1d_optimizer = self._orig_mise1d
        mt._mise1d_optimizer_jac = self._orig_mise1d_jac
        mt._mise_optimizer = self._orig_mise
        mt._ell_helper = self._orig_ell_helper
        mt.jit = self._orig_jit
        if self._orig_disable_jit is None:
            os.environ.pop("NUMBA_DISABLE_JIT", None)
        else:
            os.environ["NUMBA_DISABLE_JIT"] = self._orig_disable_jit
        globals()["mt"] = importlib.reload(mt)

    def test_force_disable_jit_pop(self):
        """Force NUMBA_DISABLE_JIT cleanup path."""
        self._orig_disable_jit = None
        os.environ["NUMBA_DISABLE_JIT"] = "1"

    def test_import_branches_and_env(self):
        """Test Import branches and env."""
        # Force ImportError for simpson to hit fallback and set env for threads.
        path = os.path.join(os.path.dirname(mt.__file__), "kde.py")
        real_import = builtins.__import__
        seen = {"first": True}

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Fake import."""
            if name == "scipy.integrate" and "simpson" in fromlist:
                if seen["first"]:
                    seen["first"] = False
                    raise ImportError("force fallback")
            if name == "scipy.integrate" and "simps" in fromlist:
                dummy = types.SimpleNamespace(simps=lambda x: x)
                return dummy
            return real_import(name, globals, locals, fromlist, level)

        with patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}):
            with patch("builtins.__import__", side_effect=fake_import):
                importlib.reload(mt)
                self.assertEqual(mt.n_threads, 2)
        importlib.reload(mt)

    def test_bandwidth_helpers(self):
        """Test Bandwidth helpers."""
        mt._mise1d_optimizer = types.SimpleNamespace(py_func=mt._mise1d_optimizer)
        mt._mise1d_optimizer_jac = types.SimpleNamespace(py_func=mt._mise1d_optimizer_jac)
        mt._mise_optimizer = types.SimpleNamespace(py_func=mt._mise_optimizer)
        # Directly exercise numba helpers and MISE options.
        # use the python implementations to avoid JIT compilation issues
        if hasattr(mt._mise1d_optimizer, "py_func"):
            mt._mise1d_optimizer = mt._mise1d_optimizer.py_func
        if hasattr(mt._mise1d_optimizer_jac, "py_func"):
            mt._mise1d_optimizer_jac = mt._mise1d_optimizer_jac.py_func
        if hasattr(mt._mise_optimizer, "py_func"):
            mt._mise_optimizer = mt._mise_optimizer.py_func
        mt._mise1d_optimizer(0.5, 1, 10)
        mt._mise1d_optimizer_jac(0.5, 1, 10)
        mt._mise_optimizer(np.eye(1), 1, 5)
        res = mt.MISE_bandwidth_1d(1, 5, alpha0=0.2)
        self.assertTrue(np.all(np.isfinite(res)))
        res2 = mt.MISE_bandwidth(2, 6, alpha0=np.eye(2) * 0.3, feedback=3)
        self.assertEqual(res2.shape, (2, 2))

    def test_ucv_modes(self):
        """Test Ucv modes."""
        samples = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
        weights = np.ones(3)
        res1 = mt.UCV_bandwidth(weights, samples, mode="1d", n_nearest=3)
        self.assertEqual(res1.shape, (2, 2))
        res2 = mt.UCV_bandwidth(weights, samples, mode="diag", n_nearest=3)
        self.assertEqual(res2.shape, (2, 2))
        res3 = mt.UCV_bandwidth(weights, samples, mode="full", n_nearest=3,
                                bounds=np.array([[0.001, 10], [0.001, 10], [0.001, 10]]))
        self.assertEqual(res3.shape, (2, 2))
        res_sp = mt.UCV_SP_bandwidth(samples, weights, near=1, near_max=3)
        self.assertTrue(np.isfinite(res_sp.x))

    def test_mise_bandwidth_default_bounds(self):
        """Test MISE_bandwidth default bounds path."""
        opt = types.SimpleNamespace(success=True, x=mt.stutils.PDM_to_vector(np.eye(2)))
        with patch.object(mt.scipy.optimize, "minimize", return_value=opt):
            res = mt.MISE_bandwidth(2, 5, feedback=0)
        self.assertEqual(res.shape, (2, 2))

    def test_ucv_bandwidth_default_alpha0_1d(self):
        """Test UCV_bandwidth default alpha0 in 1D mode."""
        samples = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
        weights = np.ones(3)
        opt = types.SimpleNamespace(success=True, x=np.array([0.1]))
        with patch.object(mt.scipy.optimize, "minimize", return_value=opt):
            res = mt.UCV_bandwidth(weights, samples, mode="1d", feedback=0)
        self.assertEqual(res.shape, (2, 2))

    def test_ucv_bandwidth_full_feedback(self):
        """Test UCV_bandwidth full mode feedback."""
        samples = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
        weights = np.ones(3)
        opt = types.SimpleNamespace(success=True, x=mt.stutils.PDM_to_vector(np.eye(2)))
        with patch.object(mt.scipy.optimize, "minimize", return_value=opt):
            res = mt.UCV_bandwidth(weights, samples, mode="full", feedback=3)
        self.assertEqual(res.shape, (2, 2))

    def test_brute_force_and_neighbor_helpers(self):
        """Test Brute force and neighbor helpers."""
        white_samples = np.array([[0.0, 0.0], [0.1, 0.1]])
        weights = np.array([1.0, 1.0])
        with patch("tensiometer.mcmc_tension.kde.joblib.Parallel", DummyParallel), \
                patch("tensiometer.mcmc_tension.kde.joblib.delayed",
                      lambda f: (lambda *a, **k: lambda: f(*a, **k))):
            mt._brute_force_kde_param_shift(white_samples, weights, -10, 2, 0)
            # BALL variant
            mt._brute_force_kde_param_shift(white_samples, weights, -10, 2, 0,
                                            distance_weights=np.ones(2))
            # ELL variant
            ell = np.array([np.eye(2), np.eye(2)])
            mt._brute_force_kde_param_shift(white_samples, weights, -10, 2, 0,
                                            distance_weights=ell, weights_norm=weights)
            mt._neighbor_parameter_shift(white_samples, weights, -10, 2, 0,
                                         stable_cycle=1, smallest_improvement=0,
                                         chunk_size=2)

    def test_kde_parameter_shift_scales_and_errors(self):
        """Test KDE parameter shift scales and errors."""
        chain = make_chain_additional()
        bad_scale = np.ones((3, 3))
        with patch("tensiometer.mcmc_tension.kde.joblib.Parallel", DummyParallel), \
                patch("tensiometer.mcmc_tension.kde.joblib.delayed",
                      lambda f: (lambda *a, **k: lambda: f(*a, **k))):
            for scale in ["MISE", "AMISE", "MAX", 0.2]:
                mt.kde_parameter_shift(chain, scale=scale, feedback=0, num_samples=2)
            mt.kde_parameter_shift(chain, scale="BALL", feedback=0, near=1)
            mt.kde_parameter_shift(chain, scale="ELL", feedback=0)
            with self.assertRaises(ValueError):
                mt.kde_parameter_shift(chain, scale=bad_scale)
            with self.assertRaises(ValueError):
                mt.kde_parameter_shift(chain, method="unknown")
            # zero_prob and num_samples override branches
            mt.kde_parameter_shift(chain, scale=0.1, feedback=0,
                                   zero_prob=-1.0, num_samples=1)

    def test_kde_param_names_and_scale_errors(self):
        """Test parameter name and scale validation errors."""
        chain = make_chain_additional()
        with self.assertRaises(ValueError):
            mt.kde_parameter_shift(chain, param_names=["missing"])
        with self.assertRaises(ValueError):
            mt.kde_parameter_shift(chain, scale={"bad": 1})

    def test_kde_scale_matrix_feedback(self):
        """Test non-diagonal scale feedback in KDE."""
        chain = make_chain_additional()
        scale = np.array([[1.0, 0.2], [0.2, 1.0]])
        with patch.object(mt, "_neighbor_parameter_shift", return_value=0):
            mt.kde_parameter_shift(chain, scale=scale, feedback=1)

    def test_kde_ell_scale_feedback(self):
        """Test ELL scale feedback and setup."""
        chain = make_chain_additional()
        dets = np.ones(chain.samples.shape[0])
        mats = [np.eye(chain.samples.shape[1]) for _ in range(chain.samples.shape[0])]
        with patch.object(mt, "_ell_helper", return_value=(dets, mats)), \
                patch.object(mt, "_neighbor_parameter_shift", return_value=0):
            mt.kde_parameter_shift(chain, scale="ELL", feedback=1)

    def test_fft_and_optimize_bandwidth_errors(self):
        """Test FFT and optimize bandwidth errors."""
        chain = make_chain_additional(dim=1)
        # wrong parameter count for 1D and 2D FFT
        with self.assertRaises(ValueError):
            mt.kde_parameter_shift_1D_fft(chain, param_names=["p0", "extra"])
        with self.assertRaises(ValueError):
            mt.kde_parameter_shift_2D_fft(chain, param_names=["p0"])
        # prior mismatch
        prior = make_chain_additional(dim=1)
        with self.assertRaises(ValueError):
            mt.kde_parameter_shift_1D_fft(chain, prior_diff_chain=prior, param_names=["missing"])
        # valid 1D call
        mt.kde_parameter_shift_1D_fft(chain, feedback=0)
        # Optimize bandwidth bad parameter
        with self.assertRaises(ValueError):
            mt.OptimizeBandwidth_1D(chain, param_names=["missing"])
        mt.OptimizeBandwidth_1D(chain)

    def test_fft_prior_checks_and_probability_path(self):
        """Test FFT prior checks and probability path."""
        # 2D FFT path with prior and no samples above zero
        chain = make_chain_additional(dim=2)
        prior = make_chain_additional(dim=2)
        prior.weights *= 0.5
        mt.kde_parameter_shift_2D_fft(chain, prior_diff_chain=prior, feedback=0, nbins=16)

#########################################################################################################
# Branch coverage tests


class TestKdeBranchCoverage(unittest.TestCase):
    """KDE branch coverage test suite."""
    def setUp(self):
        """Set up test fixtures."""
        # Disable numba so jitted helpers execute in Python and count for coverage.
        self._orig_numba = os.environ.get("NUMBA_DISABLE_JIT")
        os.environ["NUMBA_DISABLE_JIT"] = "1"
        self.mt = importlib.reload(mt)
        # Make jit a no-op so inner helpers stay as plain Python functions.
        self._orig_jit = self.mt.jit
        self.mt.jit = lambda *args, **kwargs: (lambda f: f)
        # Keep python implementations of jitted helpers to avoid compile errors.
        self._orig_mise1d = self.mt._mise1d_optimizer
        self._orig_mise1d_jac = self.mt._mise1d_optimizer_jac
        self._orig_mise = self.mt._mise_optimizer
        self._orig_ell_helper = self.mt._ell_helper
        self._orig_ucv_brute = self.mt._UCV_optimizer_brute_force
        if hasattr(self.mt._mise1d_optimizer, "py_func"):
            self.mt._mise1d_optimizer = self.mt._mise1d_optimizer.py_func
        if hasattr(self.mt._mise1d_optimizer_jac, "py_func"):
            self.mt._mise1d_optimizer_jac = self.mt._mise1d_optimizer_jac.py_func
        if hasattr(self.mt._mise_optimizer, "py_func"):
            self.mt._mise_optimizer = self.mt._mise_optimizer.py_func
        if hasattr(self.mt._ell_helper, "py_func"):
            self.mt._ell_helper = self.mt._ell_helper.py_func
        if hasattr(self.mt._UCV_optimizer_brute_force, "py_func"):
            self.mt._UCV_optimizer_brute_force = self.mt._UCV_optimizer_brute_force.py_func
        # Patch logsumexp to avoid SciPy array-api quirks in this environment.
        self._orig_logsumexp = self.mt.scipy.special.logsumexp
        self.mt.scipy.special.logsumexp = lambda a, b=None, axis=None, return_sign=False, xp=None: np.log(
            np.sum(np.exp(a)*(1 if b is None else b))
        )

    def tearDown(self):
        """Clean up test fixtures."""
        if self._orig_numba is None:
            os.environ.pop("NUMBA_DISABLE_JIT", None)
        else:
            os.environ["NUMBA_DISABLE_JIT"] = self._orig_numba
        importlib.reload(mt)
        mt.scipy.special.logsumexp = self._orig_logsumexp
        mt.jit = self._orig_jit
        mt._mise1d_optimizer = self._orig_mise1d
        mt._mise1d_optimizer_jac = self._orig_mise1d_jac
        mt._mise_optimizer = self._orig_mise
        mt._ell_helper = self._orig_ell_helper
        mt._UCV_optimizer_brute_force = self._orig_ucv_brute

    def test_numba_disable_env_cleanup(self):
        """Exercise NUMBA_DISABLE_JIT cleanup branch."""
        self._orig_numba = None
        os.environ["NUMBA_DISABLE_JIT"] = "1"

    def test_numba_disable_env_restore(self):
        """Exercise NUMBA_DISABLE_JIT restore branch."""
        self._orig_numba = "0"
        os.environ["NUMBA_DISABLE_JIT"] = "1"

    def test_numba_helpers_and_mise_failures(self):
        """Test Numba helpers and mise failures."""
        # Directly exercise the jitted helpers via Python execution.
        self.mt._mise1d_optimizer(0.5, 1, 10)
        self.mt._mise1d_optimizer_jac(0.5, 1, 10)
        self.mt._mise_optimizer(np.eye(1), 1, 5)
        weights = np.array([1.0, 1.0])
        samples = np.array([[0.0], [0.1]])
        self.mt._UCV_optimizer_brute_force(np.eye(1), weights, samples)
        self.mt._ell_helper(np.array([[0, 1]]), np.array([[[0.0]], [[0.1]]])[:, :, 0], 1)
        # Force optimizer failures to hit print-feedback branches.
        bad_opt = types.SimpleNamespace(success=False, x=np.array([0.1]))
        bad_opt2 = types.SimpleNamespace(success=False, x=np.array([0.1, 0.1, 0.1]))
        with patch.object(self.mt.scipy.optimize, "minimize", return_value=bad_opt):
            self.mt.MISE_bandwidth_1d(1, 5, alpha0=0.2)
        with patch.object(self.mt.scipy.optimize, "minimize", return_value=bad_opt2):
            self.mt.MISE_bandwidth(2, 6)

    def test_ucv_bandwidth_modes_and_feedback(self):
        """Test Ucv bandwidth modes and feedback."""
        weights = np.ones(3)
        samples = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
        # Make optimizer fail to trigger logging and default bounds handling.
        opt_fail = types.SimpleNamespace(success=False, x=np.array([0.2]))
        opt_fail_diag = types.SimpleNamespace(success=False, x=np.array([0.2, 0.2]))
        opt_fail_full = types.SimpleNamespace(success=False, x=np.array([0.2, 0.2, 0.2]))
        with patch.object(self.mt.scipy.optimize, "minimize", side_effect=[opt_fail, opt_fail_diag, opt_fail_full]):
            self.mt.UCV_bandwidth(weights, samples, mode="1d", n_nearest=3, feedback=1)
            self.mt.UCV_bandwidth(weights, samples, mode="diag", n_nearest=3, feedback=1)
            self.mt.UCV_bandwidth(weights, samples, mode="full", n_nearest=3, feedback=3)

    def test_mise_bandwidth_with_bounds(self):
        """Test MISE bandwidth with explicit bounds."""
        opt = types.SimpleNamespace(success=True, x=self.mt.stutils.PDM_to_vector(np.eye(2)))
        bounds = np.array([[0.1, 1.0], [0.1, 1.0], [0.1, 1.0]])
        with patch.object(self.mt.scipy.optimize, "minimize", return_value=opt):
            res = self.mt.MISE_bandwidth(2, 5, bounds=bounds, feedback=0)
        self.assertEqual(res.shape, (2, 2))

    def test_ucv_bandwidth_alpha0_supplied(self):
        """Test UCV bandwidth with explicit alpha0."""
        weights = np.ones(3)
        samples = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
        opt = types.SimpleNamespace(success=True, x=np.array([0.2, 0.2]))
        with patch.object(self.mt.scipy.optimize, "minimize", return_value=opt):
            res = self.mt.UCV_bandwidth(weights, samples, mode="diag", alpha0=np.eye(2), feedback=0)
        self.assertEqual(res.shape, (2, 2))

    def test_ucv_bandwidth_invalid_mode(self):
        """Test UCV bandwidth invalid mode error."""
        weights = np.ones(3)
        samples = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
        with self.assertRaises(ValueError):
            self.mt.UCV_bandwidth(weights, samples, mode="bad")

    def test_optimize_bandwidth_with_param_names(self):
        """Test OptimizeBandwidth_1D with explicit parameter names."""
        chain = make_chain_additional()
        res = self.mt.OptimizeBandwidth_1D(chain, param_names=["p0"])
        self.assertEqual(res.shape, (1,))

    def test_fft_numeric_scale_branches(self):
        """Test FFT branches with numeric scale inputs."""
        chain = make_chain_branch(dim=1)

        class FakeDensity1D:
            """Fake Density1 D test suite."""
            def __init__(self):
                """Init."""
                self.P = np.array([1.0, 0.5])
                self.x = np.array([0.0, 1.0])

            def _initSpline(self):  # pragma: no cover - trivial
                """Init Spline."""
                return None

            def Prob(self, arr):
                """Prob."""
                arr = np.atleast_1d(arr)
                return np.full_like(arr, 0.5, dtype=float)

        with patch.object(chain, "get1DDensity", return_value=FakeDensity1D()):
            self.mt.kde_parameter_shift_1D_fft(chain, param_names=["p0"], feedback=0, nbins=8, scale=0.5)

        chain2d = make_chain_branch(dim=2)

        class DummySpl:
            """Dummy Spl test suite."""
            def __call__(self, xs, ys):
                """Call."""
                return np.array([[0.1]])

            def ev(self, x, y):
                """Ev."""
                return np.zeros_like(x, dtype=float)

        class FakeDensity2D:
            """Fake Density2 D test suite."""
            def __init__(self):
                """Init."""
                self.P = np.full((2, 2), 0.1)
                self.x = np.array([0.0, 1.0])
                self.y = np.array([0.0, 1.0])
                self.spl = DummySpl()

            def _initSpline(self):  # pragma: no cover - trivial
                """Init Spline."""
                return None

        with patch.object(chain2d, "get2DDensity", return_value=FakeDensity2D()):
            self.mt.kde_parameter_shift_2D_fft(chain2d, feedback=0, nbins=8, scale=0.5)

    def test_kde_ell_feedback_and_logprob(self):
        """Test ELL scale feedback and log-probability setup."""
        chain = make_chain_additional()
        dets = np.ones(chain.samples.shape[0])
        mats = [np.eye(chain.samples.shape[1]) for _ in range(chain.samples.shape[0])]
        def _fake_logpdf(point, samples, weights, distance_weights):
            """Fake logpdf."""
            return -1.0
        with patch.object(self.mt, "_ell_helper", return_value=(dets, mats)), \
                patch.object(self.mt, "_neighbor_parameter_shift", return_value=0), \
                patch.object(self.mt, "_gauss_ellkde_logpdf", side_effect=_fake_logpdf):
            res = self.mt.kde_parameter_shift(chain, scale="ELL", feedback=1)
        self.assertTrue(np.isfinite(res[0]))

    def test_neighbor_elimination_feedback_noise(self):
        """Test Neighbor elimination feedback noise."""
        white_samples = np.array([[0.0, 0.0], [0.05, 0.05], [0.1, 0.1], [0.15, 0.15]])
        weights = np.ones(4)
        # Run with verbose feedback and small chunk to hit logging branches.
        with patch("tensiometer.mcmc_tension.kde.joblib.Parallel", DummyParallel), \
                patch("tensiometer.mcmc_tension.kde.joblib.delayed",
                      lambda f: (lambda *a, **k: lambda: f(*a, **k))):
            self.mt._neighbor_parameter_shift(white_samples, weights, -10, 4, feedback=3,
                                              stable_cycle=1, chunk_size=2, smallest_improvement=0.0)

    def test_fft_branches_with_fake_densities(self):
        """Test FFT branches with fake densities."""
        chain = make_chain_branch(dim=1)

        class FakeDensity1D:
            """Fake Density1 D test suite."""
            def __init__(self):
                """Init."""
                self.P = np.array([1.0, 0.5])
                self.x = np.array([0.0, 1.0])

            def _initSpline(self):  # pragma: no cover - trivial
                """Init Spline."""
                return None

            def Prob(self, arr):
                """Prob."""
                arr = np.atleast_1d(arr)
                return np.full_like(arr, 0.5, dtype=float)

        # No prior, integral branch
        with patch.object(chain, "get1DDensity", return_value=FakeDensity1D()):
            self.mt.kde_parameter_shift_1D_fft(chain, param_names=["p0"], feedback=0, nbins=8)
            self.mt.kde_parameter_shift_1D_fft(chain, param_names=["p0"], feedback=0, nbins=8, scale="MISE")
        # With prior, same integral path
        prior = make_chain_branch(dim=1)
        with patch.object(chain, "get1DDensity", return_value=FakeDensity1D()), \
                patch.object(prior, "get1DDensity", return_value=FakeDensity1D()):
            self.mt.kde_parameter_shift_1D_fft(chain, prior_diff_chain=prior, param_names=["p0"], feedback=0, nbins=8)

        # 2D fake densities to trigger no-sample branches and prior handling
        chain2d = make_chain_branch(dim=2)
        prior2d = make_chain_branch(dim=2)

        class DummySpl:
            """Dummy Spl test suite."""
            def __call__(self, xs, ys):
                """Call."""
                return np.array([[0.1]])

            def ev(self, x, y):
                """Ev."""
                return np.zeros_like(x, dtype=float)

        class FakeDensity2D:
            """Fake Density2 D test suite."""
            def __init__(self):
                """Init."""
                self.P = np.full((2, 2), 0.1)
                self.x = np.array([0.0, 1.0])
                self.y = np.array([0.0, 1.0])
                self.spl = DummySpl()

            def _initSpline(self):  # pragma: no cover - trivial
                """Init Spline."""
                return None

        with patch.object(chain2d, "get2DDensity", return_value=FakeDensity2D()):
            self.mt.kde_parameter_shift_2D_fft(chain2d, feedback=0, nbins=8)
            self.mt.kde_parameter_shift_2D_fft(chain2d, feedback=0, nbins=8, scale="MISE")
        with patch.object(chain2d, "get2DDensity", return_value=FakeDensity2D()), \
                patch.object(prior2d, "get2DDensity", return_value=FakeDensity2D()):
            self.mt.kde_parameter_shift_2D_fft(chain2d, prior_diff_chain=prior2d, feedback=0, nbins=8)
        with self.assertRaises(ValueError):
            self.mt.kde_parameter_shift_2D_fft(chain2d, param_names=["missing"])
        with self.assertRaises(ValueError):
            self.mt.kde_parameter_shift_2D_fft(chain2d, prior_diff_chain=prior, param_names=["p0", "p1"])

    def test_fft_param_validation_branches(self):
        """Test FFT parameter validation branches."""
        chain = make_chain_branch(dim=1)
        prior = make_chain_branch(dim=1)
        prior_mismatch = MCSamples(samples=prior.samples, weights=prior.weights,
                                   names=["q0"], labels=["q0"], sampler="uncorrelated", ignore_rows=0)
        with self.assertRaises(ValueError):
            self.mt.kde_parameter_shift_1D_fft(chain, prior_diff_chain=prior_mismatch, param_names=["p0"])
        chain2d = make_chain_branch(dim=2)
        with self.assertRaises(ValueError):
            self.mt.kde_parameter_shift_1D_fft(chain2d, param_names=["p0", "p1"])

    def test_fft_prior_zero_density_branch(self):
        """Test FFT branch when prior density is zero at origin."""
        chain2d = make_chain_branch(dim=2)
        prior2d = make_chain_branch(dim=2)

        class DummySpl:
            """Dummy spline with zero prior."""
            def __call__(self, xs, ys):
                """Call."""
                return np.array([[0.0]])

            def ev(self, x, y):
                """Evaluate."""
                return np.zeros_like(x, dtype=float)

        class FakeDensity2D:
            """Fake 2D density for zero-prior branch."""
            def __init__(self, zero=False):
                """Init."""
                self.P = np.full((2, 2), 0.1)
                self.x = np.array([0.0, 1.0])
                self.y = np.array([0.0, 1.0])
                self.spl = DummySpl() if zero else DummySpl()

            def _initSpline(self):  # pragma: no cover - trivial
                """Init spline."""
                return None

        with patch.object(chain2d, "get2DDensity", return_value=FakeDensity2D(zero=False)), \
                patch.object(prior2d, "get2DDensity", return_value=FakeDensity2D(zero=True)):
            res = self.mt.kde_parameter_shift_2D_fft(chain2d, prior_diff_chain=prior2d, feedback=0, nbins=8)
            self.assertEqual(res, (1.0, None, None))

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
