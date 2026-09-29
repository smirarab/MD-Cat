"""Compare vectorized kernels and seeded fits against frozen scalar equations."""
import contextlib
import io
import random
import unittest
from unittest.mock import patch

import cvxpy as cp
import numpy as np
from treeswift import read_tree_newick
from simulator.multinomial import multinomial
from emd import emd_normal_lib as emd
import vectorization_reference as scalar


class VectorizationTest(unittest.TestCase):
    def inputs(self, n=37, k=7):
        rng = np.random.default_rng(729)
        b = rng.uniform(.001, .8, n).tolist()
        tau = rng.uniform(.01, 2, n).tolist()
        omega = rng.uniform(.001, 1, k).tolist()
        phi = rng.dirichlet(np.ones(k)).tolist()
        return b, tau, omega, phi

    def test_posteriors_and_likelihood(self):
        for n, k in ((0, 3), (1, 1), (37, 7)):
            for approximate in (False, True):
                for missing in (False, True):
                    for block in (11, 262144):
                        with self.subTest(n=n, k=k, approximate=approximate,
                                          missing=missing, block=block):
                            b, tau, omega, phi = self.inputs(n, k)
                            if missing:
                                for i in range(0, n, 3):
                                    b[i] = tau[i] = None
                            with patch.object(emd, '_EM_BLOCK_ELEMENTS', block):
                                actual = emd.run_Estep(b, 1000, omega, tau, phi,
                                                      var_apprx=approximate)
                                expected = scalar.run_Estep(b, 1000, omega, tau, phi,
                                                            var_apprx=approximate)
                                self.assertIsInstance(actual, list)
                                for a, e in zip(actual, expected):
                                    if e is None:
                                        self.assertIsNone(a)
                                    else:
                                        self.assertIsInstance(a, list)
                                        np.testing.assert_allclose(a, e, rtol=2e-12, atol=2e-14)
                                        self.assertAlmostEqual(sum(a), 1.)
                                np.testing.assert_allclose(
                                    emd.f_ll(b, 1000, tau, omega, phi, var_apprx=approximate),
                                    scalar.f_ll(b, 1000, tau, omega, phi, var_apprx=approximate),
                                    rtol=2e-14, atol=1e-10)

    def test_extreme_log_densities_and_probability_floor(self):
        b = [1e-8, .1, 100.]
        tau = [10., 1e-4, .01]
        omega = [1e-4, .1, 1., 100.]
        phi = [1e-250, .2, .3, .5]
        for approximate in (True, False):
            a = emd.run_Estep(b, 1e6, omega, tau, phi, var_apprx=approximate)
            e = scalar.run_Estep(b, 1e6, omega, tau, phi, var_apprx=approximate)
            np.testing.assert_allclose(a, e, rtol=2e-12, atol=1e-14)
            self.assertTrue(np.isfinite(a).all())
            self.assertGreater(np.min(a), 0)
            np.testing.assert_allclose(emd.f_ll(b, 1e6, tau, omega, phi, approximate),
                                       scalar.f_ll(b, 1e6, tau, omega, phi, approximate), rtol=2e-14)

    def test_scalar_rounding_with_solver_array_durations(self):
        # Solver outputs are ndarrays. Reassociation, np.square and a different
        # softmax reduction can change fitted trajectories on real calibrations.
        b, tau, omega, phi = self.inputs(113, 50)
        tau = np.asarray(tau)
        expected_q = scalar.run_Estep(b, 1000, omega, tau, phi)
        for block in (111, 262144):
            with patch.object(emd, '_EM_BLOCK_ELEMENTS', block):
                np.testing.assert_array_equal(
                    emd.run_Estep(b, 1000, omega, tau, phi), expected_q)
                self.assertEqual(emd.f_ll(b, 1000, tau, omega, phi),
                                 scalar.f_ll(b, 1000, tau, omega, phi))
                np.testing.assert_array_equal(
                    emd.compute_omega_star(tau, expected_q, b, phi),
                    scalar.compute_omega_star(tau, expected_q, b, phi))

    def test_rates_including_active_bounds_and_mean_constraint(self):
        b, tau, omega, phi = self.inputs()
        q = emd.run_Estep(b, 1000, omega, tau, phi)
        for lower, mean in ((1e-4, None), (.2, None), (.2, .25), (1e-4, .5)):
            for block in (11, 262144):
                with self.subTest(lower=lower, mean=mean, block=block), \
                        patch.object(emd, '_EM_BLOCK_ELEMENTS', block):
                    a = emd.compute_omega_star(tau, q, b, phi, eps_omg=lower, mu_avg=mean)
                    e = scalar.compute_omega_star(tau, q, b, phi, eps_omg=lower, mu_avg=mean)
                    np.testing.assert_allclose(a, e, rtol=2e-13, atol=1e-14)
                    self.assertGreaterEqual(min(a), lower-1e-14)
                    if mean is not None:
                        self.assertAlmostEqual(np.dot(a, phi), mean, places=13)

    def test_duration_objective_coefficients(self):
        b, tau, omega, phi = self.inputs()
        q = emd.run_Estep(b, 1000, omega, tau, phi)
        b[0] = tau[0] = q[0] = None
        captured = []
        def capture(problem, variable, *args, **kwargs):
            data = problem.get_problem_data(cp.OSQP)[0]
            captured.append((data['P'].toarray(), data['q']))
            return np.ones(len(b)), None
        for approximate in (True, False):
            for block in (11, 262144):
                with patch.object(emd, '_EM_BLOCK_ELEMENTS', block), \
                        patch.object(emd, '_solve_durations', capture), \
                        patch.object(scalar, '_solve_durations', capture, create=True):
                    for module in (scalar, emd):
                        module.compute_tau_star_cvxpy(tau, omega, q, b, 1000,
                            np.ones((1, len(b))), [1.], var_apprx=approximate)
                for a, e in zip(captured[-1], captured[-2]):
                    np.testing.assert_array_equal(a, e)

    def test_seeded_em_fit_against_scalar(self):
        solve = cp.Problem.solve
        def osqp(problem, **kwargs):
            return solve(problem, solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9)
        results = []
        for reference in (True, False):
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(cp.Problem, 'solve', osqp))
                if reference:
                    for name in ('run_Estep', 'f_ll', 'compute_omega_star', 'compute_tau_star_cvxpy'):
                        stack.enter_context(patch.object(emd, name, getattr(scalar, name)))
                random.seed(2)
                results.append(emd.EM_date(
                    read_tree_newick('((A:.1,B:.2)X:.1,C:.3)R;'),
                    {'R': 0., 'A': 1., 'B': 1., 'C': 1.},
                    multinomial([.1, .2, .3], [1/3]*3), maxIter=100, threads=1)[0])
        for name in ('tau', 'omega', 'Q', 'llh'):
            np.testing.assert_allclose(results[0][name], results[1][name], rtol=1e-7, atol=1e-8)
