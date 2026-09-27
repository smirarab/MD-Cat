"""EM must return one consistent, constrained iterate when it stops."""
import contextlib
import io
import random
import unittest
from unittest.mock import patch

import cvxpy as cp
import numpy as np
from treeswift import read_tree_newick
from emd import emd_normal_lib as emd
from simulator.multinomial import multinomial


class EMConsistencyTest(unittest.TestCase):
    def test_likelihood_posteriors_and_constraints_at_each_exit(self):
        original_solve = cp.Problem.solve
        def solve(problem, **kwargs):
            return original_solve(problem, solver=cp.OSQP, verbose=False,
                                  eps_abs=1e-9, eps_rel=1e-9)
        # Force first-update convergence, allow normal convergence, and hit
        # the iteration limit. The random initializer is not calibrated.
        for tolerance, iterations in ((float('inf'), 100), (5e-4, 100), (0., 1)):
            with self.subTest(tolerance=tolerance, iterations=iterations), \
                    patch.object(cp.Problem, 'solve', solve), \
                    contextlib.redirect_stdout(io.StringIO()):
                random.seed(2)
                with patch.object(emd, 'run_Mstep', wraps=emd.run_Mstep) as step:
                    result, constraints = emd.EM_date(
                        read_tree_newick('((A:.1,B:.2)X:.1,C:.3)R;'),
                        {'R': 0., 'A': 1., 'B': 1., 'C': 1.},
                        multinomial([.1, .2, .3], [1/3]*3),
                        df=tolerance, maxIter=iterations, threads=1)
                if tolerance == float('inf') or iterations == 1:
                    self.assertEqual(step.call_count, 1)
                else:
                    self.assertLess(step.call_count, iterations)
                likelihood = emd.f_ll(constraints['b'], 1000, result['tau'],
                                      result['omega'], result['phi'], var_apprx=True)
                self.assertEqual(result['llh'], likelihood)
                posterior = emd.run_Estep(constraints['b'], 1000, result['omega'],
                                         result['tau'], result['phi'], var_apprx=True)
                np.testing.assert_array_equal(result['Q'], posterior)
                np.testing.assert_allclose(np.asarray(constraints['M']) @ result['tau'],
                                           constraints['dt'], atol=1e-7, rtol=0)
                self.assertGreaterEqual(min(result['tau']), emd.EPS_tau-1e-7)
