"""CI solver fallback and failure regression tests."""
import contextlib
import io
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cvxpy as cp
import numpy as np
from treeswift import read_tree_newick
from emd import emd_normal_lib as emd


class ConfidenceIntervalTest(unittest.TestCase):
    def run_ci(self, nboots=2, lower=.025, upper=.975, samples_file=None, bw_time=False):
        tree = read_tree_newick('(A:1,B:1);')
        for idx, node in enumerate(tree.traverse_postorder()):
            node.idx = idx
        emd.get_confidence_interval(
            tree, {'A': 1, 'B': 1}, [1, 1], [1.0], [[1], [1]],
            np.array([1., 1.]), 1000, [[1, 0], [0, 1]], [1, 1],
            {'nboots': nboots, 'p_lower': lower, 'p_upper': upper, 'samples_file': samples_file}, threads=1, bw_time=bw_time)
        return tree

    def test_missing_license_falls_back_without_redrawing(self):
        original = cp.Problem.solve
        calls = []
        def solve(problem, **kwargs):
            calls.append(kwargs)
            if kwargs['solver'] == cp.MOSEK:
                raise cp.error.SolverError('MOSEK license unavailable')
            return original(problem, **kwargs)
        output = io.StringIO()
        with patch.object(cp.Problem, 'solve', solve), patch.object(emd.multinomial, 'randomize', return_value=1.) as draw, contextlib.redirect_stdout(output):
            tree = self.run_ci(nboots=10)
        self.assertEqual([c['solver'] for c in calls], [cp.MOSEK, cp.OSQP]*10)
        self.assertEqual(draw.call_count, 20)
        for call in calls[1::2]:
            self.assertNotIn('mosek_params', call)
        for node in tree.traverse_postorder():
            if not node.is_root():
                self.assertAlmostEqual(node.tau_CI[1], 1.)
                self.assertAlmostEqual(node.tau_CI[3], 1.)
        self.assertIn('CI sample 10/10: completed with OSQP', output.getvalue())

    def test_endpoint_quantiles_with_rounding(self):
        distribution = emd.multinomial(list(range(10)), [.1]*10)
        self.assertLess(distribution.acc[-1], 1.)
        self.assertEqual(distribution.get_quantize(0), 0)
        self.assertEqual(distribution.get_quantize(1), 9)
        self.assertEqual(distribution.get_quantize(.25), 2)
        self.assertEqual(emd.compute_CI([4, 1, 9, 2], 0, 1), (1, 9))
        self.assertEqual(emd.compute_CI([4], 0, 1), (4, 4))

    def test_ci_endpoints_complete(self):
        original = cp.Problem.solve
        def solve(problem, **kwargs):
            return original(problem, solver=cp.OSQP, verbose=False)
        with patch.object(cp.Problem, 'solve', solve), contextlib.redirect_stdout(io.StringIO()):
            tree = self.run_ci(lower=0, upper=1)
        for node in tree.traverse_postorder():
            self.assertEqual(node.divTime_CI[0::2], (0, 1))
            if not node.is_root():
                self.assertEqual(node.mu_CI, (0, 1., 1, 1.))
                self.assertAlmostEqual(node.tau_CI[1], 1.)
                self.assertAlmostEqual(node.tau_CI[3], 1.)

    def test_export_all_samples_without_changing_labels(self):
        original = cp.Problem.solve
        def solve(problem, **kwargs):
            return original(problem, solver=cp.OSQP, verbose=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'samples.nwk'
            with patch.object(cp.Problem, 'solve', solve), contextlib.redirect_stdout(io.StringIO()):
                tree = self.run_ci(nboots=100, lower=0, upper=1,
                                   samples_file=path, bw_time=True)
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 100)
            for line in lines:
                self.assertIn('t=-1.0,mu=1.0', line)
                sample = read_tree_newick(line)
                leaves = list(sample.traverse_leaves())
                self.assertEqual({node.label for node in leaves}, {'A', 'B'})
                for node in leaves:
                    self.assertAlmostEqual(node.edge_length, 1.)
            self.assertEqual({node.label for node in tree.traverse_leaves()}, {'A', 'B'})

    def test_optimal_negative_solution_is_rejected(self):
        original = cp.Problem.solve
        calls = []
        def solve(problem, **kwargs):
            calls.append(kwargs['solver'])
            if len(calls) == 1:
                problem._status = cp.OPTIMAL
                problem.variables()[0].value = np.array([-0.01, 1.])
                return 0.
            return original(problem, solver=cp.OSQP, verbose=False)
        output = io.StringIO()
        with patch.object(cp.Problem, 'solve', solve), contextlib.redirect_stdout(output):
            self.run_ci()
        self.assertEqual(calls, [cp.MOSEK]*3)
        self.assertIn('discarding draw 1/10 from MOSEK', output.getvalue())
        self.assertIn('minimum branch length -0.01', output.getvalue())

    def test_nonnegative_lengths_below_bound_are_accepted(self):
        for length in (0., .000791943113541, emd.EPS_tau / 2):
            calls = []
            def solve(problem, **kwargs):
                calls.append(kwargs)
                problem._status = cp.OPTIMAL
                problem.variables()[0].value = np.array([length, length])
            with self.subTest(length=length), patch.object(cp.Problem, 'solve', solve), contextlib.redirect_stdout(io.StringIO()):
                tree = self.run_ci(nboots=1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(min(n.tau_CI[1] for n in tree.traverse_leaves()), length)

    def test_retries_rebuild_same_problem_with_fresh_rates(self):
        problems, objectives, bounds, options = [], [], [], []
        def solve(problem, **kwargs):
            problems.append(problem)
            options.append(kwargs)
            variable = problem.variables()[0]
            variable.value = np.ones(2)
            objectives.append(float(problem.objective.value))
            bounds.append([c.args[0].value.copy() for c in problem.constraints])
            if len(problems) == 2:
                raise cp.error.SolverError('simulated numerical failure')
            problem._status = cp.OPTIMAL
            variable.value = np.array([-.01, 1.] if len(problems) == 1 else [1., 1.])
        with patch.object(cp.Problem, 'solve', solve), patch.object(emd.multinomial, 'randomize', side_effect=[1., 1., 2., 2., 3., 3.]) as draw, contextlib.redirect_stdout(io.StringIO()):
            self.run_ci(nboots=1)
        self.assertEqual(draw.call_count, 6)
        self.assertEqual(len({id(p) for p in problems}), 3)
        np.testing.assert_allclose(objectives, [0., 2000., 8000.])
        for constraint_bounds in bounds[1:]:
            for actual, expected in zip(constraint_bounds, bounds[0]):
                np.testing.assert_array_equal(actual, expected)
        self.assertEqual([o['solver'] for o in options], [cp.MOSEK]*3)
        self.assertEqual([o['verbose'] for o in options], [False, True, True])
        self.assertTrue(all(o['mosek_params'] == {'MSK_IPAR_NUM_THREADS': 1} for o in options))

    def test_tenth_draw_can_succeed_without_switching_solver(self):
        calls = []
        def solve(problem, **kwargs):
            calls.append(kwargs['solver'])
            problem._status = cp.OPTIMAL
            problem.variables()[0].value = np.array(
                [-.01, 1.] if len(calls) < 10 else [1., 1.])
        with patch.object(cp.Problem, 'solve', solve), patch.object(emd.multinomial, 'randomize', return_value=1.) as draw, contextlib.redirect_stdout(io.StringIO()):
            self.run_ci(nboots=1)
        self.assertEqual(calls, [cp.MOSEK]*10)
        self.assertEqual(draw.call_count, 20)

    def test_ten_invalid_draws_stop(self):
        calls = []
        def solve(problem, **kwargs):
            calls.append(kwargs['solver'])
            problem._status = cp.OPTIMAL
            problem.variables()[0].value = np.array([-.01, 1.])
        with patch.object(cp.Problem, 'solve', solve), patch.object(emd.multinomial, 'randomize', return_value=1.) as draw, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'failed after 10 draws'):
                self.run_ci(nboots=1)
        self.assertEqual(calls, [cp.MOSEK]*10)
        self.assertEqual(draw.call_count, 20)

    def test_all_solvers_fail_once_then_raise(self):
        with patch.object(cp.Problem, 'solve', side_effect=cp.error.SolverError('unavailable')) as solve, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'CI sample 1/2 failed with every solver'):
                self.run_ci()
        self.assertEqual(solve.call_count, 4)

    def test_infeasible_status_is_not_accepted(self):
        def solve(problem, **kwargs):
            problem._status = cp.INFEASIBLE
        with patch.object(cp.Problem, 'solve', solve), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'status infeasible'):
                self.run_ci()

    def test_interrupt_is_not_swallowed(self):
        with patch.object(cp.Problem, 'solve', side_effect=KeyboardInterrupt) as solve, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                self.run_ci()
        self.assertEqual(solve.call_count, 1)


if __name__ == '__main__':
    unittest.main()
