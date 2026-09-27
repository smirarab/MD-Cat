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
    def run_ci(self, nboots=2, lower=.025, upper=.975, samples_file=None, bw_time=False, seq_len=1000, target=1., rates=None):
        rates = [1.] if rates is None else rates
        tree = read_tree_newick('(A:1,B:1);')
        for idx, node in enumerate(tree.traverse_postorder()):
            node.idx = idx
        emd.get_confidence_interval(
            tree, {'A': 1, 'B': 1}, [1, 1], rates, [[1/len(rates)]*len(rates)]*2,
            np.array([1., 1.]), seq_len, [[1, 0], [0, 1]], [target, target],
            {'nboots': nboots, 'p_lower': lower, 'p_upper': upper, 'samples_file': samples_file}, threads=1, bw_time=bw_time)
        return tree

    def test_ci_objective_does_not_scale_with_sequence_length(self):
        original = cp.Problem.solve
        objectives = []
        def solve(problem, **kwargs):
            problem.variables()[0].value = np.array([2., 3.])
            objectives.append(float(problem.objective.value))
            return original(problem, solver=cp.OSQP, verbose=False)
        with patch.object(cp.Problem, 'solve', solve), contextlib.redirect_stdout(io.StringIO()):
            small = self.run_ci(nboots=1, seq_len=1)
            large = self.run_ci(nboots=1, seq_len=162228)
        np.testing.assert_allclose(objectives, [5., 5.])
        for left, right in zip(small.traverse_leaves(), large.traverse_leaves()):
            np.testing.assert_allclose(left.tau_CI, right.tau_CI)

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
        self.assertEqual(calls, [cp.MOSEK, cp.OSQP, cp.MOSEK])
        self.assertIn('completed with OSQP', output.getvalue())

    def test_nonnegative_lengths_below_bound_are_accepted(self):
        for length in (0., .000791943113541, emd.EPS_tau / 2):
            calls = []
            def solve(problem, **kwargs):
                calls.append(kwargs)
                problem._status = cp.OPTIMAL
                problem.variables()[0].value = np.array([length, length])
            with self.subTest(length=length), patch.object(cp.Problem, 'solve', solve), contextlib.redirect_stdout(io.StringIO()):
                tree = self.run_ci(nboots=1, target=length)
            self.assertEqual(len(calls), 1)
            self.assertEqual(min(n.tau_CI[1] for n in tree.traverse_leaves()), length)

    def test_fallback_preserves_problem_and_random_draw(self):
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
        self.assertEqual(draw.call_count, 2)
        self.assertEqual(len({id(p) for p in problems}), 1)
        np.testing.assert_allclose(objectives, [0., 0., 0.])
        for constraint_bounds in bounds[1:]:
            for actual, expected in zip(constraint_bounds, bounds[0]):
                np.testing.assert_array_equal(actual, expected)
        self.assertEqual([o['solver'] for o in options], [cp.MOSEK, cp.OSQP, cp.CVXOPT])
        self.assertEqual([o['verbose'] for o in options], [False, False, False])
        self.assertEqual(options[0]['mosek_params'], {'MSK_IPAR_NUM_THREADS': 1})
        self.assertTrue(all('mosek_params' not in o for o in options[1:]))

    def test_invalid_results_exhaust_solvers_on_each_draw(self):
        calls = []
        def solve(problem, **kwargs):
            calls.append(kwargs['solver'])
            problem._status = cp.OPTIMAL
            problem.variables()[0].value = np.array([-.01, 1.])
        with patch.object(cp.Problem, 'solve', solve), patch.object(emd.multinomial, 'randomize', return_value=1.) as draw, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'failed with every solver'):
                self.run_ci(nboots=1)
        self.assertEqual(calls, list(emd.DURATION_SOLVERS)*emd.CI_MAX_DRAW_ATTEMPTS)
        self.assertEqual(draw.call_count, 2*emd.CI_MAX_DRAW_ATTEMPTS)

    def test_replacement_draws_report_samples_separately_from_attempts(self):
        calls = []
        def solve(problem, **kwargs):
            calls.append(kwargs['solver'])
            problem._status = cp.INFEASIBLE if len(calls) <= 8 else cp.OPTIMAL
            problem.variables()[0].value = np.ones(2)
        warning = io.StringIO()
        with patch.object(cp.Problem, 'solve', solve), patch.object(emd.multinomial, 'randomize', return_value=1.) as draw, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(warning):
            self.run_ci(nboots=2)
        self.assertEqual(calls, list(emd.DURATION_SOLVERS)*2 + [cp.MOSEK]*2)
        self.assertEqual(draw.call_count, 8)
        self.assertIn('1/2 CI samples had to be redrawn', warning.getvalue())
        self.assertIn('(2 replacement draws)', warning.getvalue())
        self.assertIn('This may bias CI', warning.getvalue())

    def test_seeded_samples_and_rng_state_survive_invalid_solver_fallback(self):
        import random
        original = cp.Problem.solve
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            outputs = []
            for fail_first in (False, True):
                def solve(problem, **kwargs):
                    if fail_first and kwargs['solver'] == cp.MOSEK:
                        problem._status = cp.OPTIMAL_INACCURATE
                        problem.variables()[0].value = np.ones(2)
                        return 0.
                    return original(problem, solver=cp.OSQP, verbose=False)
                random.seed(42)
                path = Path(directory)/str(fail_first)
                with patch.object(cp.Problem, 'solve', solve):
                    self.run_ci(nboots=5, samples_file=path, rates=[.5, 2.])
                outputs.append((path.read_bytes(), random.getstate()))
            self.assertEqual(outputs[0], outputs[1])

    def test_checkpoint_survives_ci_failure_and_resumes_without_em(self):
        from emd.ci_checkpoint import resume
        original = cp.Problem.solve
        def solve(problem, **kwargs):
            return original(problem, solver=cp.OSQP, verbose=False)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'fit.ci-checkpoint.json'
            fitted = Path(directory) / 'fit.pre-ci.tre'
            samples = Path(directory) / 'samples.nwk'
            options = dict(nboots=3, p_lower=0., p_upper=1., seed=123,
                           checkpoint_file=str(checkpoint), fitted_file=str(fitted),
                           samples_file=str(samples))
            with patch.object(cp.Problem, 'solve', solve), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(emd, 'get_confidence_interval', side_effect=RuntimeError('CI failed')):
                    with self.assertRaisesRegex(RuntimeError, 'CI failed'):
                        emd.MDCat(read_tree_newick('(A:.1,B:.2);'), 3, nrep=1,
                                  maxIter=2, randseed=42, CI_options=options)
                self.assertTrue(checkpoint.is_file())
                self.assertTrue(fitted.is_file())
                self.assertFalse(samples.exists())
                read_tree_newick(fitted.read_text())
                before = checkpoint.read_bytes()
                with patch.object(emd, 'EM_date', side_effect=AssertionError('EM must not run')):
                    first = resume(checkpoint)[0].newick()
                    first_samples = samples.read_text()
                    second = resume(checkpoint)[0].newick()
                    self.assertEqual(first, second)
                    self.assertEqual(first_samples, samples.read_text())
                    self.assertEqual(len(first_samples.splitlines()), 3)
                    resume(checkpoint, options=dict(nboots=2, p_lower=.025, p_upper=.975), ci_seed=456)
                    self.assertEqual(len(samples.read_text().splitlines()), 2)
                self.assertEqual(before, checkpoint.read_bytes())

    def test_all_solvers_fail_until_draw_limit(self):
        with patch.object(cp.Problem, 'solve', side_effect=cp.error.SolverError('unavailable')) as solve, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'CI sample 1/2 failed after 10 draws'):
                self.run_ci()
        self.assertEqual(solve.call_count, 4*emd.CI_MAX_DRAW_ATTEMPTS)

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
