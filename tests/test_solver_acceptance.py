"""Shared duration acceptance, fallback, and last-resort clipping."""
import contextlib
import io
import unittest
from unittest.mock import patch
import cvxpy as cp
import numpy as np
from emd import emd_normal_lib as emd


class SolverAcceptanceTest(unittest.TestCase):
    def run_problem(self, outcomes, M=None, dt=None):
        M = [[1., 1.]] if M is None else M
        dt = [1.] if dt is None else dt
        variable = cp.Variable(2)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(variable)), [np.array(M) @ variable == dt])
        calls = []
        def solve(problem, **kwargs):
            calls.append(kwargs['solver'])
            outcome = outcomes[len(calls)-1]
            if isinstance(outcome, Exception):
                raise outcome
            status, values = outcome
            problem._status = status
            # save_value allows nonfinite values, as a malfunctioning solver might.
            problem.variables()[0].save_value(None if values is None else np.array(values))
        with patch.object(cp.Problem, 'solve', solve), contextlib.redirect_stdout(io.StringIO()):
            values, solver = emd._solve_durations(problem, variable, M, dt)
        return values, solver, calls

    def test_clipping_waits_for_all_solvers_and_keeps_candidate_copy(self):
        outcomes = [(cp.OPTIMAL, [-5e-9, 1.]),
                    (cp.OPTIMAL_INACCURATE, [.5, .5]),
                    cp.error.SolverError('unavailable'), (cp.INFEASIBLE, None)]
        values, solver, calls = self.run_problem(outcomes)
        np.testing.assert_array_equal(values, [0., 1.])
        self.assertEqual(solver, cp.MOSEK)
        self.assertEqual(calls, list(emd.DURATION_SOLVERS))

    def test_valid_alternative_preferred_to_clipping(self):
        values, solver, calls = self.run_problem([
            (cp.OPTIMAL, [-5e-9, 1.]), (cp.OPTIMAL, [.25, .75])])
        np.testing.assert_array_equal(values, [.25, .75])
        self.assertEqual(solver, cp.OSQP)
        self.assertEqual(len(calls), 2)

    def test_clipping_must_still_satisfy_calibrations(self):
        # Original residual is just inside tolerance; clipping makes it too big.
        with self.assertRaisesRegex(RuntimeError, 'after clipping'):
            self.run_problem([(cp.OPTIMAL, [-1e-8, 1.000000209])]*4)

    def test_large_negatives_bad_calibrations_nonfinite_and_status_rejected(self):
        for outcome in [(cp.OPTIMAL, [-1e-6, 1.000001]),
                        (cp.OPTIMAL, [.5, .6]),
                        (cp.OPTIMAL, [np.nan, 1.]),
                        (cp.OPTIMAL, [np.inf, 1.]),
                        (cp.OPTIMAL, None),
                        (cp.OPTIMAL_INACCURATE, [.5, .5])]:
            with self.subTest(outcome=outcome), self.assertRaises(RuntimeError):
                self.run_problem([outcome]*4)

    def test_calibration_absolute_and_relative_tolerances(self):
        for target in (0., 1., 100.):
            allowed = emd.SOLVER_CALIB_ATOL + emd.SOLVER_CALIB_RTOL*abs(target)
            self.assertIsNone(emd._calibration_error([target+.9*allowed], [[1]], [target]))
            self.assertIsNotNone(emd._calibration_error([target+1.1*allowed], [[1]], [target]))

    def test_main_fit_does_not_return_rejected_solution(self):
        def solve(problem, **kwargs):
            problem._status = cp.OPTIMAL_INACCURATE
            problem.variables()[0].value = np.array([-1., 2.])
        with patch.object(cp.Problem, 'solve', solve), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'failed with every solver'):
                emd.compute_tau_star_cvxpy([.5,.5], [1.], [[1.],[1.]], [.5,.5],
                                          1000, [[1.,1.]], [1.], var_apprx=True)

    def test_cli_tolerances_reach_fit_ci_and_resume(self):
        import json
        import tempfile
        from pathlib import Path
        from emd import cli
        from emd.ci_checkpoint import resume
        original = cp.Problem.solve
        def solve(problem, **kwargs):
            return original(problem, solver=cp.OSQP, verbose=False)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch.object(cp.Problem, 'solve', solve):
            folder = Path(directory)
            source = folder/'input.nwk'
            source.write_text('(A:.1,B:.2);')
            output = folder/'fit.nwk'
            values = (2e-7, 3e-7, 4e-8)
            with patch.object(emd, '_solve_durations', wraps=emd._solve_durations) as checked:
                cli.main(['-i', str(source), '-o', str(output), '-p', '1', '-k', '2',
                          '--maxIter', '2', '--randSeed', '1', '--CI', '2 0 1',
                          '--solver-tolerances', *map(str, values)])
                self.assertTrue(checked.call_count)
                self.assertTrue(any('context' in c.kwargs for c in checked.call_args_list))
                self.assertTrue(all(tuple(c.kwargs['solver_tolerances']) == values for c in checked.call_args_list))
            checkpoint = Path(str(output)+'.ci-checkpoint.json')
            self.assertEqual(json.loads(checkpoint.read_text())['solver_tolerances'], list(values))
            for override in (None, (1e-6, 2e-6, 0.)):
                with patch.object(emd, '_solve_durations', wraps=emd._solve_durations) as checked:
                    resume(checkpoint, solver_tolerances=override)
                    self.assertTrue(all(tuple(c.kwargs['solver_tolerances']) == (override or values) for c in checked.call_args_list))
            # Old checkpoints without this field use the existing defaults.
            state = json.loads(checkpoint.read_text())
            del state['solver_tolerances']
            checkpoint.write_text(json.dumps(state))
            resume(checkpoint)
        self.assertEqual(emd.validate_solver_tolerances(), (1e-7, 1e-7, 1e-8))

    def test_custom_values_change_acceptance(self):
        self.assertIsNotNone(emd._calibration_error([1.00001], [[1]], [1]))
        self.assertIsNone(emd._calibration_error([1.00001], [[1]], [1], (1e-4, 0., 0.)))
        for values in ((1., 2.), (-1., 0., 0.), (float('nan'), 0., 0.)):
            with self.assertRaises(ValueError):
                emd.validate_solver_tolerances(values)
