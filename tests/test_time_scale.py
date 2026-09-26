"""Scientific invariants for calibration-unit normalization and CI resume."""
import contextlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import cvxpy as cp
import numpy as np
from treeswift import read_tree_newick

from emd import emd_normal_lib as emd
from emd.ci_checkpoint import resume
from emd.time_scale import TimeScale
from emd.util import date_to_years


TREE = '((A:.1,B:.2)X:.1,C:.3)R;'


class TimeNormalizationTest(unittest.TestCase):
    def setUp(self):
        original_solve = cp.Problem.solve

        def solve(problem, **kwargs):
            return original_solve(problem, solver=cp.OSQP, verbose=False,
                                  eps_abs=1e-9, eps_rel=1e-9)

        solver_patch = patch.object(cp.Problem, 'solve', solve)
        solver_patch.start()
        self.addCleanup(solver_patch.stop)
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def fit(self, scale=1., origin=0., backward=False, **kwargs):
        options = dict(k=3, nrep=1, maxIter=30, randseed=[1], threads=1,
                       root_time=origin, leaf_time=origin+scale,
                       min_branch=.001*scale)
        if backward:
            options.update(root_time=-origin, leaf_time=-(origin+scale), bw_time=True)
        options.update(kwargs)
        return emd.MDCat(read_tree_newick(TREE), **options)

    def assert_scaled(self, baseline, result, scale, origin=0., ci=False):
        self.assertAlmostEqual(baseline[1], result[1], places=7)
        np.testing.assert_allclose(baseline[2], result[2], rtol=1e-8)
        np.testing.assert_allclose(baseline[3], np.array(result[3])*scale, rtol=1e-7)
        for base, node in zip(baseline[0].traverse_postorder(), result[0].traverse_postorder()):
            # Large absolute origins limit the precision of subtraction.
            self.assertAlmostEqual(base.time, (node.time-origin)/scale, places=6)
            self.assertAlmostEqual(base.mu, node.mu*scale, places=7)
            if base.q is not None:
                np.testing.assert_allclose(base.q, node.q)
            if not node.is_root():
                duration = float(str(node.edge_length).split('[')[0]) / scale
                base_duration = float(str(base.edge_length).split('[')[0])
                self.assertAlmostEqual(base_duration, duration, places=7)
                if ci:
                    np.testing.assert_allclose(np.array(node.tau_CI)[[1,3]]/scale,
                                               np.array(base.tau_CI)[[1,3]], rtol=1e-7)
                    np.testing.assert_allclose(np.array(node.mu_CI)[[1,3]]*scale,
                                               np.array(base.mu_CI)[[1,3]], rtol=1e-7)
            if ci:
                np.testing.assert_allclose((np.array(node.divTime_CI)[[1,3]]-origin)/scale,
                                           np.array(base.divTime_CI)[[1,3]], rtol=1e-7, atol=1e-8)

    def test_fit_is_invariant_to_units_and_origin(self):
        baseline = self.fit()
        # Includes the 10000-fold rescaling that previously collapsed all rates.
        for scale, origin in ((1e-6, 0.), (1e4, 0.), (1e6, 7e6), (1., 1e9)):
            with self.subTest(scale=scale, origin=origin):
                self.assert_scaled(baseline, self.fit(scale, origin), scale, origin)

    def test_active_minimum_duration_is_in_original_units(self):
        baseline = self.fit(min_branch=.4)
        result = self.fit(1e4, min_branch=4000.)
        self.assert_scaled(baseline, result, 1e4)
        durations = [node.edge_length for node in result[0].traverse_postorder()
                     if not node.is_root()]
        self.assertGreaterEqual(min(durations), 4000.-1e-4)
        self.assertAlmostEqual(min(durations), 4000., places=3)

    def test_ci_exports_and_resume_preserve_original_units(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            ci = dict(nboots=4, p_lower=0., p_upper=1., seed=321)
            baseline_samples = folder/'baseline.samples'
            baseline = self.fit(CI_options=dict(ci, samples_file=str(baseline_samples)))
            for scale, origin, backward in ((1e6, 7e6, False), (1e-6, -1e-6, True)):
                with self.subTest(scale=scale, backward=backward):
                    checkpoint, fitted, samples = [folder/name for name in
                                                  ('checkpoint.json', 'fitted.tre', 'samples.tre')]
                    options = dict(ci, checkpoint_file=str(checkpoint), fitted_file=str(fitted),
                                   samples_file=str(samples))
                    result = self.fit(scale, origin, backward, CI_options=options)
                    self.assert_scaled(baseline, result, scale, origin, ci=True)
                    state = json.loads(checkpoint.read_text())
                    self.assertAlmostEqual(state['time_scale']['span']/scale, 1.)
                    self.assertEqual(state['time_scale']['origin'], origin)
                    self.assertAlmostEqual(state['eps_tau']/scale, .001)
                    self.assertEqual(state['smpl_times']['R'], origin)
                    np.testing.assert_allclose(np.array(state['M']) @ state['tau'], state['dt'],
                                               atol=scale*1e-7)
                    np.testing.assert_allclose(np.array(state['omega'])*scale, baseline[3], rtol=1e-7)
                    # The pre-CI artifact must also contain physical durations.
                    prefit = read_tree_newick(fitted.read_text())
                    for a, b in zip(prefit.traverse_postorder(), baseline[0].traverse_postorder()):
                        if not a.is_root():
                            self.assertAlmostEqual(a.edge_length/scale,
                                                   float(str(b.edge_length).split('[')[0]), places=7)
                    original_output, original_samples = result[0].newick(), samples.read_text()
                    with patch.object(emd, 'EM_date', side_effect=AssertionError('must not refit')):
                        resumed = resume(checkpoint)
                    self.assertEqual(resumed[0].newick(), original_output)
                    self.assertEqual(samples.read_text(), original_samples)
                    for base_line, line in zip(baseline_samples.read_text().splitlines(),
                                               original_samples.splitlines()):
                        base_tree, tree = read_tree_newick(base_line), read_tree_newick(line)
                        for base_node, node in zip(base_tree.traverse_postorder(), tree.traverse_postorder()):
                            def value(n, name):
                                return float(re.search(r'\b'+name+r'=([^,\]]+)', n.node_params).group(1))
                            time = value(node, 't') * (-1 if backward else 1)
                            self.assertAlmostEqual((time-origin)/scale, value(base_node, 't'), places=7)
                            if not node.is_root():
                                self.assertAlmostEqual(node.edge_length/scale, base_node.edge_length, places=7)
                                self.assertAlmostEqual(value(node, 'mu')*scale, value(base_node, 'mu'), places=7)
                    # Tiny forward times and tiny rates must not round to zero in labels.
                    self.assertTrue(all(n.mu > 0 for n in result[0].traverse_postorder()))

    def test_legacy_checkpoint_without_transform_uses_original_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint = folder/'checkpoint.json'
            result = self.fit(2., 3., CI_options=dict(nboots=2, p_lower=0., p_upper=1., seed=4,
                              checkpoint_file=str(checkpoint), fitted_file=str(folder/'fitted.tre')))
            state = json.loads(checkpoint.read_text())
            del state['time_scale']
            checkpoint.write_text(json.dumps(state))
            original_ci = emd.get_confidence_interval
            with patch.object(emd, 'get_confidence_interval', wraps=original_ci) as ci:
                resumed = resume(checkpoint)
            self.assertEqual(ci.call_args.kwargs['time_scale'], TimeScale())
            self.assertAlmostEqual(resumed[0].root.time, 3.)
            for node in resumed[0].traverse_leaves():
                self.assertAlmostEqual(node.time, 5.)
            np.testing.assert_allclose(resumed[3], result[3])

    def test_reference_tree_is_normalized_without_mutating_caller(self):
        reference = read_tree_newick('((A:.65,B:.65)X:.35,C:1)R;')
        baseline = self.fit(refTree=reference, fixed_tau=True)
        for node in reference.traverse_postorder():
            if not node.is_root():
                node.edge_length *= 1e4
        before = reference.newick()
        result = self.fit(1e4, refTree=reference, fixed_tau=True)
        self.assertEqual(before, reference.newick())
        self.assert_scaled(baseline, result, 1e4)

    def test_calendar_dates_restore_before_formatting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'dates.txt'
            path.write_text('R 2010-01-01\nA 2015-01-01\nB 2015-01-01\nC 2015-01-01\n')
            origin = date_to_years('2010-01-01')
            span = date_to_years('2015-01-01')-origin
            ci = dict(nboots=2, p_lower=0., p_upper=1., seed=8)
            baseline = self.fit(CI_options=ci)
            result = self.fit(sampling_time=str(path), root_time=None, leaf_time=None,
                              as_date=True, min_branch=.001*span, CI_options=ci)
            self.assert_scaled(baseline, result, span, origin, ci=True)
            self.assertIn('t=2010-01-01', result[0].root.label)
            for node in result[0].traverse_leaves():
                self.assertIn('t=2015-01-01', node.label)

    def test_unidentifiable_or_nonfinite_times_fail_before_solving(self):
        for root, leaf in ((1., 1.), (float('nan'), 1.), (0., float('inf'))):
            with self.subTest(root=root, leaf=leaf), patch.object(emd, 'rtt_mu') as rtt:
                with self.assertRaisesRegex(ValueError, 'calibration times'):
                    self.fit(root_time=root, leaf_time=leaf)
                rtt.assert_not_called()


if __name__ == '__main__':
    unittest.main()
