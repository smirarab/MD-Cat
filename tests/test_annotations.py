"""Non-CI output must honor the same annotation levels as CI output."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cvxpy as cp
from treeswift import read_tree_newick
from emd import cli, emd_normal_lib as emd


class AnnotationTest(unittest.TestCase):
    def test_cli_annotation_levels_with_and_without_ci(self):
        original_solve = cp.Problem.solve
        def solve(problem, **kwargs):
            return original_solve(problem, solver=cp.OSQP, verbose=False)
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()), \
                patch.object(cp.Problem, 'solve', solve):
            folder = Path(directory)
            source = folder/'input.nwk'
            source.write_text('((A:.1,B:.2)X:.1,C:.3)R;')
            for ci in (False, True):
                baseline = None
                for level in (None, 0, 1, 2, 3):
                    with self.subTest(ci=ci, level=level):
                        output = folder/'output.nwk'
                        flags = ['-i', str(source), '-o', str(output), '-k', '3',
                                 '-p', '1', '--maxIter', '15', '--randSeed', '1', '--threads', '1']
                        if level is not None:
                            flags += ['--annotate', str(level)]
                        if ci:
                            flags += ['--CI', '3 .025 .975', '--ci-seed', '123']
                        cli.main(flags)
                        text = output.read_text()
                        expected_level = 2 if level is None else level
                        self.assertEqual(text.count('[t='), 5 if expected_level else 0)
                        self.assertEqual(',mu=' in text, expected_level >= 2)
                        self.assertEqual(',q=(' in text, expected_level == 3)
                        self.assertEqual('t_lower=' in text, ci and expected_level > 0)
                        if level == 0:
                            self.assertNotIn('[', text)
                            if ci:
                                from emd.ci_checkpoint import resume
                                resumed = resume(str(output) + '.ci-checkpoint.json', threads=1)
                                self.assertEqual(resumed[0].newick(), text.strip())
                        # Remove node annotations to compare topology, original
                        # labels and branch lengths across annotation levels.
                        tree = read_tree_newick(text)
                        for node in tree.traverse_postorder():
                            node.label = node.label.split('[', 1)[0]
                        plain = [(node.label, node.edge_length) for node in tree.traverse_postorder()]
                        if baseline is None:
                            baseline = plain
                        self.assertEqual(plain, baseline)

    def test_non_ci_annotations_use_restored_units_without_changing_fit(self):
        original_solve = cp.Problem.solve
        def solve(problem, **kwargs):
            return original_solve(problem, solver=cp.OSQP, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(cp.Problem, 'solve', solve):
            for backward in (False, True):
                options = dict(k=3, nrep=1, maxIter=15, randseed=[1], threads=1,
                               root_time=10 if backward else 2000,
                               leaf_time=0 if backward else 2010, bw_time=backward,
                               place_q=True)
                with patch.object(emd, 'annotate_divergence_time'):
                    before = emd.MDCat(read_tree_newick('((A:.1,B:.2)X:.1,C:.3)R;'), **options)
                after = emd.MDCat(read_tree_newick('((A:.1,B:.2)X:.1,C:.3)R;'), **options)
                clean = emd.MDCat(read_tree_newick('((A:.1,B:.2)X:.1,C:.3)R;'),
                                  annotate=False, **options)
                self.assertEqual(before[0].newick(), clean[0].newick())
                self.assertEqual(before[1:], clean[1:])
                self.assertEqual(before[1:], after[1:])
                for a, b in zip(before[0].traverse_postorder(), after[0].traverse_postorder()):
                    for attr in ('time', 'mu', 'q', 'edge_length'):
                        self.assertEqual(getattr(a, attr), getattr(b, attr))
                    time = -b.time if backward else b.time
                    self.assertIn('[t=' + str(time), b.label)
                    self.assertIn(',mu=' + str(b.mu), b.label)

    def test_ci_bounds_follow_reported_coordinate_order(self):
        from copy import deepcopy
        from emd.util import date_to_years
        date_mid = date_to_years("2015-01-01")
        date_bounds = (date_to_years("2010-01-01"), date_to_years("2020-01-01"))
        cases = [
            (False, False, -5., (-10., -2.), '-10.0', '-2.0'),
            (True, False, -5., (-10., -2.), '2.0', '10.0'),
            (True, False, -5., (-5., -5.), '5.0', '5.0'),
            (False, True, date_mid, date_bounds, '2010-01-01', '2020-01-01'),
            (True, True, date_mid, date_bounds, '2010-01-01', '2020-01-01'),
        ]
        for backward, date, time, bounds, lower, upper in cases:
            with self.subTest(backward=backward, date=date, bounds=bounds):
                tree = read_tree_newick('(A:1,B:1)R;')
                for node in tree.traverse_postorder():
                    node.time = time
                    node.divTime_CI = (.025, bounds[0], .975, bounds[1])
                    node.mu = 2.
                    node.mu_CI = (.025, 1., .975, 3.)
                    node.q = None
                    node.tau_CI = (.025, .5, .975, 1.5)
                before = deepcopy(tree)
                emd.annotate_divergence_time(tree, bw_time=backward, as_date=date)
                for old, node in zip(before.traverse_postorder(), tree.traverse_postorder()):
                    self.assertIn(f',t_lower={lower},t_upper={upper}', node.label)
                    self.assertIn(',mu_lower=1.0,mu_upper=3.0', node.label)
                    for attr in ('time', 'mu', 'divTime_CI', 'mu_CI', 'tau_CI', 'edge_length'):
                        self.assertEqual(getattr(old, attr), getattr(node, attr))

    def test_backward_ci_annotation_does_not_change_fit_or_replicates(self):
        import re
        original = cp.Problem.solve
        def solve(problem, **kwargs):
            return original(problem, solver=cp.OSQP, verbose=False)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch.object(cp.Problem, 'solve', solve):
            samples = Path(directory)/'samples.nwk'
            def fit(annotate):
                result = emd.MDCat(read_tree_newick('((A:.1,B:.2)X:.1,C:.3)R;'),
                    3, nrep=1, maxIter=15, randseed=[1], threads=1,
                    bw_time=True, root_time=10., leaf_time=0., annotate=annotate,
                    CI_options=dict(nboots=5, p_lower=.025, p_upper=.975,
                                    seed=123, samples_file=str(samples)))
                return result, samples.read_bytes()
            clean, clean_samples = fit(False)
            annotated, annotated_samples = fit(True)
            self.assertEqual(clean[1:], annotated[1:])
            self.assertEqual(clean_samples, annotated_samples)
            for a, b in zip(clean[0].traverse_postorder(), annotated[0].traverse_postorder()):
                for attr in ('time', 'mu', 'divTime_CI', 'mu_CI', 'tau_CI'):
                    self.assertEqual(getattr(a, attr, None), getattr(b, attr, None))
                lower = float(re.search(r't_lower=([^,\]]+)', b.label).group(1))
                upper = float(re.search(r't_upper=([^,\]]+)', b.label).group(1))
                self.assertLessEqual(lower, upper)
                self.assertEqual(lower, -b.divTime_CI[3])
                self.assertEqual(upper, -b.divTime_CI[1])
