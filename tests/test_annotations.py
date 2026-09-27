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
