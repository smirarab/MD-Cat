"""Keep category probabilities paired with their rates when sorting."""
import contextlib
import importlib
import io
from itertools import permutations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cvxpy as cp
from treeswift import read_tree_newick

from emd import emd_normal_lib as emd

distributions = importlib.import_module('simulator.multinomial')


def legacy_init(self, omega, phi):
    """Pre-fix constructor, retained only for before/after regression checks."""
    pairs = sorted(zip(omega, phi))
    self.omega = [x[0] for x in pairs]
    self.phi = [x[1] for x in pairs]
    self.acc = distributions.cdf_from_pdf(phi)


class CategoryOrderingTest(unittest.TestCase):
    def test_unsorted_unequal_probabilities_have_correct_quantiles(self):
        dist = distributions.multinomial([10., 1.], [.9, .1])
        self.assertEqual(dist.omega, [1., 10.])
        self.assertEqual(dist.phi, [.1, .9])
        self.assertEqual(dist.acc, [.1, 1.])
        self.assertEqual([dist.get_quantize(q) for q in (0., .05, .5, .95, 1.)],
                         [1., 1., 10., 10., 10.])

    def test_permutations_preserve_sampling_and_quantiles(self):
        pairs = [(1., .1), (4., .3), (10., .6)]
        draws = [.0, .05, .2, .35, .5, .95]
        expected = [1., 1., 4., 4., 10., 10.]
        for permutation in permutations(pairs):
            with self.subTest(permutation=permutation):
                omega, phi = zip(*permutation)
                dist = distributions.multinomial(omega, phi)
                self.assertEqual([dist.get_quantize(q) for q in draws], expected)
                with patch.object(distributions, 'random', side_effect=draws):
                    self.assertEqual([dist.randomize() for _ in draws], expected)

    def test_sorted_weighted_and_unsorted_uniform_inputs_are_unchanged(self):
        cases = [([1., 4., 10.], [.1, .3, .6]),
                 ([10., 1., 4.], [1/3]*3)]
        for omega, phi in cases:
            with self.subTest(omega=omega, phi=phi):
                fixed = distributions.multinomial(omega, phi)
                with patch.object(distributions.multinomial, '__init__', legacy_init):
                    old = distributions.multinomial(omega, phi)
                self.assertEqual(fixed.acc, old.acc)
                self.assertEqual([fixed.get_quantize(q) for q in (0., .025, .5, .975, 1.)],
                                 [old.get_quantize(q) for q in (0., .025, .5, .975, 1.)])
                # Identical random inputs must select identical categories.
                draws = [i/1000 for i in range(1000)]
                with patch.object(distributions, 'random', side_effect=draws):
                    before = [old.randomize() for _ in draws]
                with patch.object(distributions, 'random', side_effect=draws):
                    after = [fixed.randomize() for _ in draws]
                self.assertEqual(before, after)


class CategoryValidationTest(unittest.TestCase):
    def test_invalid_inputs_are_rejected(self):
        cases = [
            ([], [], 'nonempty'),
            ([1], [], 'nonempty'),
            ([1, 2], [1], 'matching lengths'),
            ([1], [.5, .5], 'matching lengths'),
            ([float('nan')], [1], 'values must be finite'),
            ([float('inf')], [1], 'values must be finite'),
            ([1, 2], [-.1, 1.1], 'between 0 and 1'),
            ([1, 2], [float('nan'), 1], 'between 0 and 1'),
            ([1, 2], [float('inf'), 0], 'between 0 and 1'),
            ([1, 2], [0, 0], 'sum to 1'),
            ([1, 2], [.2, .3], 'sum to 1'),
            ([1, 2], [.6, .6], 'sum to 1'),
        ]
        for omega, phi, message in cases:
            with self.subTest(omega=omega, phi=phi):
                with self.assertRaisesRegex(ValueError, message):
                    distributions.multinomial(omega, phi)

    def test_zero_probabilities_and_rounding_are_allowed(self):
        dist = distributions.multinomial(iter([2, 1]), iter([1, 0]))
        self.assertEqual(dist.phi, [0, 1])
        self.assertEqual(dist.get_quantize(.5), 2)
        dist = distributions.multinomial(range(10), [.1]*10)
        self.assertEqual(dist.get_quantize(1), 9)


class DatingOrderingRegressionTest(unittest.TestCase):
    def test_fits_and_time_ci_samples_match_legacy_constructor(self):
        # The second example has produced unsorted categories in real EM fits.
        cases = [
            ('((A:.1,B:.2)X:.1,C:.3)R;', 3, 1),
            ('((A:.014920682545637963,B:.004912560316893886)X:.0028142446486662666,'
             '(C:.033615149298403875,D:.0022270236633807963)Y:.7021828314606268)R;', 5, 27),
        ]
        original_solve = cp.Problem.solve

        def solve(problem, **kwargs):
            return original_solve(problem, solver=cp.OSQP, verbose=False)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(cp.Problem, 'solve', solve), \
                contextlib.redirect_stdout(io.StringIO()):
            folder = Path(directory)

            def fit(newick, k, seed, name):
                samples = folder/name
                result = emd.MDCat(read_tree_newick(newick), k, nrep=1,
                                   maxIter=15, randseed=[seed], threads=1,
                                   CI_options=dict(nboots=4, p_lower=.025,
                                                   p_upper=.975, seed=123,
                                                   samples_file=str(samples)))
                return result, samples.read_bytes()

            for newick, k, seed in cases:
                with self.subTest(tree=newick):
                    with patch.object(distributions.multinomial, '__init__', legacy_init):
                        old, before_samples = fit(newick, k, seed, 'before.tre')
                    fixed, after_samples = fit(newick, k, seed, 'after.tre')
                    # Exact equality, not a relaxed floating-point tolerance.
                    self.assertEqual(old[1:], fixed[1:])  # likelihood, phi, omega
                    self.assertEqual(before_samples, after_samples)
                    for before, after in zip(old[0].traverse_postorder(),
                                             fixed[0].traverse_postorder()):
                        for attr in ('edge_length', 'time', 'mu', 'q', 'divTime_CI'):
                            self.assertEqual(getattr(before, attr), getattr(after, attr))
                        if not before.is_root():
                            self.assertEqual(before.tau_CI, after.tau_CI)
                        # Rate-CI annotations may legitimately change for an
                        # unsorted, nonuniform posterior; they are not time CIs.


if __name__ == '__main__':
    unittest.main()
