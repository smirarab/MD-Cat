"""Scientific and input-format regression tests for per-node calibrations."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy import integrate, stats

from emd.calibration import convert, distribution_draws, sample_to_folder, tree_constraints
from emd.calibration_distributions import CalibrationDensity, specification
from emd.calibration_inputs import read_calibrations


class DensityTests(unittest.TestCase):
    def test_truncated_cdf_matches_uniform_probabilities(self):
        cases = [
            ('exponential', dict(scale=2), stats.expon(loc=4, scale=2)),
            ('uniform', dict(scale=8), stats.uniform(loc=4, scale=8)),
            ('normal', dict(mean=2, sd=3), stats.norm(loc=6, scale=3)),
            ('gamma', dict(shape=2, scale=3), stats.gamma(2, loc=4, scale=3)),
            ('lognormal', dict(meanlog=1, sdlog=.8), stats.lognorm(.8, loc=4, scale=np.e)),
            ('skew-t', dict(location=2, scale=3, shape=0, df=3), stats.t(3, loc=6, scale=3)),
        ]
        for family, parameters, base in cases:
            with self.subTest(family=family):
                density = CalibrationDensity(specification(family, dict(parameters, offset=4, lower=5, upper=11)))
                x = density.draw(np.random.default_rng(123), np.full(20000, 5.))
                self.assertTrue(np.all((x >= 5) & (x <= 11)))
                transformed = (base.cdf(x)-base.cdf(5))/(base.cdf(11)-base.cdf(5))
                self.assertLess(stats.kstest(transformed, 'uniform').statistic, .012)

    def test_azzalini_skew_t_against_integrated_density(self):
        # Positive/negative skew and both central and tail truncation. This
        # distinguishes Azzalini ST from Jones-Faddy and ordinary Student's t.
        for shape, lo, hi in [(6., 0., 8.), (-6., 2.5, 9.), (6., 0., 1.5)]:
            with self.subTest(shape=shape, lower=lo):
                spec = specification('skew-t', dict(location=2, scale=1.5, shape=shape, df=2.2, lower=lo, upper=hi))
                x = CalibrationDensity(spec).draw(np.random.default_rng(82), np.full(30000, lo))
                def pdf(age):
                    z = (age-2)/1.5
                    return 2/1.5*stats.t.pdf(z, 2.2)*stats.t.cdf(shape*z*np.sqrt(3.2/(2.2+z*z)), 3.2)
                mass = integrate.quad(pdf, lo, hi)[0]
                for cut in np.linspace(lo, hi, 7)[1:-1]:
                    expected = integrate.quad(pdf, lo, cut)[0]/mass
                    self.assertLess(abs(np.mean(x <= cut)-expected), .012)

    def test_extreme_tails_and_empty_conditional_intervals(self):
        for family, parameters, lo, hi in [
            ('normal', dict(mean=0, sd=1), 40., 41.),
            ('exponential', dict(scale=1), 1000., 1001.),
            ('gamma', dict(shape=2, scale=1), 40., 41.),
            ('lognormal', dict(meanlog=0, sdlog=1), 1e5, 2e5),
        ]:
            density = CalibrationDensity(specification(family, dict(parameters, lower=lo, upper=hi)))
            x = density.draw(np.random.default_rng(42), np.full(1000, lo))
            self.assertTrue(np.all((x >= lo) & (x <= hi)))
            self.assertGreater(np.std(x), 0)
            self.assertTrue(np.isnan(density.draw(np.random.default_rng(1), [hi, hi+1, np.nan])).all())

    def test_bad_parameters(self):
        for family, parameters in [
            ('exponential', dict(scale=0)), ('normal', dict(mean=0, sd=-1)),
            ('gamma', dict(shape=0, scale=1)), ('skew-t', dict(location=0, scale=1, shape=2, df=0)),
            ('uniform', dict(scale=2, lower=3)), ('lognormal', dict(meanlog=1000, sdlog=1)),
            ('normal', dict(mean=0, sd=1, upper=0)), ('gamma', dict(shape=1, rate=1)),
            ('normal', dict(mean='nan', sd=1)), ('normal', dict(mean=0, sd=1, lower=-1)),
            ('unknown', dict()),
        ]:
            with self.subTest(family=family, parameters=parameters), self.assertRaises(ValueError):
                specification(family, parameters)

    def test_bottom_up_conditions_without_changing_scale(self):
        density = CalibrationDensity(specification('exponential', dict(offset=1, scale=2, upper=10)))
        draws, valid = distribution_draws(np.random.default_rng(13), 30000,
                                         np.array([1., 4.]), np.array([10., 4.]), np.array([1., 4.]),
                                         [0], [1], [.1], {0: density}, 'bottom-up', 'uniform')
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(draws[:, 1], np.full(30000, 4.))
        x = draws[:, 0]
        transformed = (1-np.exp(-(x-4.1)/2))/(1-np.exp(-(10-4.1)/2))
        self.assertLess(stats.kstest(transformed, 'uniform').statistic, .012)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.tree = self.path/'tree.nwk'
        self.tree.write_text('((A:.1,B:.1):.1,C:.1);')

    def test_native_mixed_sampling_and_reproducibility(self):
        metadata = read_calibrations('mrca = R A C\ndistribution = R gamma shape=3 scale=2 offset=3 upper=20\n'
                                     'mrca = AB A B\nmin = AB 1\nmax = AB 4\n')
        for strategy in ('independent', 'bottom-up'):
            for repeat in (1, 2):
                out = self.path/f'{strategy}{repeat}'
                sample_to_folder(metadata, self.tree, out, 20, 5, 10000, .01, strategy=strategy)
                for path in out.glob('sample_*.txt'):
                    ages = [float(line.split()[1]) for line in path.read_text().splitlines()]
                    self.assertGreaterEqual(ages[0]-ages[1], .01)
                    self.assertTrue(3 <= ages[0] <= 20 and 1 <= ages[1] <= 4)
            self.assertEqual((self.path/f'{strategy}1/sample_001.txt').read_text(),
                             (self.path/f'{strategy}2/sample_001.txt').read_text())
            manifest = json.loads((self.path/f'{strategy}1/manifest.json').read_text())
            self.assertEqual(manifest['node_distributions']['R']['scale'], 2)
            self.assertEqual(manifest['distribution'], 'per-node')

    def test_native_validation(self):
        bad = ['distribution = R normal sd=1',
               'distribution = R gamma shape=2 shape=3 scale=1',
               'distribution = R exponential scale=1\nmin = R 1',
               'distribution = R uniform scale=2 offset=3 upper=2']
        for directive in bad:
            with self.subTest(directive=directive), self.assertRaises(ValueError):
                read_calibrations('mrca = R A C\n'+directive)
        with self.assertRaisesRegex(ValueError, 'sampling workflow'):
            convert('mrca = R A C\ndistribution = R exponential scale=1')

    def test_infeasible_bounds_and_duplicate_mrca(self):
        for text in [
            'mrca = R A C\ndistribution = R normal mean=1 sd=1 upper=2\n'
            'mrca = AB A B\ndistribution = AB uniform offset=3 scale=1',
            'mrca = R A C\ndistribution = R exponential scale=2\n'
            'mrca = R2 B C\ndistribution = R2 exponential scale=2',
        ]:
            with self.assertRaises(ValueError):
                tree_constraints(read_calibrations(text), self.tree)

    def test_mcmctree(self):
        metadata = read_calibrations("3 1\n((A,B)'ST(2, 1.5, 6, 2.2)',C)'G(2,0.1)';")
        entries = list(metadata['calibrations'].values())
        self.assertEqual(entries[0]['distribution']['shape'], 6)
        self.assertEqual(entries[1]['distribution']['scale'], 10)
        for strategy in ('independent', 'bottom-up'):
            sample_to_folder(metadata, self.tree, self.path/strategy, 5, 1, 10000, .001, strategy=strategy)
        for text in ["(A,B)'B(1,2)';", "(A,B)'ST(1,2,3)';", "3 2\n(A,B)'G(1,2)';", "(A,B)'G(1,-2)';"]:
            with self.assertRaises(ValueError):
                read_calibrations(text)
        with self.assertRaisesRegex(ValueError, 'not a clade'):
            tree_constraints(read_calibrations("((A,C)'ST(2,1,6,2)',B);"), self.tree)

    def xml(self, distribution, extra=''):
        return f'''<beast>
          <taxon id="A" spec="Taxon"/><taxon id="B" spec="Taxon"/>
          <taxonset id="ab" spec="TaxonSet"><taxon idref="A"/><taxon idref="B"/></taxonset>
          {extra}
          <distribution id="ab.prior" spec="beast.base.evolution.tree.MRCAPrior" taxonset="@ab" monophyletic="true">
            {distribution}
          </distribution>
        </beast>'''

    def test_beast_parameter_conventions(self):
        cases = [
            ('<Exponential name="distr" mean="3" offset="4"/>', 'exponential', 'scale', 3),
            ('<distr spec="Normal" mean="5" tau="4"/>', 'normal', 'sd', .5),
            ('<distr spec="Gamma" alpha="2" beta="3"/>', 'gamma', 'scale', 3),
            ('<distr spec="Gamma" alpha="2" beta="4" mode="ShapeRate"/>', 'gamma', 'scale', .25),
            ('<distr spec="Gamma" alpha="2" beta="4" mode="ShapeMean"/>', 'gamma', 'scale', 2),
            ('<distr spec="Gamma" alpha="4" mode="OneParameter"/>', 'gamma', 'scale', .25),
            ('<distr spec="Uniform" lower="2" upper="5" offset="10"/>', 'uniform', 'lower', 12),
            ('<distr spec="LogNormalDistributionModel" M="10" S=".5" meanInRealSpace="true"/>', 'lognormal', 'meanlog', np.log(10)-.125),
            ('<distr spec="LogNormalDistributionModel"/>', 'lognormal', 'meanlog', 0),
        ]
        for xml, family, param, expected in cases:
            with self.subTest(family=family):
                metadata = read_calibrations(self.xml(xml))
                spec = metadata['calibrations']['ab.prior']['distribution']
                self.assertEqual(spec['family'], family)
                self.assertAlmostEqual(spec[param], expected)
        metadata = read_calibrations(self.xml('<distr spec="Exponential"><parameter name="mean" idref="m"/></distr>',
                                             '<parameter id="m" spec="RealParameter" estimate="false" value="2"/>'))
        self.assertEqual(metadata['calibrations']['ab.prior']['distribution']['scale'], 2)

    def test_beast_rejects_unsupported_semantics(self):
        for xml in [
            self.xml('<distr spec="Beta"/>'),
            self.xml('<distr spec="Normal" lower="2"/>'),
            self.xml('<distr spec="Gamma" mode="unknown"/>'),
            self.xml('<distr spec="Exponential"><parameter name="mean" estimate="true">2</parameter></distr>'),
            self.xml('<distr spec="Exponential" mean="@missing"/>'),
            self.xml('<distr spec="Exponential" mean="@m"/>', '<state><parameter id="m" value="2"/></state>'),
            self.xml('<distr spec="Normal"/>').replace('monophyletic="true"', 'tipsonly="true"'),
            self.xml('<distr spec="Normal"/>').replace('monophyletic="true"', 'useOriginate="true"'),
        ]:
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                read_calibrations(xml)

    def test_cli_formats_and_resume(self):
        configs = {
            'treepl': 'mrca = AB A B\ndistribution = AB skew-t location=2 scale=1 shape=6 df=2.2 offset=1 lower=1 upper=10',
            'beast2': self.xml('<distr spec="Gamma" alpha="2" beta="1" offset="1"/>'),
            'mcmctree': "3 1\n((A,B)'ST(2,1,6,2.2)',C);",
        }
        for fmt, contents in configs.items():
            config, work = self.path/f'{fmt}.config', self.path/f'{fmt}.runs'
            config.write_text(contents)
            result = subprocess.run([sys.executable, '-m', 'emd.sample', '-i', str(self.tree), '-t', str(config),
                                     '--workdir', str(work), '-S', '2', '--randSeed', '42', '--dry-run'],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((work/'calibrations/manifest.json').read_text())
            self.assertEqual(manifest['input_format'], fmt)
            from emd.sample import load_plan
            self.assertEqual(len(load_plan(work)['jobs']), 2)

    def test_explicit_end_to_end_dating(self):
        config = self.path/'calibrations.config'
        config.write_text('mrca = Root A C\ndistribution = Root gamma shape=3 scale=1 offset=2 upper=10\n'
                          'mrca = AB A B\ndistribution = AB skew-t location=1 scale=.2 shape=6 df=2.2 lower=.5 upper=2\n')
        for strategy in ('independent', 'bottom-up'):
            with self.subTest(strategy=strategy):
                output = self.path/f'{strategy}.nex'
                result = subprocess.run([sys.executable, '-m', 'emd.sample', '-i', str(self.tree), '-t', str(config),
                                         '-o', str(output), '-S', '2', '-p', '1', '-k', '2', '--maxIter', '2',
                                         '--randSeed', '42', '--strategy', strategy],
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertTrue(output.exists())
                self.assertEqual(json.loads(Path(str(output)+'.json').read_text())['n_runs'], 2)


if __name__ == '__main__':
    unittest.main()
