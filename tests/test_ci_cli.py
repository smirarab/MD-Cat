"""Validate CI arguments before loading numerical libraries or input files."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = '''
import builtins, json, runpy, sys
original_import = builtins.__import__
def observe(name, *args, **kwargs):
    if name == 'emd.emd_normal_lib':
        print(json.dumps(sys._getframe(1).f_globals['args']['CI']))
        raise SystemExit(0)
    return original_import(name, *args, **kwargs)
builtins.__import__ = observe
sys.argv = ['md_cat.py', '-i', 'nonexistent-tree'] + sys.argv[1:]
runpy.run_path('md_cat.py', run_name='__main__')
'''


class CIArgumentsTest(unittest.TestCase):
    def run_cli(self, flags):
        return subprocess.run([sys.executable, '-c', PROBE] + flags,
                              cwd=ROOT, capture_output=True, text=True, timeout=10)

    def test_invalid_ci_fails_before_imports(self):
        for value in ('100 1.0 0.0', '100 .5 .5', '100 0 0', '100 1 1', '100 -.1 .9', '100 .1 1.1',
                      '100 nan .9', '100 .1 inf', '0 .025 .975',
                      '-1 .025 .975', '1.5 .025 .975', 'abc .025 .975',
                      '100 lower upper', '100 .025', '100 .025 .975 extra',
                      '100,.025,.975', ''):
            with self.subTest(value=value):
                result = self.run_cli(['--CI', value])
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('error: argument --CI:', result.stderr)
                self.assertNotIn('Traceback', result.stderr)
                self.assertEqual(result.stdout, '')

    def test_export_requires_ci(self):
        result = self.run_cli(['--CI-samples', 'samples.nwk'])
        self.assertEqual(result.returncode, 2)
        self.assertIn('--CI-samples requires --CI', result.stderr)
        self.assertEqual(result.stdout, '')

    def test_export_path_passed_to_ci(self):
        result = self.run_cli(['--CI', '100 0 1', '--CI-samples', 'samples.nwk'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['samples_file'], 'samples.nwk')

    def test_valid_ci_and_default(self):
        for flags, expected in (([], None), (['--CI', '100 0.025 0.975'],
                {'nboots': 100, 'p_lower': .025, 'p_upper': .975}),
                (['--CI', '100 0 1'], {'nboots': 100, 'p_lower': 0., 'p_upper': 1.}),
                (['--CI', '100 0 .975'], {'nboots': 100, 'p_lower': 0., 'p_upper': .975}),
                (['--CI', '100 .025 1'], {'nboots': 100, 'p_lower': .025, 'p_upper': 1.})):
            with self.subTest(flags=flags):
                result = self.run_cli(flags)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), expected)


if __name__ == '__main__':
    unittest.main()
