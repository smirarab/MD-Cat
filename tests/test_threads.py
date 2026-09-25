"""Run with: python -m unittest discover -s tests -p 'test_threads.py'."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import cvxpy as cp
from emd import emd_normal_lib as emd
from treeswift import read_tree_newick

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class ThreadLimitsTest(unittest.TestCase):
    def test_cli_environment_before_numerical_imports(self):
        # Stop at the first application import: limits must already be set,
        # without importing NumPy or any solver in this fresh process.
        probe = '''
import builtins, json, os, runpy, sys
original_import = builtins.__import__
def observe(name, *args, **kwargs):
    if name == "emd.emd_normal_lib":
        assert "numpy" not in sys.modules
        print(json.dumps({key: os.environ[key] for key in %r}))
        raise SystemExit(0)
    return original_import(name, *args, **kwargs)
builtins.__import__ = observe
sys.argv = ["md_cat.py", "-i", "unused"] + sys.argv[1:]
runpy.run_path("md_cat.py", run_name="__main__")
''' % (VARIABLES,)
        for flags, expected in (([], "7"), (["--threads", "2"], "2"),
                                (["--cores", "1"], "1")):
            with self.subTest(flags=flags):
                env = dict(os.environ, **dict.fromkeys(VARIABLES, "7"))
                result = subprocess.run([sys.executable, "-c", probe] + flags,
                                        cwd=ROOT, env=env, capture_output=True,
                                        text=True, timeout=30, check=True)
                self.assertEqual(json.loads(result.stdout),
                                 dict.fromkeys(VARIABLES, expected))

    def test_invalid_counts(self):
        for value in ("0", "-1", "1.5", "two"):
            with self.subTest(value=value):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "md_cat.py"), "-i", "unused",
                     "--threads", value], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be a positive integer", result.stderr)

    def test_thread_limit_reaches_optimization_and_ci(self):
        original_solve = cp.Problem.solve
        for threads in (None, 2):
            calls = []

            def solve(problem, **kwargs):
                calls.append(kwargs)
                # Use a real, license-free solver after recording MOSEK options.
                return original_solve(problem, solver=cp.OSQP, verbose=False)

            with self.subTest(threads=threads), patch.object(cp.Problem, "solve", solve):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = emd.MDCat(
                        read_tree_newick("(A:0.1,B:0.2);"), 3, nrep=1,
                        maxIter=2, randseed=42, threads=threads,
                        CI_options={"nboots": 2, "p_lower": 0.025, "p_upper": 0.975})
                self.assertIsNotNone(result[0])
                self.assertGreater(len(calls), 2)
                # The final two solves are the CI bootstrap samples.
                for kwargs in calls:
                    self.assertEqual(kwargs["solver"], cp.MOSEK)
                    if threads is None:
                        self.assertNotIn("mosek_params", kwargs)
                    else:
                        self.assertEqual(kwargs["mosek_params"],
                                         {"MSK_IPAR_NUM_THREADS": threads})

    def test_fallback_does_not_receive_mosek_options(self):
        original_solve = cp.Problem.solve
        calls = []

        def solve(problem, **kwargs):
            calls.append(kwargs)
            if kwargs["solver"] == cp.MOSEK:
                raise cp.error.SolverError("MOSEK unavailable")
            return original_solve(problem, **kwargs)

        with patch.object(cp.Problem, "solve", solve):
            tau = emd.compute_tau_star_cvxpy(
                [1.0], [1.0], [[1.0]], [1.0], 1000, [[1.0]], [1.0], threads=2)
        self.assertAlmostEqual(tau[0], 1.0)
        self.assertEqual(calls[0]["mosek_params"], {"MSK_IPAR_NUM_THREADS": 2})
        self.assertEqual(calls[1]["solver"], cp.OSQP)
        self.assertNotIn("mosek_params", calls[1])


if __name__ == "__main__":
    unittest.main()
