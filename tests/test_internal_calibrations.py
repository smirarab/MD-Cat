"""Calibration coverage and exact compatibility with the legacy constraint builder."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cvxpy as cp
import numpy as np
from treeswift import read_tree_newick
from emd import emd_normal_lib as emd

TREE = '((A:.1,B:.2)X:.1,(C:.12,D:.15)Y:.2)R;'

# Frozen pre-fix implementation for same-platform before/after comparisons.
def legacy_setup_constr(tree,smpl_times,s,eps_tau=emd.EPS_tau,pseudo=0):
    N = len(list(tree.traverse_postorder()))-1

    M = []
    dt = []
    
    idx = 0
    b = [0]*N
    lb2idx = {}

    for node in tree.traverse_postorder():
        node.idx = idx
        idx += 1
        if not node.is_root():
            b[node.idx] = node.edge_length+pseudo/s
        if node.label in smpl_times:
            node.active = True
            node.t = smpl_times[node.label]
            if not node.is_root():
                node.constraint = [0.]*N
                node.constraint[node.idx] = 1
        if node.is_leaf():
            node.active = node.label in smpl_times
            continue
        # internal nodes
        active_children = [c for c in node.child_nodes() if c.active]
        node.active = (len(active_children) > 0)
        if not node.active:
            continue
        else:                    
            child0 = active_children[0]
            for child in active_children[1:]: 
                m = [x-y for (x,y) in zip(child0.constraint,child.constraint)]
                dt_i = child0.t - child.t
                M.append(m)
                dt.append(dt_i)
            if node.label in smpl_times:
                m = child0.constraint
                dt_i = child0.t - node.t
                M.append(m) 
                dt.append(dt_i) 
            elif not node.is_root():    
                node.constraint = child0.constraint
                node.constraint[node.idx] = 1
                node.t = child0.t
    return M,dt,b


class InternalCalibrationTest(unittest.TestCase):
    def test_every_subset_has_all_pairwise_calibration_constraints(self):
        newick = '(((A:1,B:1)X:1,C:1)Z:1,(D:1,E:1)Y:1)R;'
        labels = ['R', 'Z', 'X', 'Y', 'A', 'B', 'C', 'D', 'E']
        for mask in range(1, 1 << len(labels)):
            times = {label: i for i, label in enumerate(labels) if mask & (1 << i)}
            with self.subTest(times=times):
                tree = read_tree_newick(newick)
                M, dt, _ = emd.setup_constr(tree, times, 1000)
                nodes = {n.label: n for n in tree.traverse_preorder()}
                paths = {}
                for label in times:
                    path = np.zeros(8)
                    node = nodes[label]
                    while not node.is_root():
                        path[node.idx] = 1
                        node = node.parent
                    paths[label] = path
                anchor = next(iter(times))
                expected = np.array([np.r_[paths[l]-paths[anchor], times[l]-times[anchor]]
                                     for l in times if l != anchor]).reshape(-1, 9)
                actual = np.column_stack((np.array(M).reshape(-1, 8), dt))
                self.assertEqual(len(M), len(times)-1)
                if len(M):
                    self.assertEqual(np.linalg.matrix_rank(actual), len(times)-1)
                    self.assertEqual(np.linalg.matrix_rank(np.vstack((actual, expected))), len(times)-1)

    def test_fossils_full_tip_sampling_and_all_node_sampling_are_exactly_unchanged(self):
        cases = [({'R': 5, 'X': 2, 'Y': 3}, True, 0),
                 ({'A': 3, 'B': 4, 'C': 5, 'D': 6}, False, None),
                 ({'R': 0, 'X': 1, 'Y': 2, 'A': 3, 'B': 4, 'C': 5, 'D': 6}, False, None)]
        for times, backward, leaf in cases:
            with self.subTest(times=times):
                constraints = {label: -time if backward else time
                               for label, time in times.items()}
                if leaf is not None:
                    constraints.update({label: -leaf if backward else leaf
                                        for label in ('A', 'B', 'C', 'D')})
                self.assertEqual(
                    legacy_setup_constr(read_tree_newick(TREE), constraints, 1000),
                    emd.setup_constr(read_tree_newick(TREE), constraints, 1000))
                old, old_samples = self.fit(times, backward, leaf, legacy=True)
                new, new_samples = self.fit(times, backward, leaf)
                self.assertEqual(old[1:], new[1:])
                self.assertEqual(old_samples, new_samples)
                self.assertEqual(old[0].newick(), new[0].newick())
                for a, b in zip(old[0].traverse_postorder(), new[0].traverse_postorder()):
                    for attr in ('time', 'mu', 'q', 'divTime_CI', 'tau_CI', 'mu_CI'):
                        self.assertEqual(getattr(a, attr, None), getattr(b, attr, None))

    def test_mixed_and_internal_only_calibrations_hold_in_fit_and_ci(self):
        for times in ({'R': 0, 'X': 2, 'C': 3, 'D': 3},
                      {'R': 0, 'X': 2, 'Y': 3}):
            with self.subTest(times=times):
                fitted, samples = self.fit(times, False, None)
                trees = [fitted[0]] + [read_tree_newick(line) for line in samples.decode().splitlines()]
                for tree in trees:
                    distances = {}
                    for node in tree.traverse_preorder():
                        distance = 0.
                        ancestor = node
                        while not ancestor.is_root():
                            distance += float(str(ancestor.edge_length).split('[')[0])
                            ancestor = ancestor.parent
                        distances[node.label.split('[')[0]] = distance
                    anchor = next(iter(times))
                    for label, time in times.items():
                        self.assertAlmostEqual(distances[label]-distances[anchor], time-times[anchor], places=5)

    def fit(self, times, backward, leaf, legacy=False):
        original_solve = cp.Problem.solve
        def solve(problem, **kwargs):
            return original_solve(problem, solver=cp.OSQP, verbose=False, eps_abs=1e-9, eps_rel=1e-9)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(cp.Problem, 'solve', solve):
            folder = Path(directory)
            calibration = folder/'times.txt'
            calibration.write_text(''.join(f'{label} {time}\n' for label, time in times.items()))
            samples = folder/'samples.nwk'
            builder = legacy_setup_constr if legacy else emd.setup_constr
            with patch.object(emd, 'setup_constr', builder):
                result = emd.MDCat(read_tree_newick(TREE), 3, sampling_time=str(calibration),
                                   bw_time=backward, root_time=None, leaf_time=leaf,
                                   nrep=1, maxIter=30, randseed=[1], threads=1,
                                   CI_options=dict(nboots=3, p_lower=.025, p_upper=.975,
                                                   seed=123, samples_file=str(samples)))
            return result, samples.read_bytes()
