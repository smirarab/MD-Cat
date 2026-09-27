"""Sparse storage must preserve constraints, fitted results, and CI samples."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix, issparse
from treeswift import read_tree_newick
from emd import emd_normal_lib as emd
from emd.ci_checkpoint import resume

# Frozen constraint builder from before the storage-only change.
def dense_setup_constr(tree,smpl_times,s,eps_tau=emd.EPS_tau,pseudo=0):
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
        node.active = node.label in smpl_times or bool(active_children)
        # A calibrated internal node is an anchor even without sampled tips.
        # Its own constraint and time, initialized above, propagate upward.
        if not active_children:
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


def dense_compute_tau(tau,omega,Q,b,s,M,dt,eps_tau=emd.EPS_tau,var_apprx=False,solvers=['mosek','osqp','cvxopt','ecos'],threads=None,solver_tolerances=None):
    N = len(b)
    k = len(omega)
    Pd = np.zeros(N)
    q = np.zeros(N)

    for i in range(N):
        if b[i] is None:
            continue
        for j in range(k):
            if not var_apprx:
                w_ij = omega[j]*tau[i] # weight by the variance multiplied with s; use previous tau to estimate
            else:
                w_ij = b[i]    
            Pd[i] += Q[i][j]*omega[j]**2/w_ij
            q[i] -= 2*b[i]*Q[i][j]*omega[j]/w_ij
          
    P = np.diag(Pd)        
    var_tau = cp.Variable(N)
    
    objective = cp.Minimize(cp.quad_form(var_tau,P) + q.T @ var_tau)
    constraints = [np.zeros(N)+eps_tau <= var_tau, csr_matrix(M)@var_tau == np.array(dt)]
    prob = cp.Problem(objective,constraints)
    solver_map = {'mosek':cp.MOSEK,'osqp':cp.OSQP,'cvxopt':cp.CVXOPT,'ecos':cp.ECOS}
    values, _ = emd._solve_durations(prob, var_tau, M, dt,
                                 solvers=tuple(solver_map[name] for name in solvers),
                                 threads=threads, solver_tolerances=solver_tolerances)
    return values


class SparseOptimizationTest(unittest.TestCase):
    def test_constraint_entries_match_dense_for_every_calibration_subset(self):
        labels = ['R', 'Z', 'X', 'Y', 'A', 'B', 'C', 'D', 'E']
        tree = '(((A:1,B:1)X:1,C:1)Z:1,(D:1,E:1)Y:1)R;'
        for mask in range(512):
            times = {label: i for i, label in enumerate(labels) if mask & (1 << i)}
            old, dt, b = dense_setup_constr(read_tree_newick(tree), times, 1000)
            new, new_dt, new_b = emd.setup_constr(read_tree_newick(tree), times, 1000)
            self.assertTrue(issparse(new))
            np.testing.assert_array_equal(new.toarray(), np.array(old).reshape(-1, 8))
            self.assertEqual(dt, new_dt)
            self.assertEqual(b, new_b)

    def test_dense_and_sparse_fits_ci_and_checkpoint_resume_across_solvers(self):
        original = cp.Problem.solve
        original_durations = emd._solve_durations
        tested = []
        for solver in (cp.OSQP, cp.CVXOPT, cp.CLARABEL, cp.MOSEK):
            if solver not in cp.installed_solvers():
                continue
            if solver == cp.MOSEK:
                probe = cp.Variable()
                try:
                    cp.Problem(cp.Minimize(cp.square(probe)), [probe >= 1]).solve(solver=solver)
                except Exception as exc:
                    if 'license' in str(exc).lower():
                        continue
                    raise
            tested.append(solver)
            def solve(problem, **kwargs):
                return original(problem, solver=solver, verbose=False)
            def durations(*args, **kwargs):
                kwargs['solvers'] = (solver,)
                return original_durations(*args, **kwargs)
            with patch.object(emd, '_solve_durations', durations), self.subTest(solver=solver), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch.object(cp.Problem, 'solve', solve):
                folder = Path(directory)
                samples = folder/'samples.nwk'
                checkpoint = folder/'checkpoint.json'
                options = dict(nboots=3, p_lower=.025, p_upper=.975, seed=123,
                               samples_file=str(samples), checkpoint_file=str(checkpoint),
                               fitted_file=str(folder/'fit.nwk'))
                def fit():
                    return emd.MDCat(read_tree_newick('((A:.1,B:.2)X:.1,C:.3)R;'),
                                     3, nrep=2, randseed=[1,2], maxIter=30, threads=1, CI_options=options)
                # These overrides reproduce the old dense objective and builder.
                with patch.object(emd, 'setup_constr', dense_setup_constr), patch.object(emd, 'compute_tau_star_cvxpy', dense_compute_tau), patch.object(emd, 'diags', side_effect=lambda values, **kwargs: np.diag(values)):
                    before = fit()
                    before_samples = samples.read_bytes()
                after = fit()
                self.assertEqual(before[1:], after[1:])
                self.assertEqual(before[0].newick(), after[0].newick())
                self.assertEqual(before_samples, samples.read_bytes())
                after_samples = samples.read_bytes()
                self.assertEqual(resume(checkpoint)[0].newick(), after[0].newick())
                self.assertEqual(after_samples, samples.read_bytes())
                # Old dense schema-1 checkpoints must produce the same CIs too.
                state = json.loads(checkpoint.read_text())
                self.assertEqual(state['schema'], 2)
                matrix = state['M']
                state['M'] = csr_matrix((matrix['data'], matrix['indices'], matrix['indptr']), shape=matrix['shape']).toarray().tolist()
                state['schema'] = 1
                checkpoint.write_text(json.dumps(state))
                self.assertEqual(resume(checkpoint)[0].newick(), after[0].newick())
                self.assertEqual(after_samples, samples.read_bytes())
        self.assertIn(cp.OSQP, tested)

    def test_large_balanced_constraint_storage_is_sparse(self):
        labels = [f'T{i}' for i in range(8192)]
        layer = [label+':1' for label in labels]
        while len(layer) > 1:
            layer = ['('+layer[i]+','+layer[i+1]+'):1' for i in range(0, len(layer), 2)]
        matrix, _, _ = emd.setup_constr(read_tree_newick(layer[0]+';'), dict.fromkeys(labels, 1.), 1000)
        self.assertEqual(matrix.shape, (8191, 16382))
        self.assertLess(matrix.nnz, 20*len(labels))
        self.assertLess(matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes, 2_000_000)
