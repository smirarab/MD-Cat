"""Scalar reference calculations frozen before vectorization."""
from math import exp, log, pi, sqrt

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix, diags

from emd.emd_normal_lib import EPS_tau, EPS_omg, MIN_q, _solve_durations

def log_sum_exp(numlist):
    # using log-trick to compute log(sum(exp(x) for x in numlist))
    # mitigate the problem of underflow
    maxx = max(numlist)
    s = sum(exp(x-maxx) for x in numlist)
    result = maxx + log(s)
    return result

def run_Estep(b,s,omega,tau,phi,p_eps=EPS_tau,var_apprx=True):
    N = len(b)
    k = len(omega)
    Q = []

    for b_i,tau_i in zip(b,tau):
        if b_i is None:
            Q.append(None)
            continue
        lq_i = [0]*k
        for j,(omega_j,phi_j) in enumerate(zip(omega,phi)):
            var_ij = omega_j*tau_i/s if not var_apprx else b_i/s
            lq_i[j] += (-(b_i-omega_j*tau_i)**2/2/var_ij + log(phi_j) - log(var_ij)/2)
        s_lqi = log_sum_exp(lq_i)
        q_i = [exp(x-s_lqi) for x in lq_i]
        q_i = [x if x>MIN_q else MIN_q for x in q_i]
        s_qi = sum(q_i)
        if s_qi < 1e-10:
            q_i = [1.0/k]*k
        else:
            q_i = [x/s_qi for x in q_i]
        Q.append(q_i)
    return Q

def f_ll(b,s,tau,omega,phi,var_apprx=True):
    ll = 0
    k = len(phi)
    for (tau_i,b_i) in zip(tau,b):
        if b_i is None:
            continue
        ll_i = [0]*k
        for j,(omega_j,phi_j) in enumerate(zip(omega,phi)):
            var_ij = tau_i*omega_j/s if not var_apprx else b_i/s
            ll_i[j] += (-log(sqrt(2*pi))-(log(var_ij))/2-(b_i-tau_i*omega_j)**2/2/var_ij + log(phi_j))
        result = log_sum_exp(ll_i)
        ll += result
    return ll

def compute_omega_star(tau,Q,b,phi,eps_omg=EPS_omg,mu_avg=None,maxIter=100):
# IMPORTANT: only works with var_apprx. Never call this function
# when var_apprx if False
    N = len(tau)
    k = len(Q[0])
    a = [2*sum(Q[i][j]*tau[i] for i in range(N)) for j in range(k)]
    c = [2*sum(Q[i][j]*tau[i]*tau[i]/b[i] for i in range(N)) for j in range(k)]

    def __solve_Lagrange__(A):
        omega_star = [eps_omg]*k
        lambda_star = {}
        if mu_avg is None:
            ld0 = 0
        else:
            u = mu_avg # numerator
            v = 0 # denominator
            for j in range(k):
                if j in A:
                    u -= eps_omg*phi[j]
                else:
                    u -= a[j]/c[j]*phi[j]
                    v += phi[j]*phi[j]/c[j]
            ld0 = u/v
        for j in range(k):
            if j in A:
                lambda_star[j] = eps_omg*c[j]-ld0*phi[j]-a[j]
            else:
                omega_star[j] = a[j]/c[j]+ld0*phi[j]/c[j]
        return omega_star,lambda_star

    A = set({})
    omega = [mu_avg]*k if mu_avg is not None else None # feasible omega
    for i in range(maxIter):
        omega_star,lambda_star = __solve_Lagrange__(A)
        V = [j for j in range(k) if omega_star[j] < eps_omg]
        if len(V) == 0: # omega_star is feasible
            min_ld = float("inf")
            min_idx = None
            for j in lambda_star:
                if lambda_star[j] < min_ld:
                    min_ld = lambda_star[j]
                    min_idx = j
            if min_ld >= 0:
                return omega_star # feasible and optimal
            else:
                A.remove(min_idx) # remove one constraint in the active set
        elif omega is None:
            omega = [max(omg,eps_omg) for omg in omega_star] # feasible omega
            A = V
        else:
            flag = False
            for i in V:
                if abs(omega[i]-eps_omg) < 1e-10:
                    A.add(i)
                    flag = True
            if not flag:
                delta = 0
                minD = float("inf")
                minIdx = None
                for i in V:
                    delta = (omega_star[i]-eps_omg)/(omega[i]-eps_omg)
                    if delta < minD:
                        minD = delta
                        minIdx = i
                alpha = 1/(1-minD)
                A.add(minIdx)
                omega = [m+alpha*(ms-m) for m,ms in zip(omega,omega_star)] # feasible omega
    return omega

def compute_tau_star_cvxpy(tau,omega,Q,b,s,M,dt,eps_tau=EPS_tau,var_apprx=False,solvers=['mosek','osqp','cvxopt','ecos'],threads=None,solver_tolerances=None):
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

    P = diags(Pd, format='csc')
    var_tau = cp.Variable(N)

    objective = cp.Minimize(cp.quad_form(var_tau,P) + q.T @ var_tau)
    constraints = [np.zeros(N)+eps_tau <= var_tau, csr_matrix(M)@var_tau == np.array(dt)]
    prob = cp.Problem(objective,constraints)
    # Dense CVXPY quadratic-form conversion normalizes the diagonal before
    # forming a cone. Preserve that scaling explicitly without a dense matrix.
    scale = float(np.max(np.abs(Pd)))
    scaled = Pd / scale if scale else Pd
    # Match CVXPY's float64 dense quadratic-form pivot cutoff.
    mask = scaled > 1e6 * np.finfo(float).eps
    if np.any(mask):
        factor = diags(np.sqrt(scaled), format='csc')[mask, :]
        conic_quadratic = scale * cp.sum_squares(factor @ var_tau)
    else:
        conic_quadratic = cp.Constant(0.)
    conic_objective = cp.Minimize(conic_quadratic + q.T @ var_tau)
    conic_problem = cp.Problem(conic_objective, constraints)
    solver_map = {'mosek':cp.MOSEK,'osqp':cp.OSQP,'cvxopt':cp.CVXOPT,'ecos':cp.ECOS}
    values, _ = _solve_durations(prob, var_tau, M, dt,
                                 solvers=tuple(solver_map[name] for name in solvers),
                                 threads=threads, solver_tolerances=solver_tolerances,
                                 conic_problem=conic_problem)
    return values
