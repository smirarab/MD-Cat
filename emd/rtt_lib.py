from sys import argv
from treeswift import *
from random import choices,sample
from math import fsum, log2

def sample_by_depth(tree,nleaf,nsample):
# return a list of newick trees if do_extract is True
# otherwise, return a nested list of labels
    leaf_labels = []
    leaf_weights = []
    for node in tree.traverse_preorder():
        log_p_inv = log2(len(node.children)) if not node.is_leaf() else 0
        if node.is_root():
            node.log_p_inv = log_p_inv
        else:
            node.log_p_inv = node.parent.log_p_inv + log_p_inv
        if node.is_leaf():
            leaf_labels.append(node.label)
            leaf_weights.append(2**(-node.log_p_inv))
    samples = []
    for s in range(nsample):
        sample = set(choices(leaf_labels,weights=leaf_weights,k=nleaf))
        samples.append(list(sample))
    return samples           

def f_rtt(mu,t0,B,T):
    return sum((b-mu*(t-t0))**2 for b,t in zip(B,T))

def optimize_rtt(root_node,smpl_time,pseudo=0):
    T = []
    B = []
    root_node.d2root = 0
    for node in root_node.traverse_preorder():
        if node is not root_node:
            node.d2root = node.parent.d2root + node.edge_length + pseudo
        if node.label in smpl_time:        
            B.append(node.d2root)
            T.append(smpl_time[node.label]) 
    n = len(T)
    if not n:
        return None,None,None
    origin = T[0]
    T = [t-origin for t in T]
    mean_t = fsum(T)/n
    mean_b = fsum(B)/n
    centered_t = [t-mean_t for t in T]
    centered_b = [b-mean_b for b in B]
    variance = fsum(t*t for t in centered_t)
    if variance == 0:
        return None,None,None
    covariance = fsum(t*b for t,b in zip(centered_t,centered_b))
    mu = max(0.001,covariance/variance)
    t0 = origin + (mean_t-mean_b/mu)
    score = fsum((b-mu*t)**2 for b,t in zip(centered_b,centered_t))
    return mu,t0,score

def bootstrap_rtt(tree,B,T,lb2idx,nsmpl):
    n = len(B)
    I = range(n)
    for i in range(nsmpl):
        J = choices(I,k=n)    
        B_i = [B[j] for j in J]
        T_i = [T[j] for j in J]
        print(optimize_rtt(B_i,T_i))        

def rtt_mu(tree,smpl_time):
    # Merge centered regression moments once per edge instead of walking
    # every clade. Distances in each summary are relative to its clade root.
    origin = next(iter(smpl_time.values()), 0)
    summaries = {}
    mus = []
    for node in tree.traverse_postorder():
        if node.is_leaf():
            node.nleaf = 1
        else:
            node.nleaf = sum(c.nleaf for c in node.children)
        n = int(node.label in smpl_time)
        mean_t = smpl_time[node.label]-origin if n else 0.
        mean_b = variance = covariance = 0.
        for child in node.children:
            cn, ct, cb, cv, cc = summaries.pop(child)
            if not cn:
                continue
            cb += child.edge_length
            if not n:
                n, mean_t, mean_b, variance, covariance = cn, ct, cb, cv, cc
                continue
            total = n + cn
            dt, db = ct-mean_t, cb-mean_b
            weight = n*(cn/total)
            variance += cv + dt*dt*weight
            covariance += cc + dt*db*weight
            mean_t += dt*(cn/total)
            mean_b += db*(cn/total)
            n = total
        summaries[node] = n, mean_t, mean_b, variance, covariance
        if variance != 0:
            mus.append(max(0.001,covariance/variance))
    # Preserve the unweighted mean of valid clade slopes and the rate floor.
    return sum(mus)/len(mus)
