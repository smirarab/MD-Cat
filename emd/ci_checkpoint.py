"""Portable pre-CI checkpoints; no executable objects are serialized."""
import json
import os
from pathlib import Path
import random
import tempfile

from emd import PROGRAM_VERSION


def atomic_text(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        temp = handle.name
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            os.unlink(temp)
            raise
    try:
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save(path, tree, smpl_times, tau, omega, phi, Q, llh, b, M, dt, s,
         options, eps_tau, bw_time, as_date, place_mu, place_q):
    import numpy as np
    from copy import deepcopy
    from emd.emd_normal_lib import annotate_divergence_time
    options = options.copy()
    for key in ('samples_file', 'fitted_file', 'checkpoint_file'):
        if options.get(key):
            options[key] = str(Path(options[key]).resolve())
    state = dict(schema=1, version=PROGRAM_VERSION, tree=tree.newick(),
                 indices=[n.idx for n in tree.traverse_postorder()],
                 smpl_times=smpl_times, tau=tau, omega=omega, phi=phi, Q=Q,
                 llh=llh, b=b, M=M, dt=dt, s=s, options=options,
                 eps_tau=eps_tau, bw_time=bw_time, as_date=as_date,
                 place_mu=place_mu, place_q=place_q, random_state=random.getstate())
    def numeric(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Cannot checkpoint {type(value).__name__}')
    atomic_text(path, json.dumps(state, default=numeric, allow_nan=False))
    fitted = deepcopy(tree)
    annotate_divergence_time(fitted, bw_time=bw_time, as_date=as_date,
                             place_mu=place_mu, place_q=place_q)
    fitted_path = options['fitted_file']
    atomic_text(fitted_path, fitted.newick() + '\n')
    print(f'Saved fitted tree without CIs: {fitted_path}', flush=True)
    print(f'Saved CI checkpoint: {path}; resume with --resume-ci {path}', flush=True)


def finish(tree, smpl_times, tau, omega, phi, Q, llh, b, M, dt, s,
           options, eps_tau, bw_time, as_date, place_mu, place_q, threads):
    import numpy as np
    from emd.emd_normal_lib import (get_confidence_interval, convert_to_time,
                                   compute_divergence_time, annotate_divergence_time)
    get_confidence_interval(tree, smpl_times, tau, omega, Q, np.array(b), s, M, dt,
                            options, eps_tau=eps_tau, threads=threads,
                            bw_time=bw_time, as_date=as_date)
    convert_to_time(tree, tau, omega, phi, Q)
    compute_divergence_time(tree, smpl_times)
    annotate_divergence_time(tree, place_mu=place_mu, place_q=place_q,
                             as_date=as_date, bw_time=bw_time)
    for node in tree.traverse_preorder():
        if not node.is_root():
            _, lower, _, upper = node.tau_CI
            node.edge_length = str(node.edge_length) + '[' + str(lower) + ',' + str(upper) + ']'
    return tree, llh, phi, omega


def resume(path, options=None, samples_file=None, ci_seed=None, threads=None):
    from treeswift import read_tree_newick
    with open(path) as handle:
        state = json.load(handle)
    if state.get('schema') != 1:
        raise ValueError('Unsupported CI checkpoint schema')
    tree = read_tree_newick(state['tree'])
    nodes = list(tree.traverse_postorder())
    if len(nodes) != len(state['indices']):
        raise ValueError('Invalid checkpoint node indices')
    for node, index in zip(nodes, state['indices']):
        node.idx = index
    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value
    random.setstate(tuples(state['random_state']))
    saved_options = state['options'].copy()
    if options is not None:
        saved_options.update(options)
    if samples_file is not None:
        saved_options['samples_file'] = samples_file
    if ci_seed is not None:
        saved_options['seed'] = ci_seed
    print(f'Resuming CI only from {path}; optimization skipped', flush=True)
    keys = ('smpl_times', 'tau', 'omega', 'phi', 'Q', 'llh', 'b', 'M', 'dt', 's',
            'eps_tau', 'bw_time', 'as_date', 'place_mu', 'place_q')
    return finish(tree, options=saved_options, threads=threads,
                  **{key: state[key] for key in keys})
