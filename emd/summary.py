"""Summarize matching dated trees and pooled CI replicates as annotated NEXUS."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import treeswift


def read_trees(path):
    """Read one or more complete Newicks; support quoted labels and comments."""
    text = Path(path).read_text()
    if not text.strip():
        raise ValueError(f"{path}: empty tree file")
    if not text.rstrip().endswith(';'):
        raise ValueError(f"{path}: incomplete tree file (missing final semicolon)")
    if text.lstrip().upper().startswith('#NEXUS'):
        return list(treeswift.read_tree_nexus(text).values())
    result = treeswift.read_tree_newick(text)
    return result if isinstance(result, list) else [result]


def read_run(path, tolerance, clip_negative=True, clipping=None):
    trees = read_trees(path)
    if len(trees) != 1:
        raise ValueError(f'{path}: expected exactly one tree')
    return inspect_tree(trees[0], path, tolerance, clip_negative, clipping)


def inspect_tree(tree, path, tolerance, clip_negative=True, clipping=None):
    nodes, ages, keys, totals, counts, lows, highs = {}, {}, {}, {}, {}, {}, {}
    original_lengths, negatives = {}, []
    labels = set()
    for node in tree.traverse_postorder():
        # Older MD-CAT writers quote the label together with its [t=...]
        # suffix. Newick parsers correctly keep that suffix inside the label,
        # but it is metadata, not part of the taxon identity. Strip only this
        # recognized trailing annotation; preserve other bracketed names.
        if node.label is not None:
            node.label = re.sub(r'\[&?t=[^\[\]]*\]$', '', node.label)
        original_lengths[node] = node.edge_length
        if (clip_negative and not node.is_root() and node.edge_length is not None
                and math.isfinite(node.edge_length) and node.edge_length < 0):
            negatives.append(node.edge_length)
            node.edge_length = 0.
        if not node.is_root() and (node.edge_length is None or
                not math.isfinite(node.edge_length) or node.edge_length < 0):
            raise ValueError(f'{path}: missing, negative, or nonfinite branch length '
                             f'{node.edge_length!r} at node {node.label!r}')
        if node.is_leaf():
            if not node.label or node.label in labels:
                raise ValueError(f'{path}: missing or duplicate tip name {node.label!r}')
            labels.add(node.label)
            key = frozenset([node.label])
            totals[node], counts[node], lows[node], highs[node] = 0., 1, 0., 0.
        else:
            if len(node.children) < 2:
                raise ValueError(f'{path}: unary nodes are not supported')
            key = frozenset().union(*(keys[c] for c in node.children))
            counts[node] = sum(counts[c] for c in node.children)
            totals[node] = sum(totals[c] + counts[c]*c.edge_length for c in node.children)
            lows[node] = min(lows[c]+original_lengths[c] for c in node.children)
            highs[node] = max(highs[c]+original_lengths[c] for c in node.children)
        if highs[node]-lows[node] > tolerance:
            raise ValueError(f'{path}: tip-distance spread {highs[node]-lows[node]:g} '
                             f'exceeds tolerance {tolerance:g}; requires present-day ultrametric tips')
        nodes[key], keys[node] = node, key
        ages[key] = totals[node]/counts[node]
        if clip_negative and not node.is_leaf():
            ages[key] = max(ages[key], max(ages[keys[c]] for c in node.children))
    if negatives and clipping is not None:
        clipping.append(dict(tree=str(path), negative_edges=len(negatives),
                             minimum_length=min(negatives), total_added_length=-sum(negatives)))
    return tree, nodes, ages, highs[tree.root]-lows[tree.root]


def treepl_newick(tree):
    """Plain Newick with fixed six-decimal branch lengths, as treePL writes."""
    rendered = {}
    for node in tree.traverse_postorder():
        label = '' if node.label is None else str(node.label)
        if any(c.isspace() or c in "(),:;[]'" for c in label):
            label = "'" + label.replace("'", "''") + "'"
        subtree = '(' + ','.join(rendered[c] for c in node.children) + ')' if node.children else ''
        edge = '' if node.edge_length is None else f':{node.edge_length:.6f}'
        rendered[node] = subtree + label + edge
    return rendered[tree.root] + ';'


def nexus_text(tree):
    """Rooted NEXUS tree with BEAST metacomments, no internal label ambiguity."""
    rendered = {}
    for node in tree.traverse_postorder():
        label = ("'" + str(node.label).replace("'", "''") + "'") if node.is_leaf() else ''
        subtree = '(' + ','.join(rendered[c] for c in node.children) + ')' if node.children else ''
        params = getattr(node, 'node_params', '')
        annotation = '[' + params + ']' if params else ''
        edge = '' if node.edge_length is None else f':{node.edge_length:.17g}'
        rendered[node] = subtree + label + annotation + edge
    return '#NEXUS\nBegin trees;\n    Tree summary = [&R] ' + rendered[tree.root] + ';\nEnd;\n'


def summarize(paths, output, quantile=None, tolerance=0.001, clean=False,
              ci_files=None, ci=None, output_format='nexus', clip_negative=True):
    clean = clean or output_format == 'clean'
    if output_format not in ('nexus', 'annotated', 'treepl', 'clean'):
        raise ValueError('output format must be nexus, annotated, treepl or clean')
    if quantile is not None and (not math.isfinite(quantile) or not 0 <= quantile <= 1):
        raise ValueError('quantile must lie in [0, 1]')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    paths = [Path(p).resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('provide at least one input, without duplicate files')
    ci_files = [Path(p).resolve() for p in (ci_files or [])]
    if bool(ci_files) != (ci is not None):
        raise ValueError('--ci-files and --ci LOW HIGH must be supplied together')
    if ci is not None:
        if len(ci) != 2 or not all(math.isfinite(q) for q in ci) or not 0 <= ci[0] < ci[1] <= 1:
            raise ValueError('CI endpoints must satisfy 0 <= LOW < HIGH <= 1')
        if len(ci_files) != len(paths):
            raise ValueError('provide exactly one CI replicate file per input run tree')
        if len(set(ci_files)) != len(ci_files) or set(ci_files) & set(paths):
            raise ValueError('CI files must be unique and distinct from central-estimate files')
    output = Path(output)
    table, manifest = Path(str(output)+'.tsv'), Path(str(output)+'.json')
    for dest in (output, table, manifest):
        if dest.exists():
            raise ValueError(f'output already exists: {dest}')
    clipping = []
    tree, nodes, first, spread = read_run(paths[0], tolerance, clip_negative, clipping)
    values = {key: [age] for key, age in first.items()}
    spreads = [spread]
    for path in paths[1:]:
        _, other, ages, spread = read_run(path, tolerance, clip_negative, clipping)
        if nodes.keys() != other.keys():
            raise ValueError(f'{path}: tip set or rooted topology differs from {paths[0]}')
        for key in values:
            values[key].append(ages[key])
        spreads.append(spread)
    summaries = {key: float(np.mean(v) if quantile is None else np.quantile(v, quantile))
                 for key, v in values.items()}
    intervals, ci_counts, ci_spread = {}, [], 0.
    if ci_files:
        keys = list(nodes)
        blocks = []
        for path in ci_files:
            replicates = read_trees(path)
            if not isinstance(replicates, list):
                replicates = [replicates]
            if not replicates:
                raise ValueError(f'{path}: no CI replicate trees')
            block = np.empty((len(replicates), len(keys)))
            for index, replicate in enumerate(replicates):
                _, other, ages, spread = inspect_tree(replicate, f'{path} replicate {index+1}', tolerance, clip_negative, clipping)
                if other.keys() != nodes.keys():
                    raise ValueError(f'{path} replicate {index+1}: tip set or rooted topology differs')
                block[index] = [ages[key] for key in keys]
                ci_spread = max(ci_spread, spread)
            ci_counts.append(len(replicates))
            blocks.append(block)
            del replicates
        pooled = np.concatenate(blocks, axis=0)
        endpoints = np.quantile(pooled, ci, axis=0)
        intervals = {key: (float(endpoints[0,i]), float(endpoints[1,i])) for i,key in enumerate(keys)}
        del blocks, pooled
    node_keys = {node: key for key, node in nodes.items()}
    method = 'mean' if quantile is None else f'quantile_{quantile:g}'
    rows = []
    for i, (key, node) in enumerate(nodes.items(), 1):
        age = summaries[key]
        if node.is_root():
            node.edge_length = None
        else:
            duration = summaries[node_keys[node.parent]]-age
            if duration < -1e-10:
                raise ValueError('summarized ages violate ancestor ordering; input rounding may be too large')
            node.edge_length = max(0., duration)
        # Do not carry per-run rates, CIs, or other annotations into a summary.
        for attr in ('node_params', 'edge_params'):
            if hasattr(node, attr):
                delattr(node, attr)
        node.node_params = f'&height={age:.12g},t={age:.12g},n={len(paths)},summary={method}'
        if ci_files:
            lo, hi = intervals[key]
            node.node_params += f',t_lower={lo:.12g},t_upper={hi:.12g},t_ci={{{lo:.12g},{hi:.12g}}},n_ci={sum(ci_counts)}'
            node.node_params += f',height_CI={{{lo:.12g},{hi:.12g}}},CI_quantiles={{{ci[0]:.12g},{ci[1]:.12g}}},CI_method=pooled_percentile'
        v = values[key]
        rows.append([i, node.label or '', len(key), method, age, len(v),
                     float(np.mean(v)), float(np.std(v, ddof=1)) if len(v)>1 else 0.,
                     min(v), max(v), json.dumps(sorted(key))])
        if ci_files:
            rows[-1].extend([lo, hi, ci[0], ci[1], sum(ci_counts)])
        if clean or output_format == 'treepl':
            delattr(node, 'node_params')
            if clean and not node.is_leaf():
                node.label = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        if output_format == 'nexus':
            stream.write(nexus_text(tree))
        else:
            stream.write((treepl_newick(tree) if output_format == 'treepl' else tree.newick())+'\n')
    with table.open('x', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        header = ['node_id', 'label', 'n_tips', 'statistic', 'age', 'n_runs',
                  'mean', 'sd', 'min', 'max', 'descendant_tips']
        if ci_files:
            header += ['t_lower', 't_upper', 'ci_lower_quantile', 'ci_upper_quantile', 'n_ci']
        writer.writerow(header)
        writer.writerows(rows)
    with manifest.open('x') as stream:
        json.dump(dict(inputs=[str(p) for p in paths], statistic=method, quantile=quantile,
                       n_runs=len(paths), tolerance=tolerance, clean=clean, output_format=output_format, max_tip_depth_spread=max(spreads),
                       ci_files=[str(p) for p in ci_files], ci_quantiles=ci,
                       ci_replicates_per_file=ci_counts, n_ci=sum(ci_counts),
                       ci_weighting='equal weight per replicate', max_ci_tip_depth_spread=ci_spread,
                       clip_negative=clip_negative, clipping=clipping,
                       clipped_edges=sum(c['negative_edges'] for c in clipping),
                       ultrametricity_check='before clipping',
                       clipped_age_reconstruction='mean descendant-tip distance, bounded below by oldest child' if clip_negative else None,
                       age_source='mean distance to descendant tips; tip age 0'), stream, indent=2)
        stream.write('\n')
    if clipping:
        print(f"Clipped {sum(c['negative_edges'] for c in clipping)} negative branches in {len(clipping)} trees; details in {manifest}")
    return len(paths), len(nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('-o', '--output', required=True, type=Path)
    parser.add_argument('--pattern', default='*.tre', help='Glob within input directories (default: *.tre)')
    parser.add_argument('--quantile', type=float, help='Quantile in [0,1]; default is mean')
    parser.add_argument('--format', choices=('nexus', 'annotated', 'treepl', 'clean'), default='nexus', help='nexus (default): BEAST/FigTree metadata; annotated: Newick metadata; treepl: plain Newick, CIs in TSV only')
    parser.add_argument('--tolerance', type=float, default=.001, help='Allowed tip-depth spread in input time units')
    parser.add_argument('--ci-files', nargs='+', type=Path, help='CI replicate files or directories, one file per run')
    parser.add_argument('--ci-pattern', default='*.CIreplicates', help='Glob for CI directories (default: *.CIreplicates)')
    parser.add_argument('--ci', nargs=2, type=float, metavar=('LOW', 'HIGH'), help='Pooled CI quantiles, e.g. 0.025 0.975')
    args = parser.parse_args()
    try:
        paths = []
        for path in args.inputs:
            matches = sorted(p for p in path.glob(args.pattern) if p.is_file()) if path.is_dir() else [path]
            if not matches:
                raise ValueError(f'no files matching {args.pattern} in {path}')
            paths.extend(matches)
        ci_paths = []
        for path in args.ci_files or []:
            matches = sorted(p for p in path.glob(args.ci_pattern) if p.is_file()) if path.is_dir() else [path]
            if not matches:
                raise ValueError(f'no files matching {args.ci_pattern} in {path}')
            ci_paths.extend(matches)
        runs, nodes = summarize(paths, args.output, args.quantile, args.tolerance, False, ci_paths, args.ci, args.format, True)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'error: {exc}\n')
    print(f'Summarized {runs} runs / {nodes} nodes: {args.output}')


if __name__ == '__main__':
    main()
