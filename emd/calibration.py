"""Parse treePL calibration bounds and sample feasible backward-time ages."""

from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys
import json
import hashlib


def convert(text, single_bound="error"):
    """Return MD-CAT text and metadata, rejecting ambiguous calibrations."""
    calibrations = {}
    metadata = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, sep, value = line.partition("=")
        key = key.strip().lower()
        if key not in {"mrca", "min", "max", "treefile", "numsites"}:
            continue
        if not sep or not value.strip():
            raise ValueError(f"line {lineno}: missing value for {key}")
        if key in ("treefile", "numsites"):
            if key in metadata:
                raise ValueError(f"line {lineno}: duplicate {key}")
            metadata[key] = value.strip()
            continue
        fields = value.split()
        if (key == "mrca" and len(fields) < 3) or (key != "mrca" and len(fields) != 2):
            raise ValueError(f"line {lineno}: malformed {key} directive")
        name = fields[0]
        entry = calibrations.setdefault(name, {})
        if key in entry:
            raise ValueError(f"line {lineno}: duplicate {key} for {name}")
        if key == "mrca":
            if len(set(fields[1:])) < 2:
                raise ValueError(f"line {lineno}: MRCA needs at least two distinct taxa")
            if any(any(c in token for c in "+=\"'") for token in fields):
                raise ValueError(f"line {lineno}: name/taxon contains unsupported MD-CAT delimiter or quote")
            entry[key] = fields[1:]
        else:
            try:
                age = Decimal(fields[1])
            except InvalidOperation:
                raise ValueError(f"line {lineno}: invalid age {fields[1]!r}") from None
            if not age.is_finite() or age < 0:
                raise ValueError(f"line {lineno}: age must be finite and nonnegative")
            entry[key] = age

    rows = []
    for name, entry in calibrations.items():
        if "mrca" not in entry:
            raise ValueError(f"{name}: bound has no MRCA definition")
        bounds = [entry[k] for k in ("min", "max") if k in entry]
        if not bounds:
            raise ValueError(f"{name}: no calibration bounds")
        if len(bounds) == 1 and single_bound != "use-bound":
            raise ValueError(f"{name}: only one bound; use --single-bound use-bound to treat it as an exact age")
        if len(bounds) == 2 and bounds[0] > bounds[1]:
            raise ValueError(f"{name}: min exceeds max")
        age = sum(bounds) / len(bounds)
        rows.append(f"{name}={'+'.join(entry['mrca'])}\t{age:f}\n")
    if not rows:
        raise ValueError("no calibrations found")
    metadata['calibrations'] = calibrations
    return "".join(rows), metadata


def tree_constraints(metadata, tree_path, min_branch=0.001, allow_above_max=False):
    """Build bounds and ancestor constraints shared by adjustment and sampling."""
    import numpy as np
    import treeswift
    from scipy.optimize import linprog

    if not np.isfinite(min_branch) or min_branch <= 0:
        raise ValueError("--min-branch must be finite and positive")
    tree = treeswift.read_tree_newick(str(tree_path))
    if isinstance(tree, list):
        raise ValueError("--tree must contain exactly one tree")
    leaves = {}
    heights = {}
    for node in tree.traverse_postorder():
        if node.is_leaf():
            if node.label in leaves:
                raise ValueError(f"duplicate tree tip: {node.label}")
            leaves[node.label] = node
            heights[node] = 0
        else:
            heights[node] = 1 + max(heights[c] for c in node.children)
    entries = metadata['calibrations']
    names = list(entries)
    nodes = {}
    lower, upper, target = [], [], []
    for i, name in enumerate(names):
        entry = entries[name]
        paths = []
        for taxon in entry['mrca']:
            if taxon not in leaves:
                raise ValueError(f"{name}: taxon absent from tree: {taxon}")
            node = leaves[taxon]
            path = []
            while node is not None:
                path.append(node)
                node = node.parent
            paths.append(path)
        node = next(n for n in paths[0] if all(n in p for p in paths[1:]))
        if node in nodes:
            raise ValueError(f"{name} and {names[nodes[node]]} resolve to the same node; combine these calibrations first")
        nodes[node] = i
        lo = float(entry.get('min', entry.get('max')))
        hi = float(entry.get('max', entry.get('min')))
        target.append((lo + hi) / 2)
        lower.append(max(lo, heights[node] * min_branch))
        upper.append(hi)
    matrix, gaps = [], []
    for node, i in nodes.items():
        parent, distance = node.parent, 1
        while parent is not None:
            if parent in nodes:
                row = np.zeros(len(names))
                row[nodes[parent]], row[i] = 1, -1
                matrix.append(row)
                gaps.append(distance * min_branch)
            parent, distance = parent.parent, distance + 1
    lower, upper, target = map(np.asarray, (lower, upper, target))
    # Zero-width intervals remain fixed even when exponential tails are allowed.
    feasibility_upper = np.where(upper > np.array([float(entries[n].get('min', entries[n].get('max'))) for n in names]), np.inf, upper) if allow_above_max else upper
    if np.any(lower > feasibility_upper):
        raise ValueError("bounds cannot accommodate minimum branch durations to present-day tips")
    matrix = np.asarray(matrix).reshape((-1, len(names)))
    gaps = np.asarray(gaps)
    feasible = linprog(np.zeros(len(names)), A_ub=-matrix if len(gaps) else None,
                       b_ub=-gaps if len(gaps) else None,
                       bounds=list(zip(lower, feasibility_upper)), method='highs')
    if not feasible.success:
        raise ValueError(f"calibration bounds are infeasible on this tree: {feasible.message}")
    return names, lower, upper, target, matrix, gaps, feasible.x


def adjust_to_tree(metadata, tree_path, min_branch=0.001):
    """Project midpoint ages onto bounded, topology-consistent fixed ages."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, minimize
    names, lower, upper, target, matrix, gaps, initial = tree_constraints(metadata, tree_path, min_branch)
    entries = metadata['calibrations']
    constraints = [LinearConstraint(matrix, gaps, np.inf)] if len(gaps) else []
    fit = minimize(lambda x: np.sum((x-target)**2), initial,
                   jac=lambda x: 2*(x-target), bounds=Bounds(lower, upper),
                   constraints=constraints, method='SLSQP',
                   options={'ftol': 1e-12, 'maxiter': 2000})
    if not fit.success:
        raise ValueError(f"age adjustment failed: {fit.message}")
    # Validate the actual serialized ages, not just the optimizer result.
    ages = np.array([float(f'{age:.12f}') for age in fit.x])
    if (np.any(ages < lower-1e-8) or np.any(ages > upper+1e-8)
            or np.any(matrix @ ages < gaps-1e-8)):
        raise ValueError("adjusted ages failed bounds/topology validation")
    rows = []
    for name, age, midpoint in zip(names, ages, target):
        rows.append(f"{name}={'+'.join(entries[name]['mrca'])}\t{age:.12f}\n")
        if abs(age-midpoint) > 1e-7:
            print(f"Adjusted {name}: {midpoint:g} -> {age:.12f}", file=sys.stderr)
    return ''.join(rows)


def bottom_up_draws(rng, size, lower, upper, parent, child, gaps, distribution="exponential"):
    """Draw a batch in descendant-first order, with whole-draw rejection."""
    import numpy as np
    descendants = [[] for _ in lower]
    for p, c, gap in zip(parent, child, gaps):
        descendants[p].append((c, gap))
    # The constraints contain every ancestor/descendant pair. A proper
    # descendant always has fewer calibrated descendants than its ancestor.
    order = sorted(range(len(lower)), key=lambda i: (len(descendants[i]), i))
    draws = np.empty((size, len(lower)))
    valid = np.ones(size, dtype=bool)
    for i in order:
        offset = np.full(size, lower[i])
        for c, gap in descendants[i]:
            offset = np.maximum(offset, draws[:, c]+gap)
        valid &= offset <= upper[i]
        # Invalid rows still consume RNG values for reproducible batching;
        # they remain rejected. A zero-width valid interval is a fixed age.
        scale = np.maximum(upper[i]-offset, 0.) / np.log(20)
        draws[:, i] = (offset + rng.exponential(scale) if distribution == "exponential"
                       else rng.uniform(offset, np.maximum(offset, upper[i])))
        valid &= draws[:, i] <= upper[i]
    return draws, valid


def sample_to_folder(metadata, tree_path, output, count, seed, max_draws, min_branch,
                     distribution='exponential', allow_above_max=False, strategy='bottom-up'):
    """Whole-vector rejection sampling, with bounded memory and attempt limit."""
    import numpy as np
    if count < 1 or max_draws < count:
        raise ValueError("sample count must be positive and max draws must be at least the count")
    if distribution not in ('uniform', 'exponential'):
        raise ValueError("unknown sampling distribution")
    if allow_above_max and distribution != 'exponential':
        raise ValueError("--allow-above-max requires --distribution exponential")
    if strategy not in ('independent', 'bottom-up'):
        raise ValueError('unknown sampling strategy')
    if strategy == 'bottom-up' and allow_above_max:
        raise ValueError('bottom-up requires hard maxima (omit --allow-above-max)')
    if output.exists():
        raise ValueError(f"output folder already exists: {output}")
    sampling_min_branch = min_branch * (1 + 1e-8)
    try:
        names, lower, upper, _, matrix, gaps, _ = tree_constraints(metadata, tree_path, sampling_min_branch, allow_above_max)
    except ValueError:
        # Exact-age calibrations can leave precisely the required duration.
        # The numerical margin must not make an otherwise feasible model fail.
        sampling_min_branch = min_branch
        names, lower, upper, _, matrix, gaps, _ = tree_constraints(metadata, tree_path, sampling_min_branch, allow_above_max)
    entries = metadata['calibrations']
    # Draw from original bounds, not the tightened tip-distance bounds.
    original_lower = np.array([float(entries[n].get('min', entries[n].get('max'))) for n in names])
    rng = np.random.Generator(np.random.PCG64(seed))
    parent = np.argmax(matrix, axis=1) if len(gaps) else []
    child = np.argmin(matrix, axis=1) if len(gaps) else []
    accepted, attempts = [], 0
    while len(accepted) < count and attempts < max_draws:
        size = min(10000, max_draws-attempts)
        if strategy == 'bottom-up':
            draws, valid = bottom_up_draws(rng, size, lower, upper, parent, child, gaps, distribution)
        elif distribution == 'uniform':
            draws = rng.uniform(original_lower, upper, size=(size, len(names)))
        else:
            draws = original_lower + rng.exponential((upper-original_lower)/np.log(20), size=(size, len(names)))
        if strategy == 'independent':
            valid = np.ones(size, dtype=bool)
        valid &= np.all(draws >= lower, axis=1)
        if not allow_above_max:
            valid &= np.all(draws <= upper, axis=1)
        for p, c, gap in zip(parent, child, gaps):
            valid &= draws[:, p]-draws[:, c] >= gap
        indices = np.flatnonzero(valid)
        needed = count-len(accepted)
        accepted.extend(draws[indices[:needed]])
        attempts += int(indices[needed-1])+1 if len(indices) >= needed else size
        if attempts % 1000000 == 0:
            print(f"Accepted {len(accepted)}/{count} after {attempts} draws", file=sys.stderr)
    if len(accepted) < count:
        raise ValueError(f"only {len(accepted)}/{count} accepted after {attempts} draws; increase --max-draws (no output written)")
    # Seventeen significant digits round-trip the validated doubles exactly.
    texts = [''.join(f"{n}={'+'.join(entries[n]['mrca'])}\t{a:.17g}\n"
                     for n, a in zip(names, ages)) for ages in accepted]
    output.mkdir(parents=True)
    for i, text in enumerate(texts, 1):
        (output/f'sample_{i:03d}.txt').write_text(text)
    manifest = dict(method=f'{strategy} {distribution} draws, whole-vector rejection',
                    strategy=strategy, batch_size=10000,
                    bottom_up_offset='max(min, descendant age + edge-count * min_branch, tip-distance * min_branch)' if strategy == 'bottom-up' else None,
                    distribution=distribution, allow_above_max=allow_above_max,
                    exponential_scale=('(max-offset)/ln(20)' if strategy == 'bottom-up' else '(max-min)/ln(20)') if distribution == 'exponential' else None,
                    seed=seed, rng='numpy.PCG64', numpy_version=np.__version__,
                    samples=count, calibrations=len(names), attempts=attempts,
                    acceptance_rate=count/attempts, min_branch=min_branch, sampling_min_branch=sampling_min_branch,
                    tree=str(tree_path.resolve()),
                    tree_sha256=hashlib.sha256(tree_path.read_bytes()).hexdigest(),
                    config_sha256=metadata.get('config_sha256'),
                    config=metadata.get('config_path'), leaf_age=0)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f"Wrote {count} samples to {output}; accepted {count}/{attempts} draws ({count/attempts:.3%}).", file=sys.stderr)
