"""Single-run CLI and shared argument definitions (no numerical imports here)."""
import argparse
import math
import os
import sys
import time
from emd import PROGRAM_NAME, PROGRAM_VERSION

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def positive_int(value):
    try:
        count = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('must be a positive integer')
    if count < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return count


def positive_float(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('must be finite and positive')
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def seed_int(value):
    try:
        number = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('must be a nonnegative integer seed')
    if number < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer seed')
    return number


def ci_options(value):
    tokens = value.split()
    if len(tokens) != 3:
        raise argparse.ArgumentTypeError('expected three space-separated numbers in quotes, e.g. --CI "100 0.025 0.975"')
    nboots = positive_int(tokens[0])
    try:
        lower, upper = map(float, tokens[1:])
    except ValueError:
        raise argparse.ArgumentTypeError('LOWER and UPPER must be numeric quantiles')
    if not 0 <= lower < upper <= 1:
        raise argparse.ArgumentTypeError('quantiles must satisfy 0 <= LOWER < UPPER <= 1')
    return dict(nboots=nboots, p_lower=lower, p_upper=upper)


def thread_limits(threads):
    if threads is not None:
        for variable in THREAD_VARIABLES:
            os.environ[variable] = str(threads)


def add_dating_options(parser, sampled=False):
    parser.add_argument('--version', action='version', version=f'{PROGRAM_NAME} {PROGRAM_VERSION}')
    parser.add_argument('-i', '--input', required=not sampled, help='Input Newick tree (required for new analyses)' if sampled else 'Input Newick tree (required)')
    parser.add_argument('-t', '--samplingTime', help='treePL calibration config (required for new analyses)' if sampled else 'Sampling times / fixed calibrations (default: none)')
    parser.add_argument('-o', '--output', help='Final summary tree (default: INPUT.sampled.nex)' if sampled else 'Dated output tree (default: INPUT.mdcatTree)')
    parser.add_argument('-k', '--ncat', type=positive_int, default=50, help='Rate categories (default: 50)')
    parser.add_argument('-p', '--rep', type=positive_int, default=100, help='Optimization initializations per dating run (default: 100)')
    parser.add_argument('-l', '--seqLen', type=positive_int, help='Sequence length; default: config numsites, then 1000' if sampled else 'Sequence length (default: 1000)')
    parser.add_argument('--maxIter', type=positive_int, default=100, help='Maximum EM iterations (default: 100)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose dating output (default: off)')
    parser.add_argument('--CI', type=ci_options, metavar='"N LOWER UPPER"', help='CI replicates and endpoint quantiles, e.g. "100 0.025 0.975" (default: off)')
    parser.add_argument('--randSeed', type=seed_int if sampled else str,
                        help='Master integer seed for sampling and all jobs (default: randomly generated and saved)' if sampled else 'Integer seed or quoted list of -p seeds (default: auto-select)')
    parser.add_argument('--annotate', type=int, choices=(1, 2, 3), default=2, help='Per-run annotations: 1=times, 2=times and rates, 3=also full rate probabilities (default: %(default)s)')
    parser.add_argument('--threads', '--cores', type=positive_int, default=1 if sampled else None,
                        help='Numerical/solver threads per dating process (default: 1)' if sampled else 'Numerical/solver threads (default: library defaults)')
    parser.add_argument('--min-branch', type=positive_float, default=.001,
                        help='Minimum dated branch duration, in calibration units (default: 0.001)')
    if not sampled:
        parser.add_argument('-r', '--rootTime')
        parser.add_argument('-f', '--leafTime')
        parser.add_argument('-b', '--backward', action='store_true', help='Backward ages and contemporaneous tips')
        parser.add_argument('-d', '--asDate', action='store_true')
        parser.add_argument('--CI-samples', metavar='FILE', help='Write every CI replicate; requires --CI')
        parser.add_argument('--ci-seed', type=seed_int, help=argparse.SUPPRESS)


def execute(args):
    """Run with already-validated arguments and thread limits set before imports."""
    print(f'{PROGRAM_NAME} {PROGRAM_VERSION}', flush=True)
    from emd.emd_normal_lib import MDCat
    from treeswift import read_tree_newick
    from emd.util import date_to_years
    start = time.time()
    as_date = args['asDate']
    bw = args['backward'] and not as_date
    def age(value):
        return date_to_years(value) if as_date else float(value)
    root = age(args['rootTime']) if args['rootTime'] is not None else (None if args['samplingTime'] else (1 if bw else 0))
    leaf = age(args['leafTime']) if args['leafTime'] is not None else (0 if bw else (None if args['samplingTime'] else 1))
    randseed = None
    if args['randSeed'] is not None:
        seeds = [int(x) for x in str(args['randSeed']).split()]
        # Preserve the original CLI's -p 1 behavior: one explicitly supplied
        # seed is the restart seed itself, not a seed for generating it.
        randseed = seeds[0] if len(seeds) == 1 and args['rep'] != 1 else seeds
        if isinstance(randseed, list) and len(randseed) != args['rep']:
            raise ValueError('--randSeed list must contain exactly -p seeds')
    tree = read_tree_newick(args['input'])
    if isinstance(tree, list):
        raise ValueError('expected one input tree')
    print('MD-Cat was called as follows: ' + ' '.join(sys.argv), flush=True)
    result = MDCat(tree, args['ncat'], sampling_time=args['samplingTime'],
                   s=args['seqLen'] or 1000, nrep=args['rep'], maxIter=args['maxIter'],
                   verbose=args['verbose'], randseed=randseed, pseudo=1,
                   root_time=root, leaf_time=leaf, bw_time=bw, as_date=as_date,
                   place_mu=args['annotate'] >= 2, place_q=args['annotate'] >= 3,
                   CI_options=args['CI'], threads=args['threads'], min_branch=args['min_branch'])
    output = args['output'] or (args['input'] + '.mdcatTree')
    result[0].write_tree_newick(output)
    print('Best log-likelihood:', result[1])
    print('Runtime:', time.time()-start)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Date a tree with fixed calibrations.')
    add_dating_options(parser)
    # Retain a module-level args mapping for existing embedding/instrumentation.
    global args
    args = vars(parser.parse_args(argv))
    if args['CI_samples'] is not None:
        if args['CI'] is None:
            parser.error('--CI-samples requires --CI')
        args['CI']['samples_file'] = args['CI_samples']
    if args['ci_seed'] is not None:
        if args['CI'] is None:
            parser.error('--ci-seed requires --CI')
        args['CI']['seed'] = args['ci_seed']
    thread_limits(args['threads'])
    try:
        execute(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'error: {exc}\n')


if __name__ == '__main__':
    main()
