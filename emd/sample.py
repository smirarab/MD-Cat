"""Sample treePL calibrations, execute isolated dating jobs, and summarize them."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import socket
import subprocess
import sys
import tempfile
import time

from emd import PROGRAM_VERSION
from emd import PROGRAM_NAME, PROGRAM_VERSION
from emd.cli import add_dating_options, positive_int, thread_limits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        temporary = stream.name
    os.replace(temporary, path)


@contextmanager
def exclusive(path):
    """Prevent simultaneous workers or coordinators from owning the same work."""
    path = Path(path)
    try:
        stream = path.open('x')
    except FileExistsError:
        raise ValueError(f'{path}: another process owns this lock; if interrupted, verify it has stopped before removing the lock') from None
    with stream:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname(), started=time.time()), stream)
    try:
        yield
    finally:
        path.unlink()


def parser():
    result = argparse.ArgumentParser(description=(
        'Sample treePL bounds and run MD-CAT. Ages must be nonnegative backward '
        'times; all tips are present-day (0). Calendar/forward times are unsupported.'),
        epilog=('Defaults shown apply to new analyses. With --resume or '
                '--summarize-only, omitted scientific settings, summary method, '
                'output format, and output path come from the saved manifest. '
                'By default, a new analysis prepares inputs, runs all jobs, and '
                'writes a complete summary.'))
    add_dating_options(result, sampled=True)
    result.add_argument('-S', '--calibration-samples', type=positive_int, default=100,
                        help='Accepted calibration samples / dating jobs (default: %(default)s)')
    result.add_argument('--strategy', choices=('bottom-up', 'independent'), default='bottom-up',
                        help='Calibration sampling strategy (default: %(default)s)')
    result.add_argument('--distribution', choices=('exponential', 'uniform'), default='exponential',
                        help='Calibration proposal distribution (default: %(default)s)')
    result.add_argument('--max-draws', type=positive_int, default=10000000,
                        help='Maximum calibration proposal attempts (default: %(default)s)')
    result.add_argument('--jobs', type=positive_int, default=1, help='Concurrent dating processes (default: 1)')
    result.add_argument('--summary', choices=('mean', 'median'), default='mean',
                        help='Central age statistic across fitted runs (default: %(default)s)')
    result.add_argument('--format', choices=('nexus', 'annotated', 'treepl', 'clean'), default='nexus',
                        help='Summary tree format (default: %(default)s)')
    result.add_argument('--workdir', type=Path, help='Permanent working directory (default: OUTPUT.runs)')
    mode = result.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Prepare inputs and executable commands only (default: off)')
    mode.add_argument('--resume', action='store_true', help='Run missing/failed jobs from the manifest (default: off)')
    mode.add_argument('--summarize-only', action='store_true', help='Summarize available complete, valid runs; report exclusions (default: off)')
    mode.add_argument('--run-job', type=positive_int, help=argparse.SUPPRESS)
    return result


def create_plan(args):
    # Import only after setting the numerical library thread limits.
    import numpy as np
    import treeswift
    from emd.calibration import convert, sample_to_folder
    if not args.input or not args.samplingTime:
        raise ValueError('new analyses require -i TREE and -t TREEPL_CONFIG')
    source, config = Path(args.input).resolve(), Path(args.samplingTime).resolve()
    output = Path(args.output or (str(source)+'.sampled.nex')).resolve()
    workdir = (args.workdir or Path(str(output)+'.runs')).resolve()
    if workdir.exists():
        if not (workdir/'manifest.json').exists():
            raise ValueError(f'working directory exists without a completed manifest: {workdir}; choose a new --workdir')
        raise ValueError(f'working directory exists: {workdir}; use --resume or --summarize-only')
    if any(Path(str(output)+suffix).exists() for suffix in ('', '.tsv', '.json')):
        raise ValueError(f'output already exists: {output}')
    _, metadata = convert(config.read_text())
    length = args.seqLen
    if length is None:
        length = positive_int(metadata['numsites']) if 'numsites' in metadata else 1000
    tree = treeswift.read_tree_newick(str(source))
    if isinstance(tree, list):
        raise ValueError('input must contain exactly one rooted tree')
    for node in tree.traverse_postorder():
        if not node.is_leaf() and len(node.children) < 2:
            raise ValueError('unary input nodes are not supported')
        if not node.is_root() and (node.edge_length is None or not np.isfinite(node.edge_length) or node.edge_length < 0):
            raise ValueError('input substitution branch lengths must be finite and nonnegative')
    master_seed = args.randSeed if args.randSeed is not None else secrets.randbits(64)
    children = np.random.SeedSequence(master_seed).spawn(args.calibration_samples+1)
    def child_seed(sequence):
        return int(sequence.generate_state(1, dtype=np.uint64)[0])
    sample_seed = child_seed(children[0])
    workdir.mkdir(parents=True)
    (workdir/'input.tre').write_bytes(source.read_bytes())
    (workdir/'calibrations.config').write_bytes(config.read_bytes())
    metadata.update(config_path=str(config), config_sha256=digest(config))
    sample_to_folder(metadata, workdir/'input.tre', workdir/'calibrations',
                     args.calibration_samples, sample_seed, args.max_draws, args.min_branch,
                     args.distribution, False, args.strategy)
    print(f'Master seed: {master_seed}; sequence length: {length}', flush=True)
    options = vars(args).copy()
    for key in ('workdir', 'dry_run', 'resume', 'summarize_only', 'run_job', 'jobs'):
        options.pop(key, None)
    options.update(input=str(source), samplingTime=str(config), output=str(output),
                   seqLen=length, randSeed=master_seed)
    plan = dict(schema=1, version=PROGRAM_VERSION, created=time.time(), options=options,
                workdir=str(workdir), input_sha256=digest(workdir/'input.tre'),
                config_sha256=digest(workdir/'calibrations.config'), sample_seed=sample_seed, jobs=[])
    (workdir/'jobs').mkdir()
    for index, sequence in enumerate(children[1:], 1):
        fit_sequence, ci_sequence = sequence.spawn(2)
        fit_seed, ci_seed = child_seed(fit_sequence), child_seed(ci_sequence)
        run = workdir/'runs'/f'sample_{index:03d}'
        run.mkdir(parents=True)
        calibration = workdir/'calibrations'/f'sample_{index:03d}.txt'
        command = [sys.executable, '-m', 'emd.cli', '-i', str(workdir/'input.tre'),
                   '-t', str(calibration), '-o', str(run/'fitted.tre'), '-b',
                   '-k', str(args.ncat), '-p', str(args.rep), '-l', str(length),
                   '--maxIter', str(args.maxIter), '--randSeed', str(fit_seed),
                   '--threads', str(args.threads), '--min-branch', str(args.min_branch),
                   '--annotate', str(args.annotate)]
        if args.verbose:
            command.append('-v')
        if args.CI:
            command += ['--CI', f"{args.CI['nboots']} 0 1", '--CI-samples', str(run/'ci-replicates.tre'), '--ci-seed', str(ci_seed)]
        job = dict(id=index, run=str(run), calibration=str(calibration),
                   calibration_sha256=digest(calibration), fit_seed=fit_seed,
                   ci_seed=ci_seed, command=command)
        job['fingerprint'] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
        plan['jobs'].append(job)
        worker = [sys.executable, '-m', 'emd.sample', '--workdir', str(workdir), '--run-job', str(index)]
        script = workdir/'jobs'/f'sample_{index:03d}.sh'
        # The commented dating command is directly runnable; the worker adds
        # locking, separate logs, and a completion record for safe resume.
        script.write_text('#!/bin/sh\n# Dating command: '+shlex.join(command)+'\nexec '+shlex.join(worker)+'\n')
    commands = [shlex.join(['sh', str(workdir/'jobs'/f"sample_{job['id']:03d}.sh")]) for job in plan['jobs']]
    (workdir/'commands.txt').write_text('\n'.join(commands)+'\n')
    # NUL-separated script paths work with xargs even when paths contain spaces.
    (workdir/'jobs.nul').write_bytes(b''.join(os.fsencode(workdir/'jobs'/f"sample_{j['id']:03d}.sh")+b'\0' for j in plan['jobs']))
    plan['fingerprint'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    atomic_json(workdir/'manifest.json', plan)
    print(f'Prepared {len(plan["jobs"])} jobs in {workdir}', flush=True)
    return plan


def load_plan(workdir):
    path = Path(workdir).resolve()/'manifest.json'
    plan = json.loads(path.read_text())
    if plan.get('schema') != 1:
        raise ValueError('unsupported sampling manifest schema')
    expected = plan.get('fingerprint')
    actual = hashlib.sha256(json.dumps({k:v for k,v in plan.items() if k!='fingerprint'}, sort_keys=True).encode()).hexdigest()
    if expected != actual:
        raise ValueError('manifest was modified; prepare a new analysis')
    if plan['version'] != PROGRAM_VERSION:
        raise ValueError(f"manifest uses MD-Cat {plan['version']}; current version is {PROGRAM_VERSION}")
    if Path(plan['workdir']) != path.parent:
        raise ValueError('working directory moved; generated commands contain absolute paths')
    for filename, key in [('input.tre', 'input_sha256'), ('calibrations.config', 'config_sha256')]:
        if digest(path.parent/filename) != plan[key]:
            raise ValueError(f'{filename} differs from the manifest')
    for job in plan['jobs']:
        if digest(job['calibration']) != job['calibration_sha256']:
            raise ValueError(f"calibration input changed for job {job['id']}")
    return plan


def validate_job(plan, job):
    """Accept only complete parseable trees, matching taxa, and all expected CIs."""
    from emd.summary import read_run, read_trees, inspect_tree
    import treeswift
    run = Path(job['run'])
    files = [run/'fitted.tre'] + ([run/'ci-replicates.tre'] if plan['options']['CI'] else [])
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    _, nodes, _, _ = read_run(files[0], .001)
    # Compare with the original substitution-tree topology, not a possibly
    # invalid first fitted run. Branch lengths in the reference are not ages.
    reference = treeswift.read_tree_newick(str(Path(plan['workdir'])/'input.tre'))
    clades, keys = set(), {}
    for node in reference.traverse_postorder():
        key = frozenset([node.label]) if node.is_leaf() else frozenset().union(*(keys[c] for c in node.children))
        keys[node] = key
        clades.add(key)
    if set(nodes) != clades:
        raise ValueError('fitted tree topology or tips differ from input')
    if len(files) == 2:
        trees = read_trees(files[1])
        if len(trees) != plan['options']['CI']['nboots']:
            raise ValueError(f"expected {plan['options']['CI']['nboots']} CI replicates, found {len(trees)}")
        for index, tree in enumerate(trees, 1):
            _, other, _, _ = inspect_tree(tree, f'{files[1]} replicate {index}', .001)
            if set(other) != clades:
                raise ValueError(f'CI replicate {index} topology or tips differ from input')
    after = [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    if before != after:
        raise ValueError('output files changed during validation; job may still be running')
    return {p.name: digest(p) for p in files}


def available(plan, job):
    run = Path(job['run'])
    if (run/'running.lock').exists():
        return False, 'job is running or has an interrupted-process lock'
    try:
        state = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else None
        if state and (state.get('status') != 'complete' or state.get('fingerprint') != job['fingerprint']):
            return False, 'job not marked complete for this manifest'
        checksums = validate_job(plan, job)
        if state and checksums != state['outputs']:
            return False, 'outputs changed after completion'
        return True, None
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        return False, str(exc)


def run_job(plan, job):
    run = Path(job['run'])
    with exclusive(run/'running.lock'):
        atomic_json(run/'status.json', dict(status='running', fingerprint=job['fingerprint']))
        try:
            with (run/'stdout.log').open('w') as out, (run/'stderr.log').open('w') as err:
                command = list(job['command'])
                checkpoint = run/'fitted.tre.ci-checkpoint.json'
                if plan['options']['CI'] and checkpoint.is_file():
                    command += ['--resume-ci', str(checkpoint)]
                result = subprocess.run(command, stdout=out, stderr=err, check=False)
            if result.returncode:
                raise RuntimeError(f'dating process exited {result.returncode}; see {run}/stderr.log')
            outputs = validate_job(plan, job)
            atomic_json(run/'status.json', dict(status='complete', fingerprint=job['fingerprint'], outputs=outputs))
        except BaseException as exc:
            atomic_json(run/'status.json', dict(status='failed', fingerprint=job['fingerprint'], error=str(exc)))
            raise
    return job['id']


def write_summary(plan, output, statistic, output_format, partial):
    from emd.summary import summarize, read_run
    included, excluded = [], []
    for job in plan['jobs']:
        valid, reason = available(plan, job)
        if valid:
            included.append(job)
        else:
            excluded.append(dict(job=job['id'], reason=reason))
    print(f"Summary: {len(included)}/{len(plan['jobs'])} complete runs; {len(excluded)} excluded", flush=True)
    for item in excluded:
        print(f"  Excluded job {item['job']}: {item['reason']}", flush=True)
    if excluded and not partial:
        raise ValueError('not all jobs completed; use --resume, or --summarize-only for an explicitly partial summary')
    if not included:
        raise ValueError('no complete valid runs available to summarize')
    output = Path(output).resolve()
    # Replacing our own earlier partial summary is safe; protect unrelated files.
    destinations = [Path(str(output)+suffix) for suffix in ('', '.tsv', '.json')]
    if any(p.exists() for p in destinations):
        meta = json.loads(destinations[2].read_text()) if destinations[2].exists() else {}
        if meta.get('sampling_manifest') != plan['fingerprint']:
            raise ValueError(f'output already exists and belongs to another analysis: {output}')
    trees = [Path(j['run'])/'fitted.tre' for j in included]
    ci_files = [Path(j['run'])/'ci-replicates.tre' for j in included] if plan['options']['CI'] else []
    ci = plan['options']['CI']
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp)/'summary'
        summarize(trees, staged, quantile=.5 if statistic=='median' else None,
                  ci_files=ci_files, ci=(ci['p_lower'],ci['p_upper']) if ci else None,
                  output_format=output_format)
        meta_path = Path(str(staged)+'.json')
        meta = json.loads(meta_path.read_text())
        meta.update(sampling_manifest=plan['fingerprint'], expected_runs=len(plan['jobs']),
                    included_jobs=[j['id'] for j in included], excluded_jobs=excluded)
        atomic_json(meta_path, meta)
        combined = Path(temp)/'fitted.trees'
        combined.write_text(''.join(read_run(p,.001)[0].newick()+'\n' for p in trees))
        for suffix in ('', '.tsv', '.json'):
            os.replace(Path(str(staged)+suffix), Path(str(output)+suffix))
        os.replace(combined, Path(plan['workdir'])/'fitted.trees')
    print(f'Wrote {output}', flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command_parser = parser()
    args = command_parser.parse_args(argv)
    print(f'{PROGRAM_NAME} {PROGRAM_VERSION}', flush=True)
    # Ask argparse which options were supplied, including compact forms such
    # as -S2 and -p1, rather than trying to tokenize flags ourselves.
    explicit_parser = parser()
    for action in explicit_parser._actions:
        action.default = argparse.SUPPRESS
    specified = set(vars(explicit_parser.parse_args(argv)))
    try:
        existing = args.resume or args.summarize_only or args.run_job is not None
        if existing:
            if args.workdir is None:
                raise ValueError('--workdir is required with --resume, --summarize-only, or a generated job')
            plan = load_plan(args.workdir)
            allowed = {'workdir','resume','summarize_only','run_job','jobs','output','summary','format'}
            for key in specified-allowed:
                value = getattr(args,key)
                if key in ('input','samplingTime'):
                    value = str(Path(value).resolve())
                if value != plan['options'].get(key):
                    raise ValueError(f'{key} conflicts with saved manifest; create a new workdir to change dating or sampling settings')
                if key in ('input', 'samplingTime'):
                    expected = plan['input_sha256' if key=='input' else 'config_sha256']
                    if digest(value) != expected:
                        raise ValueError(f'{key} contents conflict with the saved input snapshot')
            thread_limits(plan['options']['threads'])
        else:
            thread_limits(args.threads)
            plan = create_plan(args)
        workdir = Path(plan['workdir'])
        if args.run_job is not None:
            if args.run_job > len(plan['jobs']):
                raise ValueError('job number out of range')
            job = plan['jobs'][args.run_job-1]
            valid,_ = available(plan,job)
            if not valid:
                run_job(plan,job)
            return
        if args.dry_run:
            print('Run externally, for example:\n  xargs -0 -n 1 -P 4 sh < '+shlex.quote(str(workdir/'jobs.nul')))
            print('Then summarize:\n  md_cat_sample.py --workdir '+shlex.quote(str(workdir))+' --summarize-only')
            return
        output = args.output or plan['options']['output']
        statistic = args.summary if 'summary' in specified else plan['options']['summary']
        output_format = args.format if 'format' in specified else plan['options']['format']
        with exclusive(workdir/'coordinator.lock'):
            if not args.summarize_only:
                pending = [job for job in plan['jobs'] if not available(plan,job)[0]]
                if pending:
                    print(f"Launching {len(pending)} dating jobs "
                          f"(up to {min(args.jobs, len(pending))} concurrent processes, "
                          f"{plan['options']['threads']} numerical threads per job). "
                          f"Per-job logs: {workdir / 'runs'}/sample_*/stdout.log and stderr.log",
                          flush=True)
                else:
                    print('All jobs are already complete; proceeding to summary.', flush=True)
                # Threads only coordinate waiting on independent Python processes;
                # every numerical workload and RNG lives in its own subprocess.
                with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                    futures = {pool.submit(run_job,plan,job):job for job in pending}
                    for future in as_completed(futures):
                        job = futures[future]
                        try:
                            future.result()
                            print(f"Completed job {job['id']}/{len(plan['jobs'])}",flush=True)
                        except Exception as exc:
                            print(f"Failed job {job['id']}: {exc}",file=sys.stderr,flush=True)
            write_summary(plan, output, statistic, output_format, partial=args.summarize_only)
    except (OSError, ValueError, RuntimeError, KeyError, argparse.ArgumentTypeError) as exc:
        command_parser.exit(1,f'error: {exc}\n')


if __name__ == '__main__':
    main()
