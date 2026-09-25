## Sample calibration ranges (since 1.1.0)

`md_cat_sample.py` samples calibration ages from treePL bounds, dates the tree
for each accepted sample, and summarizes the fitted trees. It is an installed
package command; from a source checkout you can also use `python -m emd.sample`.
The existing `md_cat.py` command continues to accept fixed calibrations.

```bash
md_cat_sample.py -i input.tre -t calibrations.config -o summary.nex \
  --calibration-samples 100 --strategy bottom-up --distribution exponential \
  --randSeed 42 --jobs 4 --threads 2 --CI "100 0.025 0.975"
```

The defaults are 100 calibration samples (`-S`), bottom-up sampling, exponential
ages, one concurrent job, one numerical thread per job, and a mean summary.
`--summary median` selects the median instead. All ages must be nonnegative
**backward ages**, in a common unit, with tips at present time (zero). This
command does not accept `-b`, `-d`, `-r`, or `-f`. Positive numbers alone cannot
identify the direction of a time scale; providing ages before present is the
user's responsibility.

The treePL input must define `mrca`, `min`, and `max` for each calibration. Both
bounds are required; equal bounds specify an exact age. The input tree is
explicitly selected with `-i`, not the config's `treefile`. `numsites` supplies
MD-CAT's sequence length (`-l`); an explicit `-l` overrides it. If neither is
provided, the default is 1000. The selected value is printed and saved.
TreePL's optimization settings, smoothing parameter, and output path are not
used.

### Sampling distributions and branch durations

With `--strategy independent`, each age is drawn independently within its
original interval. With `--strategy bottom-up`, calibrated descendants are drawn
first, and an ancestor's offset is the maximum of its own minimum and the ages
below it plus the required intervening branch durations. Uncalibrated nodes
propagate these minimum-duration requirements; they receive no additional
random calibration. Tips are fixed at zero.

* Exponential: `age = offset + Exponential(scale=(max-offset)/ln(20))`.
  The maximum is the proposal distribution's 95th percentile.
* Uniform: `age = Uniform(offset, max)`.

For independent sampling, `offset = min`. For both strategies, the **entire
calibration proposal** is rejected if any age/offset exceeds its maximum or
violates an ancestor constraint. Rejection conditions the accepted distribution;
the bottom-up and independent strategies represent different sampling models.
There is no automatic switch of strategy if acceptance is low. The default
limit is 10,000,000 attempts, adjustable with `--max-draws`.

`--min-branch` is the minimum **time duration per branch**, shared by sampling,
MD-CAT optimization, and CI optimization. The default is `0.001` in calibration
units (1,000 years if the input is in millions of years). For example, five
intervening edges require at least five times this duration. Sampling uses a
relative numerical margin of `1e-8` above the requested bound when feasible
(omitted if exact calibrations leave no room for it). Values must be
finite and positive; very small values may challenge solver precision.
Fitted branch lengths retain full floating-point precision to avoid rounding
small positive durations to zero.

### Three different replication counts

| Option | Meaning | Default |
| --- | --- | --- |
| `-S`, `--calibration-samples` | Accepted calibration draws / dating jobs | 100 |
| `-p`, `--rep` | Optimization initializations **per dating job** | 100 |
| `--CI "N LOW HIGH"` | CI replicates per fitted tree, and final pooled quantiles | Off |

Thus, 100 calibration samples and `--CI "100 0.025 0.975"` pool 10,000 CI
replicate trees. The central estimate uses only the 100 fitted trees. Each job
internally uses `--CI "100 0 1"` and exports all CI replicates; the requested
2.5th and 97.5th percentiles are computed only after pooling. Per-job CI filenames
are managed by the workflow; `--CI-samples` is not a sampling-command option.
Without `--CI`, only the central summary is produced: variation between
calibration draws is not silently presented as a confidence interval.

`--jobs` launches separate Python **processes**, so dating jobs do not share the
GIL or random state. `--threads` limits numerical/solver threads **within each
job**. Plan for roughly `jobs * threads` CPU cores and memory for every concurrent
solver. The `-p` optimization initializations within a job remain sequential.
Other dating options (`-k`, `-l`, `--maxIter`, `--annotate`, `-v`) keep their usual
meanings; `--annotate` controls per-run trees, while summary annotations are
controlled by `--format`.

One integer `--randSeed` seeds the entire sampling workflow. Independent child
seeds are derived for calibration sampling, each fitted run, and each run's CI
stage. If omitted, a master seed is generated and saved. Seeds do not depend on
job scheduling or resume order. Solver/platform differences can still affect
floating-point results.

### Dry-run, external execution, and resume

```bash
md_cat_sample.py -i input.tre -t calibrations.config -o summary.nex \
  -S 100 --CI "100 0.025 0.975" --randSeed 42 --workdir sampled_runs --dry-run

# Execute generated jobs concurrently (safe for paths containing spaces).
xargs -0 -n 1 -P 4 sh < sampled_runs/jobs.nul

# Summarize the complete runs that are available.
md_cat_sample.py --workdir sampled_runs --summarize-only

# Or run missing/failed jobs, then require a complete summary.
md_cat_sample.py --workdir sampled_runs --resume --jobs 4
```

The permanent working directory defaults to `<output>.runs/`. It includes
copies of the input tree/config, every sampled calibration in `calibrations/`,
a manifest with hashes/settings/seeds, executable `jobs/*.sh`, readable
`commands.txt`, and separate fitted trees, CI files, logs, and completion records
in `runs/sample_NNN/`. Every generated script includes its underlying
`python -m emd.cli` dating command. After summarization, `fitted.trees` combines
the included fitted trees into one multi-Newick file. All per-run files are
retained with or without CI; no local helper scripts are needed.

Run the generated scripts in the same installed Python environment. Their paths
are absolute, so execute the analysis in its original working directory location.
Resume uses the saved copies and rejects modified calibration inputs or a
modified manifest. It also checks the package version and output hashes. To
change sampling/dating settings, start a new working directory. `--jobs` can
change on resume; per-job thread limits and scientific settings stay recorded.

Normal execution and `--resume` produce a final summary only when every run is
complete and valid. `--summarize-only` permits a partial summary and lists every
excluded run and reason. With CI enabled, inclusion requires both a valid fitted
tree and **all expected CI replicates**; central estimates and intervals use the
same runs. No valid runs is an error. Summary metadata records all included and
excluded job IDs. Partial summaries should not be interpreted as the complete
experiment, particularly if failures depend on the sampled calibrations.

Running jobs are protected by lock files. If a process was killed and left a
lock, verify it is no longer running before removing that lock. Do not delete
locks belonging to active jobs. Summaries can be refreshed after more jobs
complete; unrelated existing output files are protected from overwriting.

### Output formats

`--format nexus` (default) stores the central estimate as `height` and pooled
percentile intervals as `height_CI={lower,upper}` in BEAST-style NEXUS metadata.
In FigTree, select **Node Bars → height_CI**. These are percentile CIs, not HPDs.
Other choices are `annotated` (Newick metadata), `treepl` (plain Newick with simple
labels and six-decimal branch lengths), and `clean` (plain Newick without any
internal labels or annotations). All formats have companion `.tsv` statistics
and `.json` provenance files; plain formats keep CIs in the TSV.

Negative branches from older solver outputs are clipped to zero during
summarization and recorded in the JSON. Ages are reconstructed from mean
distances to present-day descendant tips, with parent ages kept at least as old
as their children after clipping. Missing, nonfinite, truncated, or mismatched
trees are rejected. New CI runs also validate solver feasibility before accepting
a replicate.

The standalone packaged summarizer is also available:

```bash
md_cat_summarize.py fitted_runs/ --ci-files ci_runs/ \
  --ci-pattern '*.CIreplicates' --ci 0.025 0.975 -o summary.nex
```

Use `--quantile 0.5` on that command for a median. Directory inputs select
`*.tre` by default; use `--pattern` to exclude unrelated trees. The integrated
sampling workflow uses its manifest rather than broad directory globs.

## Export CI replicate trees

Add `--CI-samples FILE` together with `--CI` to save every CI replicate:

```bash
python md_cat.py -i input.nwk -o dated.nwk --CI "100 0.025 0.975" --CI-samples ci_samples.nwk
```

After all CI samples succeed, `ci_samples.nwk` contains 100 Newick trees,
one per line in sampling order. Each tree preserves the topology and labels,
with branch lengths equal to that replicate's estimated durations. Node comments
record `t` (divergence time) and, for non-root nodes, `mu` (the drawn mutation
rate). The samples file is overwritten if it already exists. 


