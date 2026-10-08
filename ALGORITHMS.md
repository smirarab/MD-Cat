# Algorithm and workflow details

This guide describes calibration sampling, confidence-interval handling, and
numerical solver acceptance. See the [README](README.md) for installation and
basic usage.

- [Calibration sampling](#sample-calibration-ranges-since-110)
- [Sampling distributions and branch durations](#sampling-distributions-and-branch-durations)
- [Replication counts and random seeds](#three-different-replication-counts)
- [Execution and resume](#dry-run-external-execution-and-resume)
- [Output formats](#output-formats)
- [CI replicate exports](#export-ci-replicate-trees)
- [Solver acceptance and CI recovery](#solver-acceptance-tolerances)

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

For legacy ranges, the treePL input must define `mrca`, `min`, and `max` for each calibration. Both
bounds are required; equal bounds specify an exact age. The input tree is
explicitly selected with `-i`, not the config's `treefile`. `numsites` supplies
MD-CAT's sequence length (`-l`); an explicit `-l` overrides it. If neither is
provided, the default is 1000. The selected value is printed and saved.
Config `seed` supplies the master random seed and `nthreads` supplies numerical
threads per dating job. Explicit `--randSeed` and `--threads` (or `--cores`)
override them. Without either source, the seed is generated and saved, and
threads default to 1. `nthreads` does not change `--jobs`.
TreePL's optimization settings, smoothing parameter, and output path are not
used.

### Sampling distributions and branch durations

The following describes min/max-only calibrations. Explicit per-node densities
use the conditional sampling rules described in the next section.

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

### Per-node calibration distributions

Use `distribution = NAME FAMILY key=value ...` in the same configuration as
the MRCA definitions. Each node must have either a distribution or a min/max
pair, never both. Parameters are case-sensitive; unknown, duplicate, missing,
nonfinite, and invalid parameters fail before samples are written.

```text
numsites = 1000
seed = 42
mrca = AB A B
distribution = AB exponential scale=5 offset=20 lower=20 upper=50
mrca = Root A C
distribution = Root skew-t location=60 scale=10 shape=6 df=2.2 lower=40 upper=100
```

| Family | Required parameters | Untruncated age before applying bounds |
| --- | --- | --- |
| `exponential` | `scale` | `offset + Exp(mean=scale)` |
| `uniform` | `scale` | `offset + Uniform(0, scale)` |
| `lognormal` | `meanlog`, `sdlog` | `offset + exp(Normal(meanlog, sdlog))` |
| `normal` | `mean`, `sd` | `offset + Normal(mean, sd)` |
| `gamma` | `shape`, `scale` | `offset + Gamma(shape, scale)` |
| `skew-t` | `location`, `scale`, `shape`, `df` | `offset + AzzaliniSkewT(location, scale, shape, df)` |

Every family accepts `offset` (default 0), `lower` (default 0), and `upper`
(default `inf`). Bounds are **absolute ages after shifting**, not distances
from the offset. They truncate and renormalize the distribution. They do not
clip samples to endpoints. Scale, standard deviation, degrees of freedom,
and Gamma shape must be positive. Bounds require `0 <= lower < upper`;
support must have positive width. Fixed ages continue to use equal min/max.
For uniform ages from 80 to 100, use `uniform offset=80 scale=20`.

The lognormal parameters describe the natural logarithm of age minus offset.
Gamma uses **scale**, not rate (`scale = 1/rate`). Normal mean and skew-t
location are not hard minima: use `lower` to enforce a minimum. For positive
families, the offset already establishes a minimum, and `lower` can tighten it.
All dates must share a unit and refer backward from present-day tips.

The skew-t is MCMCTree's Azzalini distribution. For
`z = (age - offset - location)/scale`, its untruncated density is
`2/scale * t_pdf(z, df) * t_cdf(shape*z*sqrt((df+1)/(df+z*z)), df+1)`.
Positive shape produces the long tail toward older ages; negative shape
reverses it. Shape zero is ordinary Student's t. It is not SciPy's
Jones–Faddy skew-t. See the [MCMCTree calibration documentation](https://github.com/abacus-gene/paml/wiki/MCMCtree).

**Independent:** draw each specified density within its own truncation bounds,
then reject the complete vector if any tip-distance or ancestor constraint
fails. Accepted vectors follow the product of the input densities conditioned
on tree feasibility.

**Bottom-up:** draw descendants first. For each node, raise its conditional
lower bound to the maximum of its own lower bound, the tip-distance constraint,
and every calibrated descendant's age plus intervening minimum durations.
Draw from the original density conditioned on this interval. Keep the supplied
offset, location, scale, and other parameters fixed. Reject the entire vector
if an ancestor has no interval left. This is a different joint distribution
from independent rejection; accepted marginal distributions generally differ
from input distributions under either strategy.

This differs from the legacy bottom-up exponential rule, which recomputes
scale from each new offset. In mixed files, min/max-only nodes retain that
legacy rule and obey `--distribution`; explicit densities take precedence.
Uncalibrated nodes only propagate duration constraints.

Sampling uses inverse CDFs, with survival probabilities for right tails and
specialized truncated normal/exponential calculations. Skew-t uses exact
rejection from a truncated Student's t envelope. Numerically unresolvable
intervals raise an error rather than silently becoming point masses. Skew-t
has an internal cap of 10,000 proposals per age; `--max-draws` limits complete
calibration-vector attempts, not these internal proposals. Manifests record
normalized per-node parameters, detected format, SciPy/NumPy versions, and seeds.
For identical inputs, versions, and seeds the calibration samples are reproducible.

#### Importing existing calibration files

`--calibration-format auto` (the default) detects XML, a Newick/MCMCTree tree,
or a treePL-style config. Override with `beast2`, `mcmctree`, or `treepl`.
The substitution tree always comes from `-i`. Imports supply calibration
densities only: tree priors, clock models, MCMC settings, and soft-bound
semantics are not inferred. BEAST/MCMCTree imports use the workflow defaults
for sequence length, seed, and threads unless supplied on the command line.

**MCMCTree:** accepts one semicolon-terminated Newick tree, optionally preceded
by the usual `taxon-count 1` header. Quoted internal-node calibrations may be
`ST(location,scale,shape,df)` or `G(shape,rate)`:

```text
3 1
((A,B)'ST(20,5,6,2.2)',C)'G(10,0.2)';
```

The taxa below each annotated node must form the same clade in the `-i` tree.
Imported ages are truncated at zero, as in MCMCTree. Gamma rate is converted
to scale. Other annotations, including soft `B`, `L`, `U` bounds, fail explicitly;
they must not be treated as hard uniform/exponential bounds. For additional
offsets or truncation, express the same density in the config syntax above.

**BEAST 2:** accepts `MRCAPrior` elements with explicit taxa and fixed scalar
`Exponential`, `Uniform`, `LogNormalDistributionModel`, `Normal`, or `Gamma`
distributions. For example, this calibration-only XML supplies the same
exponential density as `exponential offset=20 scale=5`:

```xml
<beast>
  <distribution id="AB.prior" spec="beast.base.evolution.tree.MRCAPrior">
    <taxonset spec="TaxonSet">
      <taxon id="A" spec="Taxon"/>
      <taxon id="B" spec="Taxon"/>
    </taxonset>
    <distr spec="beast.base.inference.distribution.Exponential" mean="5" offset="20"/>
  </distribution>
</beast>
```

This minimal fragment omits BEAST's analysis tree; a full BEAST analysis XML
can also be supplied. The reader supports `idref`/`@id` references, standard
`map` aliases, scalar attributes, and nested scalar parameters. It preserves
BEAST's exponential mean, normal sigma or precision tau, lognormal M/S and
`meanInRealSpace`, distribution offsets, and Gamma parameterization modes.
See [BEAST's MRCA prior](https://www.beast2.org/xml/beast.base.evolution.tree.MRCAPrior.html)
and [distribution reference](https://www.beast2.org/xml/contents.html).

Taxon sets must explicitly list taxa; alignment-derived sets, templates,
estimated hyperparameters, plugins/custom distributions, tip-only or
originate priors, dated-tip trees, and multiple calibration trees are rejected.
`monophyletic=true` is checked against the supplied tree. BEAST distributions
are additionally conditioned on nonnegative ages. BEAST's standard normal and
Gamma elements do not provide our general truncation fields; use the config
syntax for hard bounds rather than adding unsupported XML attributes.

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

## Solver acceptance tolerances

MD-Cat logs solver outcomes and failures, including a missing MOSEK license,
before trying alternatives; `-v` is not required. Fitting and CI try MOSEK,
OSQP, CVXOPT, and ECOS in that order. Results must have optimal status, finite
branch durations, and acceptable calibration residuals. An invalid result
triggers fallback on the same optimization problem.


Use one option to set the calibration absolute tolerance, calibration relative
tolerance, and tiny-negative clipping tolerance, in that order:

```bash
md_cat.py -i input.nwk --solver-tolerances 1e-7 1e-7 1e-8
```

These are the defaults. All three values must be finite and nonnegative; zero
is allowed. Calibration equations must satisfy
`abs(M @ tau - dt) <= CALIB_ATOL + CALIB_RTOL * abs(dt)`.
Absolute tolerances are in solver time coordinates, normally normalized by the
calibration span. They do not change a solver's internal convergence settings.
Nonnegative durations below `--min-branch` remain acceptable. Negative durations
within `NEGATIVE_ATOL` may be clipped to zero only after all solvers have been
tried, and only if calibration checks still pass after clipping.

The option applies to fitting and CI, including `md_cat_sample.py` jobs. CI
checkpoints preserve it; `--resume-ci` uses saved tolerances unless the option
is supplied again. Older checkpoints use the defaults. Python callers can pass
`solver_tolerances=(1e-7, 1e-7, 1e-8)` to `MDCat` or CI `resume`.

If all solvers and clipping recovery fail for a CI draw, MD-Cat draws a
replacement, up to ten total draws per requested sample. A warning reports
`X/N` samples that needed replacement and the total replacement draws, noting
that this may bias CI. Persistent failure stops CI; the fitted tree and
pre-CI checkpoint remain available. Solver fallback alone does not redraw.

## Sparse optimization storage

Calibration equations are stored in CSR format, and diagonal matrices in the
fitting and CI objectives use sparse storage. This preserves the mathematical
problem and acceptance tolerances while avoiding dense quadratic allocations.
The conic formulation explicitly retains the dense formulation's diagonal
normalization to avoid changes caused by sparse factorization.
Constraint storage scales with the number of nonzero path coefficients; it is
not guaranteed linear for every tree shape and calibration arrangement.

CI checkpoints use schema 2 to store sparse constraints. This version also reads
older schema-1 checkpoints; older versions cannot read the new schema. CVXPY
problems are still constructed for each solve; parameterized problem reuse is a
separate potential optimization.

## Time and CI endpoint conventions

Node-time annotations use the requested output coordinates. Forward times and
calendar dates increase toward the present; with `-b`, positive ages increase
backward from the present. In all cases, `t_lower` and `t_upper` are ordered by
the reported values. Thus a backward-age interval of 2–10 is written
`t_lower=2,t_upper=10`: younger bound first, older bound second. Pooled
`height_CI={2,10}` follows the same numerical ordering.

Before version 1.1.12, individual backward-time CI outputs negated the endpoints
without swapping their labels, producing `t_lower=10,t_upper=2` in this example.
Version 1.1.12 corrects those annotations, including output regenerated from CI
checkpoints. Existing files are unchanged. The correction does not alter fitted
ages, rate estimates, CI replicate trees, branch-duration or rate intervals,
forward/date annotations, or pooled summary intervals.
