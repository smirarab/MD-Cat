# Changelog

## 1.1.13

- This is misc. scalability fixes. 
- Reuse unchanged branch observations, their weights, and the uniform rate
  distribution across CI draws, avoiding redundant allocations while preserving
  random draw order, objectives, and solver fallback/redraw behavior.
- Compute RTT initialization in one postorder pass using centered regression
  statistics, preserving the rate floor and unweighted average of clade slopes.
- Replace recursive calibration Euler traversal with an explicit stack so
  calibration lookup works on deeply unbalanced trees.

## 1.1.12 (important convention change)

- This is a convention change. 
- Order backward-time node-age CI annotations numerically: `t_lower` is the
  younger age and `t_upper` the older age. Previously these labels were reversed
  in individual `-b` CI outputs. Scripts relying on that ordering must adjust.
  Fitted values, CI samples, forward/date output, rate and branch-duration CIs,
  and pooled summary intervals are unchanged. Existing files are not rewritten.

## 1.1.11 

- This is a scalability release.
- Construct calibration constraints directly as CSR matrices and use sparse
  diagonals in fitting and CI, eliminating unnecessary dense quadratic storage.
  Preserve the dense conic formulation's scaling for MOSEK/CVXOPT/ECOS,
  avoiding numerical drift from a direct sparse quadratic-form substitution.
  Mathematical objectives, constraints, and solver tolerances are unchanged.
- Save sparse constraints in schema-2 CI checkpoints; continue reading schema-1
  dense checkpoints. Older MD-Cat versions cannot read schema-2 checkpoints.
- Compare dense/sparse fits and CI samples with OSQP, CVXOPT, Clarabel, and
  licensed MOSEK, including the astralpro.l6p1 calibration cases.

## 1.1.10

- Try alternative solvers and clipping recovery on the same CI draw first.
  If all fail, redraw up to ten total attempts per sample, and warn how many
  samples required replacement and that this may bias CI. Share numerical
  acceptance checks between fitting and CI; persistent failures stop explicitly.
- Allow tiny-negative clipping only after exhausting all solvers, with calibration
  checks before and after clipping; do not reject nonnegative durations below the
  optimization minimum.
- Add `--solver-tolerances CALIB_ATOL CALIB_RTOL NEGATIVE_ATOL` (defaults:
  `1e-7 1e-7 1e-8`). Save values in CI checkpoints and sampled-run manifests;
  CI resume preserves saved values unless explicitly overridden.

## 1.1.9

- Accept the final EM update and recompute its posterior probabilities before
  stopping on convergence. Returned durations, rates, likelihood, and posteriors
  now describe the same iterate, including first-iteration convergence.
  Fitted estimates, restart selection, and confidence intervals can change.
- Test consistency and calibration constraints at convergence and iteration-limit
  exits, including a forced first-iteration convergence case.

## 1.1.8 (default output format change)

- Write time and requested rate/probability annotations for runs without
  `--CI`, honoring all three `--annotate` levels and the default level 2.
  Annotations use restored time/rate units and do not change fitted results.
- Add `--annotate 0` to write an unannotated tree, reproducing older non-CI
  output. CI checkpoints retain this setting; the default remains level 2.


## 1.1.7 (bug fix)

- Previously, some internal node constraints were omitted from the optimizer's equality
  constraints when some tips had sampling times and others didn't. 
  Fossil runs with default tip times (`-b`) or runs with `-f` and trees with all
  tips given sampling times were not affected.
- Add exhaustive calibration-subset coverage and exact before/after checks
  for fossil, fully tip-sampled, and fully node-sampled fits and CI samples.

## 1.1.6

- Build categorical CDFs from the probabilities in sorted rate order. This
  corrects sampling and quantiles for unsorted categories with unequal weights.
  It can change branch-rate CI annotations and custom simulations, but leaves
  dating estimates and uniformly sampled time-CI replicates unchanged.
- Add permutation and exact before/after regression tests for fitted dates,
  rates, likelihoods, time intervals, and exported CI replicate trees.

## 1.1.5

- Normalize calibration times by their span before rate initialization and EM
  fitting. Rate floors and convergence thresholds now use these normalized
  coordinates, so changing calibration units does not change the normalized fit
  when `--min-branch` is scaled with them.
- Solve CIs in the same coordinates and restore original times, durations, and
  rates in fitted trees, CI exports, and checkpoints. Preserve precision in time
  and rate annotations instead of rounding small converted values to zero.
- Save the time transform with new CI checkpoints; schema-1 checkpoints without
  it continue to resume in their original coordinates. Sampled-run manifests
  retain the exact-version check to prevent mixing old and new fits.
- Reject nonfinite calibration times or a zero calibration-time span with a
  clear error before optimization.

## 1.1.0

- Add the installed `md_cat_sample.py` workflow for treePL calibration bounds,
  with independent or bottom-up uniform/exponential sampling and permanent
  per-run inputs/results. Defaults: 100 calibration samples, bottom-up,
  exponential, and one job with one numerical thread.
- Add reproducible master/child seeds, subprocess parallelism, generated external
  job scripts, dry-run preparation, validated resume, and partial summarize-only.
- Pool exported CI replicates while computing central mean/median estimates from
  fitted run trees. Save complete run inclusion/exclusion and clipping records.
- Add `md_cat_summarize.py`, NEXUS/FigTree percentile interval metadata, and
  annotated, treePL-style, and clean Newick output modes.
- Expose a shared positive `--min-branch` duration (default 0.001) through
  calibration sampling, fitting, initialization, and CI optimization. Preserve
  full-precision time branch lengths rather than rounding small durations away.
- Keep the existing fixed-calibration `md_cat.py` command. Its CLI is now a
  package entry point sharing option definitions with the sampling workflow.
