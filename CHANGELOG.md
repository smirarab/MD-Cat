# Changelog

## 1.1.7 (bug fix)

- Previously some internal node constraints were omitted from the optimizer's equality
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
