# Changelog

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
