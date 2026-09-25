# Development and releases
## Local development

Use Python 3.10 or newer (3.12 recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The PyPI distribution name is `mdcat-date`; the application name remains
MD-Cat. Installed commands are `md_cat.py`, `md_cat_sample.py`,
`md_cat_summarize.py`, and `simulate.py`. The dating and summary commands
are packaging entry points into `emd`; the repository-root `md_cat.py`
is a source-checkout compatibility launcher.
Package metadata lives in `setup.py`; the sole version definition is
`PROGRAM_VERSION` in `emd/__init__.py`. The build backend is declared in
`pyproject.toml`. Keep the dependency list aligned with imports and the solver
fallbacks. MOSEK is imported by the runtime, so its Python package is required,
but its license is not required for the OSQP/CVXOPT fallback.

The supported Python floor is now 3.10, replacing the historical 3.7 README
claim. The package workflow checks Python 3.10, 3.12, and 3.13 on Linux and macOS.

## Validate a candidate

Start from a clean checkout of the intended release commit, with no old
`build/`, `dist/`, or `*.egg-info/` artifacts. Do not include solver licenses or
local analysis outputs in a commit or distribution.

```bash
python -m build
python -m twine check --strict dist/*
```

The default build creates an sdist and then builds the wheel from that sdist.
In a second, fresh virtual environment, install the wheel using its absolute
path, change to a directory outside the checkout, and run:

```bash
python -m pip install /absolute/path/to/dist/mdcat_date-1.0.2-py3-none-any.whl
python -m pip check
md_cat.py -h
simulate.py -h
printf '(A:0.1,B:0.2);\n' > input.nwk
md_cat.py -i input.nwk -o output.nwk -k 2 -p 1 --maxIter 2 --randSeed 1 --threads 1 --CI '2 0.025 0.975'
```

Substitute the actual wheel filename/version. Check that `output.nwk` is
nonempty and readable. `.github/workflows/package.yml` automates builds,
metadata validation, installed CLI checks, histogram compatibility, and a small
dating run with confidence intervals outside the checkout. CI has no MOSEK
license and therefore exercises open-source solver fallback. Also run any
scientific regression tests relevant to algorithm changes before release.

## One-time publication setup

1. Use the `smirarab` account on [PyPI](https://pypi.org/) with two-factor
   authentication and ownership of the project `mdcat-date`. The original
   distribution name `MD-Cat` was taken. A pending publisher does not reserve
   a project name.
2. Under PyPI account **Publishing**, add a pending GitHub publisher:
   project `mdcat-date`, GitHub owner `uym2`, repository `MD-Cat`, workflow filename
   `release.yml`, environment `pypi`. If the project already exists under your
   control, add the publisher in that project's settings instead. Adjust the
   owner/repository if publishing from a different canonical repository. The
   GitHub owner identifies the repository running Actions, not the PyPI
   account: using PyPI account `smirarab` does not change `uym2`. If releasing
   from `smirarab/MD-Cat` on GitHub, use GitHub owner `smirarab` instead.
3. In GitHub repository settings, create the environment `pypi`. Configure
   required reviewers if you want a manual publication gate. Enable Actions.
4. Merge the packaging files and workflows before creating the release.

This uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/);
no long-lived PyPI token is needed. These account settings must be configured
by a maintainer; adding workflow files alone does not publish anything.

## Every substantial update

A push or merge does **not** update the package on PyPI. To make an update
available through pip:

1. Choose a new version and change `PROGRAM_VERSION` in `emd/__init__.py`.
   Use a patch increment for fixes, a minor increment for compatible features,
   and a major increment for breaking changes. The first prepared candidate is
   `1.0.2`; use it only if that version has not already been published.
2. Update README/help and prepare release notes describing user-visible changes,
   compatibility changes, and relevant validation. After the first successful
   publication, replace README's planned-publication wording with the actual
   PyPI installation status.
3. Run the validation above and relevant regression tests. Commit all intended
   changes, push, and require the package workflow to pass.
4. Create and publish a GitHub Release with tag `v<VERSION>` pointing to that
   exact commit (for example `v1.0.2`), and include the release notes. Publishing
   the release triggers `release.yml`; saving a draft does not.
5. The workflow rebuilds and tests the distributions, verifies that the tag
   matches the wheel version, then uploads those tested artifacts to PyPI.
   Approve the `pypi` environment if required by your repository configuration.
6. Check the workflow and PyPI release page. In a fresh environment, run
   `python -m pip install --index-url https://pypi.org/simple 'mdcat-date==1.0.2'`
   (substitute the version), then repeat the installed CLI smoke test.

PyPI files cannot be overwritten. Fixes after publication need a new version
and a new release; do not move an existing release tag to different code. If a
workflow fails before uploading, correct the cause and rerun as appropriate.
For a partial upload, inspect PyPI before retrying; do not blindly enable
`skip-existing`. Documentation-only edits do not require a package release
unless they must appear on the PyPI project page.

If Bioconda is added later, each new upstream version also needs a recipe
version/source checksum update and a passing Bioconda pull request; a PyPI
release alone does not guarantee immediate Bioconda availability.

## 1.1.0 validation

Run the focused tests with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The sampling tests cover seeded distributions, shared minimum durations,
process jobs, CI pooling, dry-run external execution, partial summaries,
resume, input checksums, and paths containing spaces. Package validation
must also run `md_cat_sample.py` outside the checkout after installing the
wheel; local `treepl_to_mdcat.py` and `summarize_mdcat.py` are not dependencies
and must not be included in distributions. See CHANGELOG.md for the release
summary. The installed workflow records a package version in every manifest.
