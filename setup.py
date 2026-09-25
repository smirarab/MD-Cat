"""Package metadata; build with python -m build."""
from pathlib import Path
from setuptools import setup

ROOT = Path(__file__).parent
metadata = {}
exec((ROOT / "emd" / "__init__.py").read_text(encoding="utf-8"), metadata)

setup(
    name="mdcat-date",
    version=metadata["PROGRAM_VERSION"],
    description=metadata["PROGRAM_DESCRIPTION"],
    author=", ".join(metadata["PROGRAM_AUTHOR"]),
    license=metadata["PROGRAM_LICENSE"],
    license_files=["LICENSE"],
    url="https://github.com/uym2/MD-Cat",
    project_urls={"Issues": "https://github.com/uym2/MD-Cat/issues"},
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=["emd", "simulator"],
    include_package_data=False,
    scripts=["simulate.py"],
    entry_points={"console_scripts": [
        "md_cat.py=emd.cli:main",
        "md_cat_sample.py=emd.sample:main",
        "md_cat_summarize.py=emd.summary:main",
    ]},
    install_requires=[
        "treeswift", "scipy>=1.6", "bitsets", "numpy>=1.18.5",
        "jenkspy", "mosek", "cvxpy", "cvxopt", "osqp",
    ],
    extras_require={"dev": ["build", "twine"]},
    keywords="Phylogenetics Evolution Biology",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
