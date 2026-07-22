# `databases/` — research archive (not installed)

This directory holds large CSV masters and a small filter helper used in
historical data curation workflows:

- `properties_master.csv` — property table
- `descriptors_master.csv` — descriptor table (~64 MB)
- `filter_property.py` — filtering helper

## Install status

**These files are not part of the ECNet wheel or sdist.** Packaging excludes
`databases*` from the installable package. Installing from PyPI or
`pip install -e .` does **not** require this directory.

Bundled property sets used by `ecnet.datasets.load_*` live under
`src/ecnet/datasets/data/` (`.smiles` / `.target` pairs) and are documented in
the Sphinx page *Bundled property datasets* (`docs/source/data.rst`).

## Intended use

Treat `databases/` as a **repository research archive** for maintainers who
clone the full git tree. Downstream users who only need prediction and the
bundled loaders can ignore this directory.
