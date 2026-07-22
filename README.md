[![UML Energy & Combustion Research Laboratory](https://sites.uml.edu/hunter-mack/files/2021/11/ECRL_final.png)](http://faculty.uml.edu/Hunter_Mack/)

# ECNet: machine learning models for fuel property prediction

[![GitHub version](https://badge.fury.io/gh/ecrl%2FECNet.svg)](https://badge.fury.io/gh/ecrl%2FECNet)
[![PyPI version](https://badge.fury.io/py/ecnet.svg)](https://badge.fury.io/py/ecnet)
[![status](https://joss.theoj.org/papers/10.21105/joss.00401/status.svg)](https://doi.org/10.21105/joss.00401)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/ECRL/ECNet/master/LICENSE.txt)
[![Documentation Status](https://readthedocs.org/projects/ecnet/badge/?version=latest)](https://ecnet.readthedocs.io/en/latest/?badge=latest)

**ECNet** is an open-source Python package for predicting fuel properties from
molecular structure using quantitative structure–property relationship (QSPR)
descriptors and multilayer perceptron models built with
[PyTorch](https://pytorch.org/).

The current **v4** API centers on `ECNet`, bundled property loaders
(`ecnet.datasets.load_*`), hyperparameter-tuning helpers, training callbacks,
and analytical blend-property equations. Descriptor backends include
[PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) (default) and
[alvaDesc](https://www.alvascience.com/alvadesc/) (optional; requires a valid
license).

## Installation

Requires **Python 3.11** or newer. Java is needed for the default PaDEL backend.

```bash
pip install ecnet
```

From a clone of this repository:

```bash
pip install -e .
pip install -e ".[dev]"   # pytest, ruff, pre-commit, pip-audit
pip install -e ".[docs]"  # Sphinx + Furo
```

## Documentation and examples

- User guide and API reference: [ecnet.readthedocs.io](https://ecnet.readthedocs.io/en/latest/)
- Example notebooks: [`examples/`](https://github.com/ecrl/ecnet/tree/master/examples)
- Stability policy: Sphinx *API stability* page (source: `docs/source/stability.rst`)
- Bundled dataset cards: Sphinx *Bundled property datasets* page (`docs/source/data.rst`)

## Historical JOSS architecture note

The 2017 Journal of Open Source Software article
([doi:10.21105/joss.00401](https://doi.org/10.21105/joss.00401)) and the
accompanying `paper/paper.md` describe a **prior generation** of ECNet based on
a project / build / node ensemble workflow. That architecture is **not** the
current public API. For v4 usage, follow the Sphinx documentation and the
imports documented under `ecnet`, `ecnet.datasets`, `ecnet.tasks`,
`ecnet.blends`, and `ecnet.callbacks`. The JOSS paper remains an appropriate
citation for the software’s publication history.

## Citation

If you use ECNet in scholarly work, please cite:

Kessler, T., & Mack, J. H. (2017). ECNet: Large scale machine learning projects
for fuel property prediction. *Journal of Open Source Software*, 2(17), 401.
https://doi.org/10.21105/joss.00401

```bibtex
@article{Kessler2017,
  doi = {10.21105/joss.00401},
  url = {https://doi.org/10.21105/joss.00401},
  year = {2017},
  publisher = {The Open Journal},
  volume = {2},
  number = {17},
  pages = {401},
  author = {Kessler, Travis and Mack, John Hunter},
  title = {ECNet: Large scale machine learning projects for fuel property prediction},
  journal = {Journal of Open Source Software}
}
```

## Contributing and support

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for local development setup, hooks, and
checks. Report bugs and feature requests via GitHub issues (include OS, Python
version, and relevant error output).

Contact: Travis Kessler (<travis.j.kessler@gmail.com>) and John Hunter Mack
(<Hunter_Mack@uml.edu>).
