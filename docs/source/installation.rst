Installation
============

Requires **Python 3.11** or newer. Java is required for the default PaDEL
descriptor backend. A licensed alvaDesc installation is required only when
using ``backend='alvadesc'``. The alvaDesc Python wrapper still imports
``pkg_resources``; if that import fails under newer setuptools, either use
PaDEL (``backend='padel'``, the default) or install ``setuptools<82``.

Install from PyPI
-----------------

.. code-block:: bash

   pip install ecnet

Runtime dependencies (``torch``, ``scikit-learn``, ``padelpy``, ``alvadescpy``,
``ecabc``) are installed with the package. For custom PyTorch builds (for
example GPU wheels), install those first, then install ECNet.

Upgrade
-------

.. code-block:: bash

   pip install --upgrade ecnet

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/ecrl/ecnet.git
   cd ecnet
   pip install -e .

Development extras
------------------

.. code-block:: bash

   pip install -e ".[dev]"    # tests, ruff, pre-commit, pip-audit
   pip install -e ".[docs]"   # Sphinx + Furo and related tools

See the repository ``CONTRIBUTING.md`` for local hooks and checks.
