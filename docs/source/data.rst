Bundled property datasets
=========================

ECNet ships curated SMILES/target pairs as package data under
``ecnet/datasets/data/``. Load them with the ``ecnet.datasets.load_*`` helpers
(see :doc:`api/datasets`). Units conventions for blend helpers are summarized
in :doc:`units`.

These cards describe what the package actually distributes. Compound-level
literature provenance is **not** encoded in the bundled ``.smiles`` /
``.target`` files; do not treat compound counts or property labels as a
substitute for primary citations.

Common attributes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Value
   * - Format
     - Parallel UTF-8 text files: one SMILES string per line (``*.smiles``) and
       one numeric target per line (``*.target``), same row order and length
   * - Package path
     - ``src/ecnet/datasets/data/`` (installed as package data)
   * - Access
     - ``from ecnet.datasets import load_<prop>``; optional
       ``as_dataset=True`` returns ``QSPRDatasetFromFile``
   * - Default descriptors
     - Built at load/train time via PaDEL (``backend='padel'``); alvaDesc is
       optional and licensed separately
   * - License
     - Distributed with ECNet under the project MIT license
   * - Limitations
     - No silent unit conversion in ``ECNet.fit``; descriptor backends may be
       slow or unavailable without Java (PaDEL) or an alvaDesc license

Property cards
--------------

Counts below are the number of lines in the bundled ``.smiles`` files at the
time of documentation (one compound per line).

Boiling point (``bp``)
~~~~~~~~~~~~~~~~~~~~~~

* **Loader:** ``load_bp``
* **Files:** ``bp.smiles``, ``bp.target``
* **Size:** 205 compounds
* **Target unit:** °C
* **Scope:** Boiling-point regression targets paired with SMILES

Cetane number (``cn``)
~~~~~~~~~~~~~~~~~~~~~~

* **Loader:** ``load_cn``
* **Files:** ``cn.smiles``, ``cn.target``
* **Size:** 460 compounds
* **Target unit:** Cetane number (dimensionless scale as stored)
* **Scope:** Ignition-quality targets for fuel-like structures

Cloud point (``cp``)
~~~~~~~~~~~~~~~~~~~~

* **Loader:** ``load_cp``
* **Files:** ``cp.smiles``, ``cp.target``
* **Size:** 43 compounds
* **Target unit:** °C
* **Scope:** Cloud-point targets; blend I/O for ``cloud_point`` is also °C
  (:doc:`units`)

Kinematic viscosity (``kv``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Loader:** ``load_kv``
* **Files:** ``kv.smiles``, ``kv.target``
* **Size:** 213 compounds
* **Target unit:** mm²/s (cSt) at 313 K
* **Scope:** Viscosity targets for the KV blend helper (cSt)

Lower heating value (``lhv``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Loader:** ``load_lhv``
* **Files:** ``lhv.smiles``, ``lhv.target``
* **Size:** 388 compounds
* **Target unit:** MJ/kg (= kJ/g as documented in the loader)
* **Scope:** Heating-value regression targets

Motor octane number (``mon``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Loader:** ``load_mon``
* **Files:** ``mon.smiles``, ``mon.target``
* **Size:** 308 compounds
* **Target unit:** MON (dimensionless scale as stored)
* **Scope:** Motor octane targets

Melting point (``mp``)
~~~~~~~~~~~~~~~~~~~~~~

* **Loader:** ``load_mp``
* **Files:** ``mp.smiles``, ``mp.target``
* **Size:** 147 compounds
* **Target unit:** °C
* **Scope:** Melting-point regression targets

Pour point (``pp``)
~~~~~~~~~~~~~~~~~~~

* **Loader:** ``load_pp``
* **Files:** ``pp.smiles``, ``pp.target``
* **Size:** 41 compounds
* **Target unit:** °C
* **Scope:** Pour-point regression targets (smallest bundled set)

Research octane number (``ron``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Loader:** ``load_ron``
* **Files:** ``ron.smiles``, ``ron.target``
* **Size:** 308 compounds
* **Target unit:** RON (dimensionless scale as stored)
* **Scope:** Research octane targets

Yield sooting index (``ysi``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Loader:** ``load_ysi``
* **Files:** ``ysi.smiles``, ``ysi.target``
* **Size:** 558 compounds
* **Target unit:** Unified YSI scale as stored
* **Scope:** Sooting-tendency targets

Repository research archive (``databases/``)
--------------------------------------------

The top-level ``databases/`` directory (CSV masters and a filter script) is a
**non-install research archive**. It is excluded from wheels and sdists and is
not required to use the installed package. See ``databases/README.md`` in the
source repository.
