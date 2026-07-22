API stability policy
====================

This page records the **frozen public API** for the ``4.1.x`` / ``4.2.x``
compatibility series and the versioning rules that govern change. It
summarizes design ``docs/design/DESIGN.md`` §8.1 and §8.4. Characterization
tests under ``tests/`` are the executable contract.

Frozen public surface
---------------------

Downstream code may rely on the following import paths and names remaining
available with compatible signatures and defaults:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Public names
   * - ``ecnet``
     - ``ECNet``, ``__version__``
   * - ``ecnet.model``
     - ``load_model``
   * - ``ecnet.datasets``
     - ``load_bp``, ``load_cn``, ``load_cp``, ``load_kv``, ``load_lhv``,
       ``load_mon``, ``load_mp``, ``load_pp``, ``load_ron``, ``load_ysi``,
       ``QSPRDataset``, ``QSPRDatasetFromFile``, ``QSPRDatasetFromValues``
   * - ``ecnet.tasks``
     - ``select_rfr``, ``tune_batch_size``, ``tune_model_architecture``,
       ``tune_training_parameters``
   * - ``ecnet.blends``
     - ``cetane_number``, ``yield_sooting_index``, ``kinematic_viscosity``,
       ``cloud_point``, ``lower_heating_value``, ``linear_blend_err``,
       ``exponential_blend_err``, ``kv_error``
   * - ``ecnet.callbacks``
     - ``LRDecayLinear``, ``Validator``, ``Callback``, ``CallbackOperator``

Advanced import: ``PCADataset``
-------------------------------

``PCADataset`` lives in ``ecnet.datasets.structs`` and is **not** part of the
``ecnet.datasets`` public export surface for this compatibility series. Use:

.. code-block:: python

   from ecnet.datasets.structs import PCADataset

Additive optional keyword arguments on frozen callables are allowed when
defaults preserve current behavior.

Versioning policy
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Release class
     - Examples
     - Contract
   * - Patch (``4.1.x``)
     - Bugfixes, packaging, tests, documentation corrections
     - Signature and numeric oracles must continue to match
   * - Minor (``4.2.0``)
     - Internal hardening, Sphinx/governance completion, dependency range
       expansions proven on CI
     - Still API-compatible with the frozen surface
   * - Major (``5.0.0``)
     - Intentional breaks only with explicit approval
     - Examples: removing pickle-based full-module ``torch.load``, changing
       default descriptor backends, altering bundled dataset schemas

Executable contract
-------------------

CI and local verification should keep these green:

* Signature locks — ``tests/test_api_signatures.py``
* Blend oracles — ``tests/blends/``, ``tests/fixtures/``
* Dataset / model / tasks / callbacks characterization suites

Coverage floors: ``ecnet.blends`` ≥95% line coverage; overall package ≥90%,
enforced in CI with ``--cov-fail-under=90``.
