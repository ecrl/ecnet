Units and numeric conventions
=============================

Canonical units for the public API (design §8.3). ``ECNet.fit`` does **not**
perform silent unit conversion on model targets; values are used as supplied
by the caller or by bundled ``.target`` files.

Blend properties
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Quantity
     - Canonical unit
     - Notes
   * - Cloud point (blend I/O)
     - °C
     - Internal Rankine conversion remains an implementation detail; public
       inputs and outputs stay in °C.
   * - Kinematic viscosity
     - cSt
     - Mixing rule follows Ding et al. as implemented in
       ``ecnet.blends.kinematic_viscosity``.
   * - Cetane number, YSI, LHV, octane numbers (RON/MON)
     - Dimensionless property scales
     - Same scales as the bundled target files; see :doc:`data` for per-property
       cards and known provenance limits.

Model targets
-------------

Regression targets for ``ECNet`` follow the units of the supplied dataset.
Bundled loaders return the scales stored in package data without converting
temperature, viscosity, or other engineering units inside the model.
