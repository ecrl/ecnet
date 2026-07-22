Quickstart
==========

The examples below use synthetic descriptor values so they run without calling
PaDEL. For real QSPR workflows, construct a dataset from SMILES (or use a
bundled ``load_*`` helper) with the default ``backend='padel'``.

Train a small model
-------------------

.. code-block:: python

   from ecnet import ECNet
   from ecnet.datasets import QSPRDatasetFromValues

   desc_vals = [
       [0.0, 0.1, 0.2, 0.3],
       [0.1, 0.2, 0.3, 0.4],
       [0.2, 0.3, 0.4, 0.5],
       [0.3, 0.4, 0.5, 0.6],
   ]
   target_vals = [[1.0], [2.0], [3.0], [4.0]]
   dataset = QSPRDatasetFromValues(desc_vals, target_vals)

   model = ECNet(input_dim=4, output_dim=1, hidden_dim=16, n_hidden=1)
   train_loss, valid_loss = model.fit(
       dataset=dataset,
       epochs=10,
       batch_size=2,
       random_state=0,
   )

Save and reload
---------------

``ECNet.save`` writes an ``ecnet-state-v1`` checkpoint. ``load_model`` also
reads legacy full-module ``.pt`` pickles.

.. code-block:: python

   from ecnet.model import load_model

   model.save("example_model.pt")
   restored = load_model("example_model.pt")

Next steps
----------

- :doc:`api/index` — autodoc for the frozen public surface
- :doc:`stability` — compatibility and versioning policy
- :doc:`units` — cloud point (°C), kinematic viscosity (cSt), and scales
- Example notebooks under ``examples/`` on GitHub (PaDEL/Java; run manually —
  not executed in CI; see ``CONTRIBUTING.md``).
