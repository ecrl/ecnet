"""PaDEL loaders must not import alvadescpy at package import time."""

from __future__ import annotations

import sys


def test_importing_datasets_does_not_import_alvadescpy() -> None:
    for name in list(sys.modules):
        if name == "alvadescpy" or name.startswith("alvadescpy."):
            del sys.modules[name]
        if name == "ecnet.datasets" or name.startswith("ecnet.datasets."):
            del sys.modules[name]

    import ecnet.datasets  # noqa: F401
    from ecnet.datasets import load_cp  # noqa: F401

    assert "alvadescpy" not in sys.modules
