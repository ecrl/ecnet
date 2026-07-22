"""Tests for dataset descriptor utilities."""

from ecnet.datasets.utils import _qspr_from_padel


def test_dataset_utils(n_desc: int) -> None:
    smiles = ["CCC", "CCCC", "CCCCC"]
    desc, keys = _qspr_from_padel(smiles)
    assert len(keys) == n_desc
    assert len(desc) == 3
    for d in desc:
        assert len(d) == n_desc
