"""Tests for QSPR dataset structures."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ecnet.datasets.structs import (
    QSPRDataset,
    QSPRDatasetFromFile,
    QSPRDatasetFromValues,
)


def test_qsprdataset(backend: str, n_desc: int) -> None:
    smiles = ["CCC", "CCCC", "CCCCC"]
    targets = [[3.0], [4.0], [5.0]]
    ds = QSPRDataset(smiles, targets, backend=backend)
    assert len(ds.smiles) == len(smiles)
    assert len(ds.target_vals) == len(targets)
    assert len(ds.target_vals[0]) == len(targets[0])
    assert len(ds.desc_vals) == len(smiles)
    assert len(ds.desc_vals[0]) == n_desc
    assert isinstance(ds.desc_vals, torch.Tensor)
    assert len(ds.desc_names) == n_desc


def test_qsprdatasetfromfile(tmp_path: Path, backend: str, n_desc: int) -> None:
    smiles_text = "CCC\nCCCC\nCCCCC"
    smiles_path = tmp_path / "sample.smiles"
    smiles_path.write_text(smiles_text)
    smiles = smiles_text.split("\n")
    targets = [[3.0], [4.0], [5.0]]
    ds = QSPRDatasetFromFile(str(smiles_path), targets, backend=backend)
    assert len(ds.smiles) == len(smiles)
    assert len(ds.target_vals) == len(targets)
    assert len(ds.target_vals[0]) == len(targets[0])
    assert len(ds.desc_vals) == len(smiles)
    assert len(ds.desc_vals[0]) == n_desc
    assert isinstance(ds.desc_vals, torch.Tensor)
    assert len(ds.desc_names) == n_desc


def test_qsprdatasetfromvalues() -> None:
    desc_vals = [
        [0.0, 0.1, 0.2, 0.3],
        [0.0, 0.2, 0.3, 0.1],
        [0.1, 0.3, 0.0, 0.2],
    ]
    target_vals = [[1.0], [2.0], [3.0]]
    ds = QSPRDatasetFromValues(desc_vals, target_vals)
    assert len(ds.smiles) == len(desc_vals)
    assert len(ds.desc_names) == len(desc_vals[0])
    assert len(ds.desc_vals) == len(desc_vals)
    assert len(ds.target_vals) == len(target_vals)
    assert len(ds.target_vals[0]) == len(target_vals[0])
    assert isinstance(ds.desc_vals, torch.Tensor)
    assert isinstance(ds.target_vals, torch.Tensor)


def test_qsprdataset_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        QSPRDataset(["CCC"], [[1.0]], backend="not-a-backend")


def test_qsprdataset_set_index_and_set_desc_index() -> None:
    desc_vals = [
        [0.0, 0.1, 0.2, 0.3],
        [1.0, 1.1, 1.2, 1.3],
        [2.0, 2.1, 2.2, 2.3],
    ]
    target_vals = [[10.0], [20.0], [30.0]]
    ds = QSPRDatasetFromValues(desc_vals, target_vals)
    ds.set_index([0, 2])
    assert len(ds) == 2
    assert ds.target_vals[0].tolist() == [10.0]
    assert ds.target_vals[1].tolist() == [30.0]
    assert ds.desc_vals.shape[0] == 2

    ds.set_desc_index([1, 3])
    assert ds.desc_vals.shape[1] == 2
    assert len(ds.desc_names) == 2
    assert ds.desc_vals[0].tolist() == pytest.approx([0.1, 0.3])
