"""Tests for bundled property file loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

from ecnet.datasets.load_data import (
    _DATA_PATH,
    _get_file_data,
    _get_prop_paths,
    _open_smiles_file,
    _open_target_file,
)


def test_open_smiles_file(tmp_path: Path) -> None:
    smiles_text = "CCC\nCCCC\nCCCCC"
    smiles_path = tmp_path / "sample.smiles"
    smiles_path.write_text(smiles_text)
    smiles = smiles_text.split("\n")
    opened_smiles = _open_smiles_file(str(smiles_path))
    assert len(smiles) == len(opened_smiles)
    for i in range(len(smiles)):
        assert smiles[i] == opened_smiles[i]


def test_open_target_file(tmp_path: Path) -> None:
    target_text = "3.0\n4.0\n5.0"
    target_path = tmp_path / "sample.target"
    target_path.write_text(target_text)
    target_vals = [[float(v)] for v in target_text.split("\n")]
    opened_targets = _open_target_file(str(target_path))
    assert len(target_vals) == len(opened_targets)
    for i in range(len(target_vals)):
        assert target_vals[i] == opened_targets[i]


def test_get_prop_paths(props: list[str]) -> None:
    for p in props:
        smiles_fn, target_fn = _get_prop_paths(p)
        assert os.path.join(_DATA_PATH, f"{p}.smiles") == smiles_fn
        assert os.path.join(_DATA_PATH, f"{p}.target") == target_fn


def test_get_file_data(props: list[str]) -> None:
    for p in props:
        smiles, targets = _get_file_data(p)
        assert len(smiles) == len(targets)
        assert type(smiles[0]) is str
        assert type(targets[0]) is list
        assert type(targets[0][0]) is float
