"""Smoke tests for public ``load_*`` property loaders (design §8.1–§8.2).

Default backend is ``padel`` only; alvaDesc is not exercised in CI (Q10).
"""

from __future__ import annotations

import pytest

from ecnet.datasets import (
    QSPRDatasetFromFile,
    load_bp,
    load_cn,
    load_cp,
    load_kv,
    load_lhv,
    load_mon,
    load_mp,
    load_pp,
    load_ron,
    load_ysi,
)
from ecnet.datasets import load_data as load_data_mod

_LOADERS = [
    (load_bp, "bp"),
    (load_cn, "cn"),
    (load_cp, "cp"),
    (load_kv, "kv"),
    (load_lhv, "lhv"),
    (load_mon, "mon"),
    (load_mp, "mp"),
    (load_pp, "pp"),
    (load_ron, "ron"),
    (load_ysi, "ysi"),
]


@pytest.mark.parametrize(
    ("loader", "prop"),
    _LOADERS,
    ids=[prop for _, prop in _LOADERS],
)
def test_load_prop_as_tuple(loader, prop: str) -> None:
    smiles, targets = loader(as_dataset=False)
    assert len(smiles) == len(targets)
    assert len(smiles) > 0
    assert type(smiles[0]) is str
    assert type(targets[0]) is list
    assert type(targets[0][0]) is float


@pytest.mark.parametrize(
    ("loader", "prop"),
    _LOADERS,
    ids=[prop for _, prop in _LOADERS],
)
def test_load_prop_as_dataset_routes_to_load_set(
    loader, prop: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All ten loaders must call ``_load_set(prop, 'padel')`` by default."""
    calls: list[tuple[str, str]] = []
    sentinel = object()

    def _fake_load_set(p: str, backend: str):
        calls.append((p, backend))
        return sentinel

    monkeypatch.setattr(load_data_mod, "_load_set", _fake_load_set)
    result = loader(as_dataset=True)
    assert result is sentinel
    assert calls == [(prop, "padel")]


@pytest.mark.integration
def test_load_pp_as_dataset_padel_smoke(n_desc: int) -> None:
    """Real PaDEL smoke on the smallest bundled set (pour point, 40 SMILES)."""
    ds = load_pp(as_dataset=True)
    assert isinstance(ds, QSPRDatasetFromFile)
    assert len(ds.smiles) == len(ds.target_vals)
    assert len(ds.smiles) > 0
    assert len(ds.desc_vals) == len(ds.smiles)
    assert len(ds.desc_vals[0]) == n_desc
    assert len(ds.desc_names) == n_desc
