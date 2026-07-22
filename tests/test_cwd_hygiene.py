"""Regression: library save/load must not pollute the process CWD."""

from __future__ import annotations

from pathlib import Path

import torch

from ecnet import ECNet
from ecnet.datasets.structs import QSPRDatasetFromValues
from ecnet.model import load_model

_SEED = 0


def test_save_load_does_not_pollute_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    torch.manual_seed(_SEED)
    ds = QSPRDatasetFromValues(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
            [0.3, 0.4, 0.5, 0.6],
        ],
        [[1.0], [2.0], [3.0], [4.0]],
    )
    net = ECNet(4, 1, 8, 1)
    net.fit(
        dataset=ds,
        epochs=2,
        batch_size=2,
        random_state=_SEED,
        shuffle=False,
    )
    net.save("model.pt")
    loaded = load_model("model.pt")
    assert isinstance(loaded, ECNet)

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["model.pt"]
    banned = (
        "_temp.smiles",
        "_temp.target",
        "_test.pt",
    )
    for name in banned:
        assert not (tmp_path / name).exists()
