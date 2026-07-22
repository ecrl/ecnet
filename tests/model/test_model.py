"""Tests for ECNet model construct, fit, and save/load."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from ecnet import ECNet
from ecnet.datasets.structs import QSPRDatasetFromValues
from ecnet.model import load_model

_SEED = 0
_INPUT_DIM = 4
_EPOCHS = 5


def _synthetic_dataset() -> QSPRDatasetFromValues:
    desc_vals = [
        [0.0, 0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
    ]
    target_vals = [[1.0], [2.0], [3.0], [4.0]]
    return QSPRDatasetFromValues(desc_vals, target_vals)


def test_model_construct() -> None:
    input_dim = 3
    output_dim = 1
    hidden_dim = 5
    n_hidden = 2
    net = ECNet(input_dim, output_dim, hidden_dim, n_hidden)
    assert len(net.model) == 2 + n_hidden
    assert net.model[0].in_features == input_dim
    assert net.model[0].out_features == hidden_dim
    assert net.model[-1].in_features == hidden_dim
    assert net.model[-1].out_features == output_dim
    for layer in net.model[1:-1]:
        assert layer.in_features == hidden_dim
        assert layer.out_features == hidden_dim


def test_model_fit_seeded_finite_losses() -> None:
    torch.manual_seed(_SEED)
    ds = _synthetic_dataset()
    net = ECNet(_INPUT_DIM, 1, 16, 1)
    tr_loss, val_loss = net.fit(
        dataset=ds,
        epochs=_EPOCHS,
        batch_size=2,
        random_state=_SEED,
        shuffle=False,
    )
    assert len(tr_loss) == len(val_loss) == _EPOCHS
    assert all(math.isfinite(float(v)) for v in tr_loss)
    # valid_size default 0.0 → placeholder zeros
    assert all(float(v) == 0.0 for v in val_loss)


def _trained_net() -> tuple[ECNet, QSPRDatasetFromValues]:
    torch.manual_seed(_SEED)
    ds = _synthetic_dataset()
    net = ECNet(_INPUT_DIM, 1, 16, 1)
    net.fit(
        dataset=ds,
        epochs=_EPOCHS,
        batch_size=2,
        random_state=_SEED,
        shuffle=False,
    )
    return net, ds


def test_model_save_load_roundtrip_state_dict(tmp_path: Path) -> None:
    net, ds = _trained_net()

    with pytest.raises(ValueError):
        net.save(str(tmp_path / "model.badext"))

    model_path = tmp_path / "model.pt"
    net.save(str(model_path))
    assert model_path.is_file()

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    assert payload["format"] == "ecnet-state-v1"

    net.eval()
    x = ds[0]["desc_vals"]
    val_0 = net(x)

    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "missing.pt"))

    loaded = load_model(str(model_path))
    loaded.eval()
    assert torch.equal(val_0, loaded(x))


def test_model_load_legacy_full_module_pickle(tmp_path: Path) -> None:
    net, ds = _trained_net()
    legacy_path = tmp_path / "legacy.pt"
    # Simulate pre-shim checkpoints that pickled the whole module.
    torch.save(net, legacy_path)

    net.eval()
    x = ds[0]["desc_vals"]
    expected = net(x)

    loaded = load_model(str(legacy_path))
    loaded.eval()
    assert isinstance(loaded, ECNet)
    assert torch.equal(expected, loaded(x))


def test_model_load_unrecognized_payload_raises(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.pt"
    torch.save({"format": "not-ecnet"}, bad_path)
    with pytest.raises(ValueError, match="Unrecognized ECNet checkpoint"):
        load_model(str(bad_path))
