"""Tests for training callbacks."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader

from ecnet import ECNet
from ecnet.callbacks import Callback, CallbackOperator, LRDecayLinear, Validator
from ecnet.datasets.structs import QSPRDatasetFromValues


class _HaltCallback(Callback):
    """Returns False from every hook so CallbackOperator short-circuits."""

    def on_train_begin(self):
        return False

    def on_train_end(self):
        return False

    def on_epoch_begin(self, epoch):
        return False

    def on_epoch_end(self, epoch):
        return False

    def on_batch_begin(self, batch):
        return False

    def on_batch_end(self, batch):
        return False

    def on_loss_begin(self, batch):
        return False

    def on_loss_end(self, batch):
        return False

    def on_step_begin(self, batch):
        return False

    def on_step_end(self, batch):
        return False


def test_callback_operator_short_circuits_on_false() -> None:
    op = CallbackOperator()
    op.add_cb(_HaltCallback())
    assert op.on_train_begin() is False
    assert op.on_train_end() is False
    assert op.on_epoch_begin(0) is False
    assert op.on_epoch_end(0) is False
    assert op.on_batch_begin(0) is False
    assert op.on_batch_end(0) is False
    assert op.on_loss_begin(0) is False
    assert op.on_loss_end(0) is False
    assert op.on_step_begin(0) is False
    assert op.on_step_end(0) is False


def test_lrlineardecay() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1),
    )
    lr = 0.001
    lrd = 0.00001
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    linear_decay = LRDecayLinear(lr, lrd, optim)
    reached_epoch = 0
    for epoch in range(10000):
        if not linear_decay.on_epoch_begin(epoch):
            break
        reached_epoch += 1
    if reached_epoch > int(lr / lrd):
        raise RuntimeError(f"Linear decay: epoch reached {reached_epoch}")


def _synthetic_loader(n_samples: int = 4, n_features: int = 3) -> DataLoader:
    desc_vals = [[float(i + j) for j in range(n_features)] for i in range(n_samples)]
    target_vals = [[float(i)] for i in range(n_samples)]
    ds = QSPRDatasetFromValues(desc_vals, target_vals)
    return DataLoader(ds, batch_size=n_samples, shuffle=False)


def test_validator_patience_stops_when_loss_does_not_improve() -> None:
    """Non-improving validation loss must trip patience (strict ``<`` best)."""
    loader = _synthetic_loader()
    net = ECNet(input_dim=3, output_dim=1, hidden_dim=4, n_hidden=1)
    # Freeze weights so validation MSE is constant after the first eval.
    for param in net.parameters():
        param.requires_grad_(False)
    net.eval()

    patience = 2
    eval_iter = 1
    validator = Validator(loader, net, eval_iter=eval_iter, patience=patience)

    # epoch 0: establishes best_loss
    assert validator.on_epoch_end(0) is True
    assert validator._epoch_since_best == 0

    # epochs 1..patience: accumulate; still continue (``>`` not ``>=``)
    for epoch in range(1, patience + 1):
        assert validator.on_epoch_end(epoch) is True
    assert validator._epoch_since_best == patience

    # next eval: epoch_since_best > patience → halt
    assert validator.on_epoch_end(patience + 1) is False


def test_validator_on_train_end_restores_best_state() -> None:
    loader = _synthetic_loader()
    net = ECNet(input_dim=3, output_dim=1, hidden_dim=4, n_hidden=1)
    validator = Validator(loader, net, eval_iter=1, patience=2)

    assert validator.on_epoch_end(0) is True
    best = copy.deepcopy(validator.best_state)

    # Mutate weights after the best checkpoint was recorded.
    with torch.no_grad():
        for param in net.parameters():
            param.add_(1.0)

    mutated = {k: v.clone() for k, v in net.state_dict().items()}
    for key in best:
        assert not torch.equal(mutated[key], best[key])

    assert validator.on_train_end() is True
    restored = net.state_dict()
    for key in best:
        assert torch.equal(restored[key], best[key])


def test_ecnet_fit_validator_wiring_early_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``valid_size > 0`` wires Validator into ``fit`` and can halt training."""
    desc_vals = [
        [0.0, 0.1, 0.2],
        [0.2, 0.3, 0.4],
        [0.4, 0.5, 0.6],
        [0.6, 0.7, 0.8],
        [0.8, 0.9, 1.0],
        [1.0, 1.1, 1.2],
    ]
    target_vals = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    ds = QSPRDatasetFromValues(desc_vals, target_vals)
    net = ECNet(input_dim=3, output_dim=1, hidden_dim=8, n_hidden=1)

    # Keep real epoch-0 bookkeeping; force halt on the next eval so the fit
    # path is deterministic (patience unit tests cover non-improving loss).
    original = Validator.on_epoch_end

    def _halt_after_epoch_zero(self, epoch: int) -> bool:
        if epoch == 0:
            return original(self, epoch)
        return False

    monkeypatch.setattr(Validator, "on_epoch_end", _halt_after_epoch_zero)

    epochs = 20
    train_losses, valid_losses = net.fit(
        dataset=ds,
        epochs=epochs,
        batch_size=2,
        valid_size=0.5,
        valid_eval_iter=1,
        patience=16,
        random_state=0,
        shuffle=False,
    )

    assert len(train_losses) == len(valid_losses)
    assert len(train_losses) < epochs
    assert len(train_losses) == 2
    # Epoch-0 validation must be evaluated before losses are recorded.
    assert all(float(v) < 1e18 for v in valid_losses)
    assert all(float(v) == float(v) for v in valid_losses)


def test_ecnet_fit_records_finite_valid_loss_from_epoch_zero() -> None:
    """Verbose/history valid loss must not leak the unset-loss sentinel."""
    desc_vals = [
        [0.0, 0.1, 0.2],
        [0.2, 0.3, 0.4],
        [0.4, 0.5, 0.6],
        [0.6, 0.7, 0.8],
        [0.8, 0.9, 1.0],
        [1.0, 1.1, 1.2],
    ]
    target_vals = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    ds = QSPRDatasetFromValues(desc_vals, target_vals)
    net = ECNet(input_dim=3, output_dim=1, hidden_dim=8, n_hidden=1)
    _, valid_losses = net.fit(
        dataset=ds,
        epochs=3,
        batch_size=2,
        valid_size=0.5,
        valid_eval_iter=1,
        patience=16,
        random_state=0,
        shuffle=False,
    )
    assert len(valid_losses) == 3
    assert all(float(v) < 1e18 for v in valid_losses)
