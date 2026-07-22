"""Tests for ABC-based hyperparameter tuning helpers."""

from __future__ import annotations

from ecnet.datasets.structs import QSPRDatasetFromValues
from ecnet.tasks.parameter_tuning import (
    CONFIG,
    tune_batch_size,
    tune_model_architecture,
    tune_training_parameters,
)


def _train_eval_datasets() -> tuple[QSPRDatasetFromValues, QSPRDatasetFromValues]:
    desc_train = [
        [0.0, 0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
        [0.4, 0.5, 0.6, 0.7],
        [0.5, 0.6, 0.7, 0.8],
    ]
    target_train = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    desc_eval = [
        [0.15, 0.25, 0.35, 0.45],
        [0.55, 0.65, 0.75, 0.85],
    ]
    target_eval = [[2.5], [5.5]]
    return (
        QSPRDatasetFromValues(desc_train, target_train),
        QSPRDatasetFromValues(desc_eval, target_eval),
    )


def test_tune_batch_size_keys_and_bounds(n_processes: int) -> None:
    ds_train, ds_eval = _train_eval_datasets()
    res = tune_batch_size(1, 1, ds_train, ds_eval, n_processes, epochs=2, patience=2)
    assert set(res.keys()) == {"batch_size"}
    assert 1 <= res["batch_size"] <= len(ds_train.target_vals)


def test_tune_model_architecture_keys_and_bounds(n_processes: int) -> None:
    ds_train, ds_eval = _train_eval_datasets()
    res = tune_model_architecture(
        1, 1, ds_train, ds_eval, n_processes, epochs=2, patience=2
    )
    assert set(res.keys()) == {"hidden_dim", "n_hidden", "dropout"}
    for key, (lo, hi) in CONFIG["architecture_params_range"].items():
        assert lo <= res[key] <= hi


def test_tune_training_parameters_keys_and_bounds(n_processes: int) -> None:
    ds_train, ds_eval = _train_eval_datasets()
    res = tune_training_parameters(
        1, 1, ds_train, ds_eval, n_processes, epochs=2, patience=2
    )
    assert set(res.keys()) == {"lr", "lr_decay"}
    for key, (lo, hi) in CONFIG["training_params_range"].items():
        assert lo <= res[key] <= hi
