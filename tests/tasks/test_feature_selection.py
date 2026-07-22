"""Tests for random-forest feature selection."""

from __future__ import annotations

from ecnet.datasets.structs import QSPRDatasetFromValues
from ecnet.tasks.feature_selection import select_rfr


def _synthetic_dataset(
    n_samples: int = 8, n_features: int = 20
) -> QSPRDatasetFromValues:
    desc_vals = [
        [float((i + 1) * (j + 1) % 7) for j in range(n_features)]
        for i in range(n_samples)
    ]
    target_vals = [[float(i)] for i in range(n_samples)]
    return QSPRDatasetFromValues(desc_vals, target_vals)


def test_select_rfr_structure_and_ordering() -> None:
    ds = _synthetic_dataset()
    n_features = len(ds.desc_vals[0])
    indices, importances = select_rfr(ds, total_importance=0.90, random_state=0)

    assert isinstance(indices, list)
    assert isinstance(importances, list)
    assert len(indices) == len(importances)
    # Implementation slices with exclusive cutoff, so the selected set is a
    # proper subset of all features when importances are non-degenerate.
    assert len(indices) < n_features
    assert importances == sorted(importances, reverse=True)
    for index in indices:
        assert 0 <= index < n_features
