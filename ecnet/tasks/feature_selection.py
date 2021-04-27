r"""Feature selection functions"""
from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor

from ..datasets.structs import QSPRDataset


def select_rfr(dataset: QSPRDataset, total_importance: float = 0.95,
               **kwargs) -> Tuple[List[int], List[float]]:
    """
    select_rfr: reduces input data dimensionality such that specified proportion of total feature
    importance (derived from random forest regression) is retained in feature subset

    Args:
        dataset (QSPRDataset): input data
        total_importance (float): total feature importance to retain
        **kwargs: additional arguments passed to sklearn.ensemble.RandomForestRegressor

    Returns:
        tuple[list[int], list[float]]: (selected feature indices, selected feature importances)
    """

    X = dataset.desc_vals
    y = [dv[0] for dv in dataset.target_vals]
    regr = RandomForestRegressor(**kwargs)
    regr.fit(X, y)
    importances = sorted(
        [(regr.feature_importances_[i], i)
         for i in range(len(dataset.desc_vals[0]))],
        key=lambda x: x[0], reverse=True
    )
    tot_imp = 0.0
    for idx, i in enumerate(importances):
        tot_imp += i[0]
        idx_cutoff = idx
        if tot_imp >= total_importance:
            break
    desc_imp = [i[0] for i in importances][:idx_cutoff]
    desc_idx = [i[1] for i in importances][:idx_cutoff]
    return (desc_idx, desc_imp)
