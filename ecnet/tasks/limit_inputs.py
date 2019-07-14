#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tasks/limit_inputs.py
# v.3.2.1
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions for selecting influential input parameters
#

# Stdlib. imports
from copy import deepcopy

# 3rd party imports
from sklearn.ensemble import RandomForestRegressor
from numpy import concatenate, ravel

# ECNet imports
from ecnet.utils.data_utils import DataFrame
from ecnet.utils.logging import logger


def limit_rforest(df: DataFrame, limit_num: int, num_estimators: int=None,
                  num_processes: int=1, **kwargs) -> list:
    '''Uses random forest regression to select input parameters

    Args:
        df (ecnet.utils.data_utils.DataFrame): loaded data
        limit_num (int): desired number of input parameters
        num_estimators (int): number of trees used by RFR algorithm
        num_processes (int): number of parallel jobs for RFR algorithm
        **kwargs: any argument accepted by
            sklearn.ensemble.RandomForestRegressor

    Returns:
        list: [(feature, importance), ..., (feature, importance)]
    '''

    logger.log('info', 'Finding {} most influential input parameters'
               .format(limit_num), call_loc='LIMIT')

    pd = df.package_sets()
    X = concatenate((pd.learn_x, pd.valid_x, pd.test_x))
    y = ravel(concatenate((pd.learn_y, pd.valid_y, pd.test_y)))

    if num_estimators is None:
        num_estimators = len(X[0])

    logger.log('debug', 'Number of estimators: {}'.format(num_estimators),
               call_loc='LIMIT')

    regr = RandomForestRegressor(
        n_jobs=num_processes,
        n_estimators=num_estimators,
        **kwargs
    )
    regr.fit(X, y)
    importances = regr.feature_importances_
    result = []
    for idx, name in enumerate(df._input_names):
        result.append((name, importances[idx]))
    result = sorted(result, key=lambda t: t[1], reverse=True)[:limit_num]
    logger.log('debug', 'Selected parameters: {}'.format(
        [r[0] for r in result]
    ), call_loc='LIMIT')
    return result
