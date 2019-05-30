#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tasks/limit_inputs.py
# v.3.1.0
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions for selecting influential input parameters
#

from copy import deepcopy

# 3rd party imports
from ditto_lib.itemcollection import ItemCollection
from ditto_lib.tasks.random_forest import random_forest_regressor
from ditto_lib.utils.logging import logger as ditto_logger
from ditto_lib.utils.dataframe import Attribute

# ECNet imports
from ecnet.utils.logging import logger


def limit_rforest(df, limit_num, num_estimators=1000, num_processes=1):
    '''Uses random forest regression to select input parameters

    Args:
        df (ecnet.utils.data_utils.DataFrame): loaded data
        limit_num (int): desired number of input parameters
        num_estimators (int): number of trees used by RFR algorithm
        num_processes (int): number of parallel jobs for RFR algorithm

    Returns:
        ecnet.utils.data_utils.DataFrame: limited data
    '''

    ditto_logger.stream_level = logger.stream_level
    if logger.file_level != 'disable':
        ditto_logger.log_dir = logger.log_dir
        ditto_logger.file_level = logger.file_level
    ditto_logger.default_call_loc('LIMIT')
    item_collection = ItemCollection(df._filename)
    for inp_name in df._input_names:
        item_collection.add_attribute(Attribute(inp_name))
    for pt in df.data_points:
        item_collection.add_item(
            pt.id,
            deepcopy([getattr(pt, i) for i in df._input_names])
        )
    for tar_name in df._target_names:
        item_collection.add_attribute(Attribute(tar_name, is_descriptor=False))
    for pt in df.data_points:
        target_vals = [getattr(pt, t) for t in df._target_names]
        for idx, tar in enumerate(target_vals):
            item_collection.set_item_attribute(
                pt.id, tar, df._target_names[idx]
            )
    item_collection.strip()
    params = [param[0] for param in random_forest_regressor(
        item_collection.dataframe,
        target_attribute=df._target_names[0],
        n_components=limit_num,
        n_estimators=num_estimators,
        n_jobs=num_processes
    )]
    for idx, param in enumerate(params):
        for tn in df._target_names:
            if tn == param:
                del params[idx]
                break

    logger.log('debug', 'Selected parameters: {}'.format(params),
               call_loc='LIMIT')
    df.set_inputs(params)
    return df
