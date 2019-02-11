#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tasks/limit_inputs.py
# v.3.0.0
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions for selecting influential input parameters
#

# 3rd party imports
from ditto_lib.generic.itemcollection import ItemCollection, Attribute,\
    logger as ditto_logger

# ECNet imports
from ecnet.utils.logging import logger


def limit_rforest(df, limit_num, num_estimators=1000, num_processes=1):
    '''Uses random forest regression to select input parameters

    Args:
        df (ecnet.utils.data_utils.DataFrame): loaded data
        limit_num (int): desired number of input parameters
        num_estimators (int): number of trees used by RFR algorithm
        num_processes (int): number parallel jobs for RFR algorithm

    Returns:
        list: input parameter names
    '''

    ditto_logger.stream_level = logger.stream_level
    if logger.file_level != 'disable':
        ditto_logger.file_level = logger.file_level
        ditto_logger.log_dir = logger.log_dir
    item_collection = ItemCollection(df._filename)
    non_desc = ['DATAID', 'ASSIGNMENT']
    non_desc.extend([sn for sn in df.string_names])
    non_desc.extend([gn for gn in df.group_names])
    non_desc.extend([tn for tn in df.target_names])
    item_collection.from_csv(
        df._filename,
        start_row=1,
        preamble_indexes=(0, 0),
        non_descriptors=non_desc
    )
    item_collection.strip()
    return [param[0] for param in item_collection.random_forest_regressor(
        target_attribute=df.target_names[0],
        n_components=limit_num,
        n_estimators=num_estimators,
        n_jobs=num_processes
    )]
