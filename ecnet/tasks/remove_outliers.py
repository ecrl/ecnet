#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tasks/remove_outliers.py
# v.3.2.1
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains function for removing outliers from ECNet DataFrame
#

# stdlib imports
from copy import deepcopy

# 3rd party imports
from ditto_lib.itemcollection import Attribute, ItemCollection
from ditto_lib.tasks.outliers import local_outlier_factor
from ditto_lib.utils.logging import logger as ditto_logger
from ditto_lib.utils.dataframe import Attribute

# ECNet imports
from ecnet.utils.data_utils import DataFrame
from ecnet.utils.logging import logger


def remove_outliers(df: DataFrame, leaf_size: int=40,
                    num_processes: int=1) -> DataFrame:
    '''Unsupervised outlier detection using local outlier factor

    Args:
        df (ecnet.utils.data_utils.DataFrame): loaded data
        leaf_size (int): used by nearest-neighbor algorithm as the number of
            points at which to switch to brute force
        num_processes (int): number of parallel jobs for LOF algorithm

    Returns:
        ecnet.utils.data_utils.DataFrame: data w/o outliers
    '''

    logger.log('info', 'Removing outliers', call_loc='OUTLIERS')
    logger.log('debug', 'Leaf size: {}'.format(leaf_size),
               call_loc='OUTLIERS')

    ditto_logger.stream_level = logger.stream_level
    if logger.file_level != 'disable':
        ditto_logger.log_dir = logger.log_dir
        ditto_logger.file_level = logger.file_level
    ditto_logger.default_call_loc('OUTLIERS')
    item_collection = ItemCollection(df._filename)
    for inp_name in df._input_names:
        item_collection.add_attribute(Attribute(inp_name))
    for pt in df.data_points:
        item_collection.add_item(
            pt.id,
            deepcopy([getattr(pt, i) for i in df._input_names])
        )
    item_collection.strip()
    outliers = local_outlier_factor(
        item_collection.dataframe,
        leaf_size=leaf_size,
        n_jobs=num_processes
    )
    logger.log('debug', 'Outliers: {}'.format(outliers), call_loc='OUTLIERS')
    for out in outliers:
        for idx, pt in enumerate(df.data_points):
            if out == pt.id:
                del df.data_points[idx]
                break
    return df
