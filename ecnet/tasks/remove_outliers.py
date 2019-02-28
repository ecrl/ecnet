from copy import deepcopy

# 3rd party imports
from ditto_lib.itemcollection import Attribute, ItemCollection
from ditto_lib.tasks.outliers import detect_outliers
from ditto_lib.utils.logging import logger as ditto_logger

from ecnet.utils.logging import logger


def remove_outliers(df, leaf_size=40, num_processes=1):
    '''Unsupervised outlier detection using local outlier factor

    Args:
        df (ecnet.utils.data_utils.DataFrame): loaded data
        leaf_size (int): used by nearest-neighbor algorithm as the number of
            points at which to switch to brute force
        num_processes (int): number of parallel jobs for LOF algorithm

    Returns:
        ecnet.utils.data_utils.DataFrame: data w/o outliers
    '''

    ditto_logger.stream_level = logger.stream_level
    if logger.file_level != 'disable':
        ditto_logger.log_dir = logger.log_dir
        ditto_logger.file_level = logger.file_level
    item_collection = ItemCollection(df._filename)
    for inp_name in df.input_names:
        item_collection.add_attribute(Attribute(inp_name))
    for pt in df.data_points:
        item_collection.add_item(pt.id, deepcopy(pt.inputs))
    item_collection.strip()
    outliers = detect_outliers(
        item_collection,
        leaf_size=leaf_size,
        n_jobs=num_processes
    )
    logger.log('debug', 'Outliers: {}'.format(outliers), call_loc='OUTLIERS')
    for idx, pt in enumerate(df.data_points):
        for out in outliers:
            if pt.id == out:
                del df.data_points[idx]
                break
    return df
