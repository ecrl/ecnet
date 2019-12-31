#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tasks/training.py
# v.3.2.3
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains function for project training (multiprocessed training)
#

# stdlib. imports
from multiprocessing import Pool, set_start_method
from operator import itemgetter
from os import name

# ECNet imports
from ecnet.utils.logging import logger
from ecnet.utils.data_utils import DataFrame, PackagedData
from ecnet.utils.server_utils import get_candidate_path, resave_df,\
    resave_model, save_df, train_model


def train_project(prj_name: str, num_pools: int, num_candidates: int,
                  df: DataFrame, sets: PackagedData, vars: dict, shuffle: str,
                  split: list, retrain: bool, validate: bool,
                  selection_set: str, selection_fn: str, num_processes: int):
    ''' train_project: populates and ECNet project's folder structure with
    neural networks, selects best neural networks in each pool

    Args:
        prj_name (str): name of the project
        num_pools (int): number of pools in the project
        num_candidates (int): number of candidates per pool
        df (DataFrame): currently loaded DataFrame
        sets (PackagedData): learn, validation, test sets
        vars (dict): learning/architecture variables
        shuffle (str): shuffles None, `train`, `all` sets for each candidate
        split (list): if shuffling, [learn%, valid%, test%]
        retrain (bool): if True, uses existing project models for additional
            training
        selection_set (str): best candidates/pool are selected using this
            set; `learn`, `valid`, `train`, `test`, None (all data)
        selection_fn (str): candidates are selected based on this error
            metric; `rmse`, `mean_abs_error`, `med_abs_error`
        num_processes (int): number of concurrent processes used to train
    '''

    if name != 'nt':
        set_start_method('spawn', force=True)

    logger.log('info', 'Training {}x{} models'.format(
        num_pools, num_candidates
    ), call_loc='TRAIN')
    logger.log('debug', 'Arguments:\n\t| shuffle:\t\t{}\n\t| split:\t\t{}\n\t'
               '| retrain:\t\t{}\n\t| validate\t\t{}\n\t| selection_set:\t{}'
               '\n\t| selection_fn:\t\t{}'.format(
                   shuffle, split, retrain, validate, selection_set,
                   selection_fn
               ), call_loc='TRAIN')

    pool_errors = [[] for _ in range(num_pools)]
    if num_processes > 1:
        train_pool = Pool(processes=num_processes)

    for pool in range(num_pools):

        for candidate in range(num_candidates):

            filename = get_candidate_path(
                prj_name,
                pool,
                candidate,
                model=True
            )
            save_df(df, filename.replace('model.ecnet', 'data.d'))

            if num_processes > 1:
                pool_errors[pool].append(train_pool.apply_async(
                    train_model, [sets, vars, selection_set, selection_fn,
                                  retrain, filename, validate]
                ))
            else:
                pool_errors[pool].append(train_model(
                    sets, vars, selection_set, selection_fn, retrain, filename,
                    validate
                )[0])

            if shuffle is not None:
                df.shuffle(sets=shuffle, split=split)
                sets = df.package_sets()

    if num_processes > 1:
        train_pool.close()
        train_pool.join()
        for p_idx, pool in enumerate(pool_errors):
            pool_errors[p_idx] = [e.get()[0] for e in pool]

    logger.log('debug', 'Pool errors: {}'.format(pool_errors),
               call_loc='TRAIN')

    for p_idx, pool in enumerate(pool_errors):
        candidate_fp = get_candidate_path(
            prj_name,
            p_idx,
            min(enumerate(pool), key=itemgetter(1))[0],
            model=True
        )
        pool_fp = get_candidate_path(prj_name, p_idx, p_best=True)
        resave_model(candidate_fp, pool_fp)
        resave_df(
            candidate_fp.replace('model.ecnet', 'data.d'),
            pool_fp.replace('model.ecnet', 'data.d')
        )
