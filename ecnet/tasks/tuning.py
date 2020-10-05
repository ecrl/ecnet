#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tasks/tuning.py
# v.3.3.1
# Developed in 2020 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions/fitness functions for tuning hyperparameters
#

# stdlib. imports
from multiprocessing import set_start_method
from os import name

# 3rd party imports
from ecabc import ABC

# ECNet imports
from ecnet.utils.data_utils import DataFrame
from ecnet.utils.logging import logger
from ecnet.utils.server_utils import default_config, train_model


def tune_hyperparameters(df: DataFrame, vars: dict, num_employers: int,
                         num_iterations: int, num_processes: int = 1,
                         shuffle: str = None, split: list = None,
                         validate: bool = True, eval_set: str = None,
                         eval_fn: str = 'rmse', epochs: int = 500) -> dict:
    '''Tunes neural network learning/architecture hyperparameters

    Args:
        df (ecnet.utils.data_utils.DataFrame): currently loaded data
        vars (dict): ecnet.Server._vars variables
        num_employers (int): number of employer bees
        num_iterations (int): number of search cycles for the colony
        num_processes (int): number of parallel processes to utilize
        shuffle (str): shuffles `train` or `all` sets if not None
        split (list): if shuffle is True, [learn%, valid%, test%]
        validate (bool): if True, uses periodic validation; otherwise, no
        eval_set (str): set used to evaluate bee performance; `learn`, `valid`,
            `train`, `test`, None (all sets)
        eval_fn (str): error function used to evaluate bee performance; `rmse`,
            `mean_abs_error`, `med_abs_error`
        epochs (int): number of training epochs per bee ANN (def: 500)

    Returns:
        dict: tuned hyperparameters
    '''

    if name != 'nt':
        set_start_method('spawn', force=True)

    logger.log('info', 'Tuning architecture/learning hyperparameters',
               call_loc='TUNE')
    logger.log('debug', 'Arguments:\n\t| num_employers:\t{}\n\t| '
               'num_iterations:\t{}\n\t| shuffle:\t\t{}\n\t| split:'
               '\t\t{}\n\t| validate:\t\t{}\n\t| eval_set:\t\t{}\n\t'
               '| eval_fn:\t\t{}'.format(
                   num_employers, num_iterations, shuffle, split, validate,
                   eval_set, eval_fn
               ), call_loc='TUNE')

    fit_fn_args = {
        'df': df,
        'shuffle': shuffle,
        'num_processes': num_processes,
        'split': split,
        'validate': validate,
        'eval_set': eval_set,
        'eval_fn': eval_fn,
        'hidden_layers': vars['hidden_layers'],
        'epochs': epochs
    }

    to_tune = [
        (1e-9, 1e-4, 'decay'),
        (1e-5, 0.1, 'learning_rate'),
        (1, len(df.learn_set), 'batch_size'),
        (64, 1024, 'patience')
    ]
    for hl in range(len(vars['hidden_layers'])):
        to_tune.append((1, 2 * len(df._input_names), 'hl{}'.format(hl)))

    abc = ABC(num_employers, tune_fitness_function, fit_fn_args, num_processes)
    for param in to_tune:
        abc.add_param(param[0], param[1], name=param[2])
    abc.initialize()

    best_ret_val = abc.best_ret_val
    best_params = abc.best_params
    for i in range(num_iterations):
        logger.log('info', 'Iteration {}'.format(i + 1), call_loc='TUNE')
        abc.search()
        new_best_ret = abc.best_ret_val
        new_best_params = abc.best_params
        logger.log('info', 'Best Performer: {}, {}'.format(
            new_best_ret, new_best_ret
        ), call_loc='TUNE')
        if new_best_ret < best_ret_val:
            best_ret_val = new_best_ret
            best_params = new_best_params

    vars['decay'] = best_params['decay']
    vars['learning_rate'] = best_params['learning_rate']
    vars['batch_size'] = best_params['batch_size']
    vars['patience'] = best_params['patience']
    for l_idx in range(len(vars['hidden_layers'])):
        vars['hidden_layers'][l_idx][0] = best_params['hl{}'.format(l_idx)]
    return vars


def tune_fitness_function(params: list, **kwargs) -> float:
    '''Fitness function used by ABC

    Args:
        params (list): bee hyperparams
        kwargs (dict): additional arguments

    Returns:
        float: error of NN with supplied hyperparams
    '''

    vars = default_config()
    vars['decay'] = params[0]
    vars['learning_rate'] = params[1]
    vars['batch_size'] = params[2]
    vars['patience'] = params[3]
    vars['hidden_layers'] = kwargs['hidden_layers']
    vars['epochs'] = kwargs['epochs']
    for l_idx in range(len(vars['hidden_layers'])):
        vars['hidden_layers'][l_idx][0] = params[4 + l_idx]

    df = kwargs['df']
    if kwargs['shuffle'] is not None:
        df.shuffle(kwargs['shuffle'], kwargs['split'])
    sets = df.package_sets()

    return train_model(sets, vars, kwargs['eval_set'], kwargs['eval_fn'],
                       validate=kwargs['validate'], save=False)[0]
