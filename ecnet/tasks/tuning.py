#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tasks/tuning.py
# v.3.0.0
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions/fitness functions for tuning hyperparameters
#

# 3rd party imports
from ecabc.abc import ABC

# ECNet imports
from ecnet.utils.logging import logger
from ecnet.utils.server_utils import default_config, train_model


def tune_hyperparameters(df, vars, num_employers, num_iterations,
                         num_processes=1, shuffle=None, split=None,
                         validate=True, eval_set=None, eval_fn='rmse'):
    '''Tunes neural network learning/architecture hyperparameters

    Args:
        df (ecnet.utils.data_utils.DataFrame): currently loaded data
        vars (dict): ecnet.Server._vars variables
        num_employers (int): number of employer bees
        num_iterations (int): number of search cycles for the colony
        num_processes (int): number of parallel processes to utilize
        shuffle (bool): if True, shuffles L/V/T data for all evals
        split (list): if shuffle is True, [learn%, valid%, test%]
        validate (bool): if True, uses periodic validation; otherwise, no
        eval_set (str): set used to evaluate bee performance; `learn`, `valid`,
            `train`, `test`, None (all sets)
        eval_fn (str): error function used to evaluate bee performance; `rmse`,
            `mean_abs_error`, `med_abs_error`

    Returns:
        dict: tuned hyperparameters
    '''

    fit_fn_args = {
        'df': df,
        'shuffle': shuffle,
        'num_processes': num_processes,
        'split': split,
        'validate': validate,
        'eval_set': eval_set,
        'eval_fn': eval_fn
    }

    value_ranges = [
        ('float', (0.0, 1.0)),
        ('float', (0.0, 1.0)),
        ('float', (0.0, 1.0)),
        ('float', (0.0, 1.0)),
        ('int', (4, 40)),
        ('int', (4, 40)),
        ('float', (0.0, 1.0))
    ]

    abc = ABC(
        tune_fitness_function,
        num_employers=num_employers,
        value_ranges=value_ranges,
        args=fit_fn_args,
        processes=num_processes
    )

    abc._logger.stream_level = logger.stream_level
    if logger.file_level != 'disable':
        abc._logger.log_dir = logger.log_dir
        abc._logger.file_level = logger.file_level
    abc.create_employers()
    for _ in range(num_iterations):
        abc.run_iteration()
        logger.log('debug', 'Best Performer: {}, {}'.format(
            abc.best_performer[0], abc.best_performer[1]
        ))
    params = abc.best_performer[1]
    vars['beta_1'] = params[0]
    vars['beta_2'] = params[1]
    vars['decay'] = params[2]
    vars['epsilon'] = params[3]
    vars['hidden_layers'][0][0] = params[4]
    vars['hidden_layers'][1][0] = params[5]
    vars['learning_date'] = params[6]
    return vars


def tune_fitness_function(params, **kwargs):
    '''Fitness function used by ABC

    Args:
        params (dict): bee hyperparams
        kwargs (dict): additional arguments

    Returns:
        float: error of NN with supplied hyperparams
    '''

    vars = default_config()
    vars['beta_1'] = params[0]
    vars['beta_2'] = params[1]
    vars['decay'] = params[2]
    vars['epsilon'] = params[3]
    vars['hidden_layers'][0][0] = params[4]
    vars['hidden_layers'][1][0] = params[5]
    vars['learning_date'] = params[6]

    df = kwargs['df']
    if kwargs['shuffle'] is not None:
        df.shuffle(kwargs['shuffle'], kwargs['split'])
    sets = df.package_sets()

    return train_model(sets, vars, kwargs['eval_set'], kwargs['eval_fn'],
                       validate=kwargs['validate'], save=False)
