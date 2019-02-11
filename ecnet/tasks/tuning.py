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
from apisoptimizer import Colony
from apisoptimizer import logger as ap_logger

# ECNet imports
from ecnet.utils.logging import logger
from ecnet.utils.server_utils import default_config, train_model


def tune_hyperparameters(df, vars, num_employers, num_iterations,
                         num_processes=1, shuffle=False, split=None,
                         selection_set=None, selection_fn='rmse'):
    '''Tunes neural network learning/architecture hyperparameters

    Args:
        df (ecnet.utils.data_utils.DataFrame): currently loaded data
        vars (dict): ecnet.Server._vars variables
        num_employers (int): number of employer bees
        num_iterations (int): number of search cycles for the colony
        num_processes (int): number of parallel processes to utilize
        shuffle (bool): if True, shuffles L/V/T data for all evals
        split (list): if shuffle is True, [learn%, valid%, test%]

    Returns:
        dict: tuned hyperparameters
    '''

    colony_args = {
        'df': df,
        'shuffle': shuffle,
        'num_processes': num_processes,
        'split': split,
        'sel_set': selection_set,
        'sel_fn': selection_fn
    }

    ap_logger.stream_level = logger.stream_level
    if logger.file_level != 'disable':
        ap_logger.file_level = logger.file_level
        ap_logger.log_dir = logger.log_dir

    abc = Colony(
        num_employers,
        tune_fitness_function,
        colony_args,
        num_processes=num_processes
    )
    abc.add_param('beta_1', 0.0, 1.0)
    abc.add_param('beta_2', 0.0, 1.0)
    abc.add_param('decay', 0.0, 1.0)
    abc.add_param('epsilon', 0.0, 1.0)
    abc.add_param('num_hidden_1', 4, 40)
    abc.add_param('num_hidden_2', 4, 40)
    abc.add_param('learning_rate', 0.0, 1.0)
    abc.initialize()
    for _ in range(num_iterations):
        logger.log('debug', 'Population fitness: {}'.format(
            abc.ave_obj_fn_val), call_loc='TUNE')
        abc.search()
    return abc.best_parameters


def tune_fitness_function(params, args):
    '''Fitness function used by ABC

    Args:
        params (dict): bee hyperparams
        args (dict): additional arguments

    Returns:
        float: mean absolute error of NN with supplied hyperparams
    '''

    vars = default_config()
    vars['beta_1'] = params['beta_1'].value
    vars['beta_2'] = params['beta_2'].value
    vars['decay'] = params['decay'].value
    vars['epsilon'] = params['epsilon'].value
    vars['hidden_layers'][0][0] = params['num_hidden_1'].value
    vars['hidden_layers'][1][0] = params['num_hidden_2'].value
    vars['learning_date'] = params['learning_rate'].value

    df = args['df']
    if args['shuffle']:
        df.shuffle('all', args['split'])
    sets = df.package_sets()

    return train_model(sets, vars, args['sel_set'], args['sel_fn'],
                       validate=False, save=False)
