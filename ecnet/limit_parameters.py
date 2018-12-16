#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/limit_parameters.py
# v.2.0.0
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains the functions necessary for reducing the input dimensionality of a
# database to the most influential input parameters, using either an iterative
# inclusion algorithm (limit_iterative_include) or a genetic algorithm
# (limit_genetic).
#

# Stdlib imports
from csv import writer, QUOTE_ALL
from copy import deepcopy
from multiprocessing import current_process

# 3rd party imports
from colorlogging import ColorLogger
from pygenetics.ga_core import Population
from pygenetics.selection_functions import minimize_best_n

# ECNet imports
import ecnet.model
import ecnet.error_utils
from ecnet.fitness_functions import limit_inputs


def limit_iterative_include(DataFrame, limit_num, vars, logger=None):
    '''Limits the dimensionality of input data using an iterative inclusion
    algorithm (best parameter is found, retained, paired with all others, best
    pair retained, continues until desired dimensionality is reached); best
    used with explicit sorting

    Args:
        DataFrame (DataFrame): ECNet DataFrame object to limit
        limit_num (int): desired input dimensionality
        vars (dict): dictionary of ECNet Server model variables
        logger (ColorLogger): ColorLogger object; if not supplied, does not log

    Returns:
        list: list of resulting input parameter names
    '''

    retained_input_list = []

    learn_input_retained = []
    valid_input_retained = []
    test_input_retained = []

    packaged_data = DataFrame.package_sets()

    while len(retained_input_list) < limit_num:

        retained_rmse_list = []

        for idx, param in enumerate(DataFrame.input_names):

            learn_input_add = [
                [sublist[idx]] for sublist in packaged_data.learn_x
            ]
            valid_input_add = [
                [sublist[idx]] for sublist in packaged_data.valid_x
            ]
            test_input_add = [
                [sublist[idx]] for sublist in packaged_data.test_x
            ]

            if len(retained_input_list) is 0:
                learn_input = learn_input_add
                valid_input = valid_input_add
                test_input = test_input_add
            else:
                learn_input = deepcopy(learn_input_retained)
                valid_input = deepcopy(valid_input_retained)
                test_input = deepcopy(test_input_retained)
                for idx_add, param_add in enumerate(learn_input_add):
                    learn_input[idx_add].append(param_add[0])
                for idx_add, param_add in enumerate(valid_input_add):
                    valid_input[idx_add].append(param_add[0])
                for idx_add, param_add in enumerate(test_input_add):
                    test_input[idx_add].append(param_add[0])

            model = ecnet.model.MultilayerPerceptron()
            model.add_layer(len(learn_input[0]), vars['input_activation'])
            for layer in vars['hidden_layers']:
                model.add_layer(layer[0], layer[1])
            model.add_layer(
                len(packaged_data.learn_y[0]),
                vars['output_activation']
            )
            model.connect_layers()

            model.fit_validation(
                learn_input,
                packaged_data.learn_y,
                valid_input,
                packaged_data.valid_y,
                learning_rate=vars['learning_rate'],
                keep_prob=vars['keep_prob'],
                max_epochs=vars['validation_max_epochs']
            )

            retained_rmse_list.append(ecnet.error_utils.calc_rmse(
                model.use(test_input),
                packaged_data.test_y))

        rmse_val, rmse_idx = min(
            (rmse_val, rmse_idx) for (rmse_idx, rmse_val) in enumerate(
                retained_rmse_list
            )
        )

        learn_retain_add = [
            [sublist[rmse_idx]] for sublist in packaged_data.learn_x
        ]
        valid_retain_add = [
            [sublist[rmse_idx]] for sublist in packaged_data.valid_x
        ]
        test_retain_add = [
            [sublist[rmse_idx]] for sublist in packaged_data.test_x
        ]

        if len(retained_input_list) is 0:
            learn_input_retained = learn_retain_add
            valid_input_retained = valid_retain_add
            test_input_retained = test_retain_add
        else:
            for idx, param in enumerate(learn_retain_add):
                learn_input_retained[idx].append(param[0])
            for idx, param in enumerate(valid_retain_add):
                valid_input_retained[idx].append(param[0])
            for idx, param in enumerate(test_retain_add):
                test_input_retained[idx].append(param[0])

        retained_input_list.append(DataFrame.input_names[rmse_idx])
        if logger is not None:
            logger.log(
                'debug',
                'Currently retained: {}'.format(retained_input_list),
                call_loc={'call_loc': 'LIMIT'}
            )
            logger.log(
                'debug',
                'Current RMSE: {}'.format(rmse_val),
                call_loc={'call_loc': 'LIMIT'}
            )

    logger.log(
        'debug',
        'Limited inputs: {}'.format(retained_input_list),
        call_loc={'call_loc': 'LIMIT'}
    )

    return retained_input_list


def limit_genetic(DataFrame, limit_num, vars, population_size, num_generations,
                  num_processes, shuffle=False, data_split=[0.65, 0.25, 0.1],
                  logger=None):
    '''Limits the dimensionality of input data using a genetic algorithm

    Args:
        DataFrame (DataFrame): ECNet DataFrame object to limit
        limit_num (int): desired input dimensionality
        vars (dict): dictionary of ECNet Server model variables
        population_size (int): size of genetic algorithm population
        num_generations (int): number of generations to run the GA for
        num_processes (int): number of concurrent processes used by the GA
        shuffle (bool): whether to shuffle the data sets for each population
            member
        data_split (list): [learn%, valid%, test%] if shuffle == True
        logger (ColorLogger): ColorLogger object; if not supplied, does not log

    Returns:
        list: list of resulting input parameter names
    '''

    packaged_data = DataFrame.package_sets()

    cost_fn_args = {
        'DataFrame': DataFrame,
        'packaged_data': packaged_data,
        'shuffle': shuffle,
        'data_split': data_split,
        'num_processes': num_processes,
        'learning_rate': vars['learning_rate'],
        'keep_prob': vars['keep_prob'],
        'hidden_layers': vars['hidden_layers'],
        'input_activation': vars['input_activation'],
        'output_activation': vars['output_activation'],
        'validation_max_epochs': vars['validation_max_epochs']
    }

    population = Population(
        size=population_size,
        cost_fn=limit_inputs,
        cost_fn_args=cost_fn_args,
        num_processes=num_processes,
        select_fn=minimize_best_n
    )

    for i in range(limit_num):
        population.add_parameter(i, 0, DataFrame.num_inputs - 1)

    population.generate_population()
    if logger is not None:
        logger.log('debug', 'Generation: 0 - Population fitness: {}'.format(
            sum(p.fitness_score for p in population.members) / len(population),
        ), call_loc={'call_loc': 'LIMIT'})

    for gen in range(num_generations):
        population.next_generation()
        if logger is not None:
            logger.log(
                'debug',
                'Generation: {} - Population fitness: {}'.format(
                    gen + 1,
                    sum(
                        p.fitness_score for p in population.members
                    ) / len(population)
                ),
                call_loc={'call_loc': 'LIMIT'}
            )

    min_idx = 0
    for new_idx, member in enumerate(population.members):
        if member.fitness_score < population.members[min_idx].fitness_score:
            min_idx = new_idx

    input_list = []
    for val in population.members[min_idx].parameters.values():
        input_list.append(DataFrame.input_names[val])

    if logger is not None:
        logger.log(
            'debug',
            'Best member fitness score: {}'.format(
                population.members[min_idx].fitness_score
            ),
            call_loc={'call_loc': 'LIMIT'}
        )
        logger.log(
            'debug',
            'Best member parameters: {}'.format(input_list),
            call_loc={'call_loc': 'LIMIT'}
        )

    return input_list
