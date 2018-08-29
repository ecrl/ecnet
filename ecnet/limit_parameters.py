#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/limit_parameters.py
# v.1.5
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains the functions necessary for reducing the input dimensionality of a
# database to the most influential input parameters, using either an iterative
# inclusion algorithm (limit_iterative_include) or a genetic algorithm
# (limit_genetic).
#

import csv
import copy
from pygenetics.ga_core import Population
from pygenetics.selection_functions import minimize_best_n

import ecnet.model
import ecnet.error_utils


def limit_iterative_include(DataFrame, limit_num):
    '''
    Limits the dimensionality of input data found in supplied *DataFrame*
    object to a dimensionality of *limit_num* using iterative inclusion
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
                learn_input = copy.deepcopy(learn_input_retained)
                valid_input = copy.deepcopy(valid_input_retained)
                test_input = copy.deepcopy(test_input_retained)
                for idx_add, param_add in enumerate(learn_input_add):
                    learn_input[idx_add].append(param_add[0])
                for idx_add, param_add in enumerate(valid_input_add):
                    valid_input[idx_add].append(param_add[0])
                for idx_add, param_add in enumerate(test_input_add):
                    test_input[idx_add].append(param_add[0])

            model = ecnet.model.MultilayerPerceptron()
            model.add_layer(len(learn_input[0]), 'relu')
            model.add_layer(8, 'relu')
            model.add_layer(8, 'relu')
            model.add_layer(len(packaged_data.learn_y[0]), 'linear')
            model.connect_layers()

            model.fit_validation(
                learn_input,
                packaged_data.learn_y,
                valid_input,
                packaged_data.valid_y,
                learning_rate=0.1,
                max_epochs=5000
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
        print(retained_input_list)
        print(rmse_val)
        print()

    return retained_input_list


def limit_genetic(DataFrame, limit_num, population_size, num_survivors,
                  num_generations, shuffle=False, print_feedback=True):
    '''
    Limits the dimensionality of input data found in supplied *DataFrame*
    object to a dimensionality of *limit_num* using a genetic algorithm.
    Optional arguments for *population_size* of genetic algorithm's population,
    *num_survivors* for selecting the best performers from each population
    generation to reproduce, *num_generations* for the number of times the
    population will reproduce, *shuffle* for shuffling the data sets for each
    population member, and *print_feedback* for printing the average fitness
    score of the population after each generation.
    '''

    def ecnet_limit_inputs(feed_dict):
        '''
        Genetic algorithm cost function, supplied to the genetic algorithm;
        returns the RMSE of the test set results from a model constructed
        using the current permutation of input parameters *feed_dict* supplied
        by the genetic algorithm
        '''

        learn_input = []
        valid_input = []
        test_input = []

        if shuffle:
            DataFrame.shuffle('l', 'v', 't')
            packaged_data_cf = DataFrame.package_sets()
        else:
            packaged_data_cf = packaged_data

        for param in feed_dict:
            learn_input_add = [
                [sublist[
                    feed_dict[param]
                ]] for sublist in packaged_data_cf.learn_x
            ]
            valid_input_add = [
                [sublist[
                    feed_dict[param]
                ]] for sublist in packaged_data_cf.valid_x
            ]
            test_input_add = [
                [sublist[
                    feed_dict[param]
                ]] for sublist in packaged_data_cf.test_x
            ]

            if len(learn_input) == 0:
                learn_input = learn_input_add
                valid_input = valid_input_add
                test_input = test_input_add
            else:
                for idx_add, param_add in enumerate(learn_input_add):
                    learn_input[idx_add].append(param_add[0])
                for idx_add, param_add in enumerate(valid_input_add):
                    valid_input[idx_add].append(param_add[0])
                for idx_add, param_add in enumerate(test_input_add):
                    test_input[idx_add].append(param_add[0])

        model = ecnet.model.MultilayerPerceptron()
        model.add_layer(len(learn_input[0]), 'relu')
        model.add_layer(8, 'relu')
        model.add_layer(8, 'relu')
        model.add_layer(len(packaged_data_cf.learn_y[0]), 'linear')
        model.connect_layers()

        model.fit_validation(
            learn_input,
            packaged_data_cf.learn_y,
            valid_input,
            packaged_data_cf.valid_y,
            learning_rate=0.1,
            max_epochs=5000
        )

        return ecnet.error_utils.calc_rmse(
            model.use(test_input),
            packaged_data_cf.test_y
        )

    packaged_data = DataFrame.package_sets()

    population = Population(
        size=population_size,
        cost_fn=ecnet_limit_inputs,
        select_fn=minimize_best_n
    )

    for i in range(limit_num):
        population.add_parameter(i, 0, DataFrame.num_inputs - 1)

    population.generate_population()

    if print_feedback:
        print('Generation: 0 - Population fitness: {}'.format(
            sum(p.fitness_score for p in population.members) / len(population)
        ))

    for gen in range(num_generations):
        population.next_generation(num_survivors=num_survivors, mut_rate=0)
        if print_feedback:
            print('Generation: {} - Population fitness: {}'.format(
                gen + 1,
                sum(p.fitness_score for p in population.members) / len(
                    population
                )
            ))

    min_idx = 0
    for new_idx, member in enumerate(population.members):
        if member.fitness_score < population.members[min_idx].fitness_score:
            min_idx = new_idx

    input_list = []
    for val in population.members[min_idx].feed_dict.values():
        input_list.append(DataFrame.input_names[val])

    if print_feedback:
        print('Best member fitness score: {}'.format(
            population.members[min_idx].fitness_score)
        )
        print(input_list)

    return input_list


def output(DataFrame, param_list, filename):
    '''
    Saves the parameters *param_list* (obtained from limit) to new database
    specified by *filename*. A *DataFrame* object is required for new database
    formatting and populating.
    '''

    if '.csv' not in filename:
        filename += '.csv'

    rows = []

    type_row = []
    type_row.append('DATAID')
    type_row.append('ASSIGNMENT')
    for string in DataFrame.string_names:
        type_row.append('STRING')
    for group in DataFrame.group_names:
        type_row.append('GROUP')
    for target in DataFrame.target_names:
        type_row.append('TARGET')
    for input_param in param_list:
        type_row.append('INPUT')
    rows.append(type_row)

    title_row = []
    title_row.append('DATAID')
    title_row.append('ASSIGNMENT')
    for string in DataFrame.string_names:
        title_row.append(string)
    for group in DataFrame.group_names:
        title_row.append(group)
    for target in DataFrame.target_names:
        title_row.append(target)
    for input_param in param_list:
        title_row.append(input_param)
    rows.append(title_row)

    input_param_indices = []
    for param in param_list:
        input_param_indices.append(DataFrame.input_names.index(param))

    for point in DataFrame.data_points:
        data_row = []
        data_row.append(point.id)
        data_row.append(point.assignment)
        for string in point.strings:
            data_row.append(string)
        for group in point.groups:
            data_row.append(group)
        for target in point.targets:
            data_row.append(target)
        for param in input_param_indices:
            data_row.append(point.inputs[param])
        rows.append(data_row)

    with open(filename, 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL, lineterminator='\n')
        for row in rows:
            wr.writerow(row)
