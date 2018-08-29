#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/server.py
# v.1.5
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains the "Server" class, which handles ECNet project creation, neural
# network model creation, data hand-off to models, and prediction error
# calculation. For example scripts, refer to https://github.com/tjkessler/ecnet
#

import yaml
import warnings
import os
import numpy as np
import zipfile
import pickle
from ecabc.abc import ABC

import ecnet.data_utils
import ecnet.error_utils
import ecnet.model
import ecnet.limit_parameters


class Server:
    '''
    Server object: handles project creation/usage for ECNet, handles data
    hand-off to neural networks for model training, selection and usage, data
    importing, error calculations, hyperparameter tuning, input dimensionality
    reduction.
    '''

    def __init__(self, config_filename='config.yml', project_file=None):
        '''
        Initialization: imports model configuration file *config_filename*; if
        configuration file not found, creates default configuration file with
        name *filename*; opens an ECNet project instead if *project_file* is
        specified.
        '''

        if project_file is not None:
            self.__open_project(project_file)
            return

        self.vars = {}
        if '.yml' not in config_filename:
            config_filename += '.yml'
        try:
            file = open(config_filename, 'r')
            self.vars.update(yaml.load(file))
        except:
            warn_str = ('Supplied configuration file not found: '
                        'creating default configuration file for {}'.format(
                            config_filename
                        ))
            warnings.warn(warn_str)
            config_dict = {
                'learning_rate': 0.1,
                'hidden_layers': [
                    [10, 'relu'],
                    [10, 'relu']
                ],
                'input_activation': 'relu',
                'output_activation': 'linear',
                'train_epochs': 500,
                'validation_max_epochs': 10000
            }
            file = open(config_filename, 'w')
            yaml.dump(config_dict, file)
            self.vars.update(config_dict)

        self.__config_filename = config_filename
        self.__using_project = False

    def create_project(self, project_name, num_builds=1, num_nodes=5,
                       num_trials=10, print_feedback=True):
        '''
        Creates the folder structure for a project (no single-model creation)

        *project_name*      - name of your project
        *num_builds*        - number of builds
        *num_nodes*         - number of nodes (per build)
        *num_trials*        - number of trials (per node)
        *print_feedback*    - whether to print current progress of build
        '''

        self.__project_name = project_name
        self.__num_builds = num_builds
        self.__num_nodes = num_nodes
        self.__num_trials = num_trials
        self.__print_feedback = print_feedback
        if not os.path.exists(self.__project_name):
            os.makedirs(self.__project_name)
        for build in range(self.__num_builds):
            path_b = os.path.join(self.__project_name, 'build_{}'
                                  .format(build + 1))
            if not os.path.exists(path_b):
                os.makedirs(path_b)
            for node in range(self.__num_nodes):
                path_n = os.path.join(path_b, 'node_{}'.format(node + 1))
                if not os.path.exists(path_n):
                    os.makedirs(path_n)
        self.__using_project = True

    def import_data(self, data_filename, sort_type='random',
                    data_split=[0.65, 0.25, 0.1]):
        '''
        Imports data from ECNet formatted CSV database

        *data_filename* - ECNet database file name
        *sort_type*     - 'random' for randomized learning, validation and
                          testing sets, 'explicit' to use ASSIGNMENT column
                          in database
        *data_split*    - if random sort type, [learn%, valid%, test%]
        '''

        self.DataFrame = ecnet.data_utils.DataFrame(data_filename)
        if sort_type == 'random':
            self.DataFrame.create_sets(random=True, split=data_split)
        elif sort_type == 'explicit':
            self.DataFrame.create_sets(random=False)
        else:
            raise ValueError('Unknown sort_type {}'.format(sort_type))
        self.__sets = self.DataFrame.package_sets()

    def limit_input_parameters(self, limit_num, output_filename,
                               use_genetic=False, population_size=500,
                               num_survivors=200, num_generations=25,
                               shuffle=False):
        '''
        Limits the input dimensionality of currently loaded DataFrame; default
        method is an iterative inclusion algorithm, options for using a genetic
        algorithm available.

        *limit_num*         - desired input dimensionality
        *output_filename*   - filename for resulting ECNet formatted database

        If *use_genetic* == True:
        *population_size*   - population size of genetic algorithm
        *num_survivors*     - number of population members to reproduce for
                              next generation
        *num_generations*   - number of generations the algorithm will run for
        *shuffle*           - whether to shuffle learning, validation and
                              testing sets for each population member

        See https://github.com/tjkessler/pygenetics for genetic algorithm
        source code.
        '''

        if use_genetic:
            params = ecnet.limit_parameters.limit_genetic(
                self.DataFrame, limit_num, population_size, num_survivors,
                shuffle=shuffle, print_feedback=self.__print_feedback
            )
        else:
            params = ecnet.limit_parameters.limit_iterative_include(
                self.DataFrame, limit_num
            )
        ecnet.limit_parameters.output(self.DataFrame, params, output_filename)

    def tune_hyperparameters(self, target_score=None, iteration_amt=50,
                             amt_employers=50, print_feedback=True):
        '''
        Tunes the neural network learning hyperparameters (learning_rate,
        validation_max_epochs, neuron counts for each hidden layer) using an
        artificial bee colony searching algorithm.

        *target_score*      - target mean absolute error score for test set
        *iteration_amt*     - if not using target score, run the ABC for this
                              number of iterations
        *amt_employers*     - number of employer "bees" for ABC
        *print_feedback*    - if not already defined in create_project(),
                              determines whether to print ABC progress

        See https://github.com/hgromer/ecabc for ABC source code.
        '''

        def test_neural_network(values):
            '''
            Fitness function used by artificial bee colony
            '''
            self.vars['learning_rate'] = values[0]
            self.vars['validation_max_epochs'] = values[1]
            for idx, layer in enumerate(self.vars['hidden_layers'], 2):
                layer[0] = values[idx]
            self.train_model(validate=True)
            return self.calc_error(
                'mean_abs_error',
                dset='test'
            )['mean_abs_error']

        hyperparameters = [
            ('float', (0.01, 0.2)),
            ('int', (1000, 25000))
        ]
        for _ in range(len(self.vars['hidden_layers'])):
            hyperparameters.append(('int', (8, 32)))

        if target_score is None:
            abc = ABC(
                iterationAmount=iteration_amt,
                fitnessFunction=test_neural_network,
                valueRanges=hyperparameters,
                amountOfEmployers=amt_employers
            )
        else:
            abc = ABC(
                endValue=target_score,
                fitnessFunction=test_neural_network,
                valueRanges=hyperparameters,
                amountOfEmployers=amt_employers
            )

        try:
            abc.printInfo(self.__print_feedback)
        except:
            abc.printInfo(print_feedback)

        use_proj = False
        if self.__using_project:
            use_proj = True
            self.__using_project = False

        new_hyperparameters = abc.runABC()

        if use_proj:
            self.__using_project = True

        self.vars['learning_rate'] = new_hyperparameters[0]
        self.vars['validation_max_epochs'] = new_hyperparameters[1]
        for idx, layer in enumerate(self.vars['hidden_layers'], 2):
            layer[0] = new_hyperparameters[idx]

        return new_hyperparameters

    def train_model(self, validate=False, shuffle=None,
                    data_split=[0.65, 0.25, 0.1]):
        '''
        create_project() not called before: trains one neural network
        create_project() called before: creates build*node*trial neural
                         networks

        *shuffle*   - if using a project, 'lvt' shuffles learning, validation
                      and testing sets for each trial; 'lv' shuffles learning
                      and validation sets for each trial
        *validate*  - if True, uses validation set to periodically check if
                      any additional learning is needed (if validation set
                      performance stops improving, stop learning)
        '''

        if not self.__using_project:
            model = self.__create_model()
            if validate:
                model.fit_validation(
                    self.__sets.learn_x,
                    self.__sets.learn_y,
                    self.__sets.valid_x,
                    self.__sets.valid_y,
                    self.vars['learning_rate'],
                    self.vars['validation_max_epochs']
                )
            else:
                model.fit(
                    self.__sets.learn_x,
                    self.__sets.learn_y,
                    self.vars['learning_rate'],
                    self.vars['train_epochs']
                )
            model.save('./tmp/model')

        else:
            for build in range(self.__num_builds):
                path_b = os.path.join(
                    self.__project_name, 'build_{}'.format(build + 1)
                )
                for node in range(self.__num_nodes):
                    path_n = os.path.join(
                        path_b, 'node_{}'.format(node + 1)
                    )
                    for trial in range(self.__num_trials):
                        if self.__print_feedback:
                            print('Build {}, Node {}, Trial {}...'.format(
                                build + 1,
                                node + 1,
                                trial + 1
                            ))
                        path_t = os.path.join(
                            path_n, 'trial_{}'.format(trial + 1)
                        )
                        model = self.__create_model()
                        if validate:
                            model.fit_validation(
                                self.__sets.learn_x,
                                self.__sets.learn_y,
                                self.__sets.valid_x,
                                self.__sets.valid_y,
                                self.vars['learning_rate'],
                                self.vars['validation_max_epochs']
                            )
                        else:
                            model.fit(
                                self.__sets.learn_x,
                                self.__sets.learn_y,
                                self.vars['learning_rate'],
                                self.vars['train_epochs']
                            )
                        model.save(path_t)
                        if shuffle == 'lv':
                            self.DataFrame.shuffle(
                                'l', 'v', split=data_split
                            )
                            self.__sets = self.DataFrame.package_sets()
                        elif shuffle == 'lvt':
                            self.DataFrame.create_sets(
                                split=data_split
                            )
                            self.__sets = self.DataFrame.package_sets()
                        elif shuffle is None:
                            continue
                        else:
                            raise ValueError(
                                'Unknown shuffle arg {}'.format(shuffle)
                            )

    def select_best(self, dset=None, error_fn='mean_abs_error'):
        '''
        Selects the best performing neural network trial from each node for
        each build (requires create_project())

        *dset*      - None    == select based on entire DataFrame performance
                    - 'learn' == select based on learning set performance
                    - 'valid' == select based on validation set performance
                    - 'train' == select based on learn + valid sets performance
                    - 'test'  == select based on test set performance
        *error_fn*  - 'rmse'  == measure performance using RMSE
                    - 'r2'    == measure performance using r-squared
                    - 'mean_abs_error' == measure performance using mean
                      absolute error
                    - 'med_abs_error'  == measure performance using median
                      absolute error
        '''

        if not self.__using_project:
            raise Exception('Project has not been created! (create_project())')
        if not os.path.exists(
            os.path.join(
                self.__project_name,
                os.path.join('build_1', os.path.join(
                        'node_1',
                        'trial_1.meta'
                    )
                )
            )
        ):
            raise Exception('Models must be trained first! (train_model())')
        if self.__print_feedback:
            print('Selecting best models from each node for each build...')
        x_vals = self.__determine_x_vals(dset)
        y_vals = self.__determine_y_vals(dset)
        for build in range(self.__num_builds):
            path_b = os.path.join(
                self.__project_name, 'build_{}'.format(build + 1)
            )
            for node in range(self.__num_nodes):
                path_n = os.path.join(
                    path_b, 'node_{}'.format(node + 1)
                )
                error_list = []
                for trial in range(self.__num_trials):
                    path_t = os.path.join(
                        path_n, 'trial_{}'.format(trial + 1)
                    )
                    model = ecnet.model.MultilayerPerceptron()
                    model.load(path_t)
                    error_list.append(
                        self.__error_fn(
                            error_fn, model.use(x_vals), y_vals
                        )
                    )
                current_min_idx = 0
                for idx, error in enumerate(error_list):
                    if error < error_list[current_min_idx]:
                        current_min_idx = idx
                model = ecnet.model.MultilayerPerceptron()
                path_t_best = os.path.join(
                    path_n, 'trial_{}'.format(current_min_idx + 1)
                )
                path_best = os.path.join(
                    path_n, 'model'
                )
                model.load(path_t_best)
                model.save(path_best)

    def use_model(self, dset=None):
        '''
        Use trained neural network(s), either single or project-built,
        to predict values for specified *dset* of currently loaded
        DataFrame

        *dset*  - None    == predict for entire DataFrame
                - 'learn' == predict for learning set
                - 'valid' == predict for validation set
                - 'train' == predict for learning and validation sets
                - 'test'  == predict for test set
        '''

        x_vals = self.__determine_x_vals(dset)
        if not self.__using_project:
            model = ecnet.model.MultilayerPerceptron()
            model.load('./tmp/model')
            return [model.use(x_vals)]
        else:
            if not os.path.exists(
                os.path.join(
                    self.__project_name,
                    os.path.join('build_1', os.path.join(
                            'node_1',
                            'model.meta'
                        )
                    )
                )
            ):
                raise Exception('Select best performers using select_best()')
            preds = []
            for build in range(self.__num_builds):
                path_b = os.path.join(
                    self.__project_name, 'build_{}'.format(build + 1)
                )
                build_preds = []
                for node in range(self.__num_nodes):
                    path_n = os.path.join(
                        path_b, 'node_{}'.format(node + 1)
                    )
                    path_best = os.path.join(
                        path_n, 'model'
                    )
                    model = ecnet.model.MultilayerPerceptron()
                    model.load(path_best)
                    build_preds.append(model.use(x_vals))
                ave_build_preds = []
                for pred in range(len(build_preds[0])):
                    node_preds = []
                    for node in range(len(build_preds)):
                        node_preds.append(build_preds[node][pred])
                    ave_build_preds.append(sum(node_preds)/len(node_preds))
                preds.append(list(ave_build_preds))
            return preds

    def calc_error(self, *args, dset=None):
        '''
        Calculates and returns error(s) for the specified *dset*; multiple
        errors can be calculated at once

        **args* - 'rmse' == calculates RMSE
                - 'r2'   == calculates r-squared
                - 'mean_abs_error'  == calculates mean absolute error
                - 'med_abs_error'   == calculates median absolute error
        *dset*  - None    == calculate errors for entire DataFrame
                - 'learn' == calculate errors for learning set
                - 'valid' == calculate errors for validation set
                - 'train' == calculate errors for learning and validation sets
                - 'test'  == calculate errors for test set
        '''

        error_dict = {}
        y_hat = self.use_model(dset)
        y = self.__determine_y_vals(dset)
        for arg in args:
            if self.__using_project:
                error_list = []
                for y_build in y_hat:
                    error_list.append(self.__error_fn(arg, y_build, y))
                error_dict[arg] = error_list
            else:
                error_dict[arg] = self.__error_fn(arg, y_hat, y)
        return error_dict

    def save_results(self, results, filename):
        '''
        Saves *results* obtained from *use_model()* to CSV file with name
        *filename*
        '''

        ecnet.data_utils.save_results(results, self.DataFrame, filename)

    def save_project(self, clean_up=True):
        '''
        Saves the current state of the Server (including currently imported
        DataFrame and model configuration), zips up the current state and
        project directory to "self.__project_name".project file

        *clean_up*  - deletes trial neural networks, leaving only best models
                      if select_best() has been run
        '''

        if not self.__using_project:
            raise Exception('Project has not been created! (create_project())')

        if clean_up:
            for build in range(self.__num_builds):
                path_b = os.path.join(
                    self.__project_name, 'build_{}'.format(build + 1)
                )
                for node in range(self.__num_nodes):
                    path_n = os.path.join(
                        path_b, 'node_{}'.format(node + 1)
                    )
                    trial_files = [
                        file for file in os.listdir(path_n) if 'trial' in file
                    ]
                    for file in trial_files:
                        os.remove(os.path.join(path_n, file))

        with open(
            os.path.join(self.__project_name, self.__config_filename),
            'w'
        ) as config_save:
            yaml.dump(
                self.vars,
                config_save,
                default_flow_style=False,
                explicit_start=True
            )
        config_save.close()

        with open(
            os.path.join(self.__project_name, 'data.d'),
            'wb'
        ) as data_save:
            pickle.dump(self.DataFrame, data_save)
        data_save.close()

        zip_file = zipfile.ZipFile(
            '{}.project'.format(self.__project_name),
            'w',
            zipfile.ZIP_DEFLATED
        )
        for root, dirs, files in os.walk(self.__project_name):
            for file in files:
                zip_file.write(os.path.join(root, file))
        zip_file.close()

    def __open_project(self, project_name):
        '''
        Private method: Opens a .project file, imports saved DataFrame,
        configuration, unzipsp roject folder stucture and model files. Called
        from Server initialization if *project_name* is supplied.

        *project_name*  - name of .project file to open
        '''

        self.__project_name = project_name.replace('.project', '')
        if '.project' not in project_name:
            project_name += '.project'
        self.__print_feedback = False

        zip_file = zipfile.ZipFile(project_name, 'r')
        zip_file.extractall('./')
        zip_file.close()

        self.__num_builds = len(
            [build for build in os.listdir(
                self.__project_name
            ) if os.path.isdir(os.path.join(self.__project_name, build))]
        )
        self.__num_nodes = len(
            [node for node in os.listdir(
                os.path.join(self.__project_name, 'build_1')
            ) if os.path.isdir(os.path.join(
                self.__project_name,
                os.path.join('build_1', node))
            )]
        )

        for root, dirs, files in os.walk(self.__project_name):
            for file in files:
                if '.yml' in file:
                    self.__config_filename = file
                    break

        with open(
            os.path.join(self.__project_name, self.__config_filename),
            'r'
        ) as config_file:
            self.vars = {}
            self.vars.update(yaml.load(config_file))
        config_file.close()

        with open(self.__config_filename, 'w') as config_file:
            yaml.dump(
                self.vars,
                config_file,
                default_flow_style=False,
                explicit_start=True
            )
        config_file.close()

        with open(
            os.path.join(self.__project_name, 'data.d'),
            'rb'
        ) as data_file:
            self.DataFrame = pickle.load(data_file)
        data_file.close()

        self.__sets = self.DataFrame.package_sets()
        self.__using_project = True

    def __determine_x_vals(self, dset):
        '''
        Private method: Helper function for determining which data set input
        data will be passed to functions
        '''

        if dset == 'test':
            return self.__sets.test_x
        elif dset == 'valid':
            return self.__sets.valid_x
        elif dset == 'learn':
            return self.__sets.learn_x
        elif dset == 'train':
            x_vals = []
            for val in self.__sets.learn_x:
                x_vals.append(val)
            for val in self.__sets.valid_x:
                x_vals.append(val)
            return np.asarray(x_vals)
        elif dset is None:
            x_vals = []
            for val in self.__sets.learn_x:
                x_vals.append(val)
            for val in self.__sets.valid_x:
                x_vals.append(val)
            for val in self.__sets.test_x:
                x_vals.append(val)
            return np.asarray(x_vals)
        else:
            raise ValueError('Unknown dset argument {}'.format(dset))

    def __determine_y_vals(self, dset):
        '''
        Private method: Helper function for determining which data set target
        data will be passed to functions
        '''

        if dset == 'test':
            return self.__sets.test_y
        elif dset == 'valid':
            return self.__sets.valid_y
        elif dset == 'learn':
            return self.__sets.learn_y
        elif dset == 'train':
            y_vals = []
            for val in self.__sets.learn_y:
                y_vals.append(val)
            for val in self.__sets.valid_y:
                y_vals.append(val)
            return np.asarray(y_vals)
        elif dset is None:
            y_vals = []
            for val in self.__sets.learn_y:
                y_vals.append(val)
            for val in self.__sets.valid_y:
                y_vals.append(val)
            for val in self.__sets.test_y:
                y_vals.append(val)
            return np.asarray(y_vals)
        else:
            raise ValueError('Unknown dset argument {}'.format(dset))

    def __create_model(self):
        '''
        Private method: Helper function for creating a neural network
        '''

        model = ecnet.model.MultilayerPerceptron()
        model.add_layer(
            self.DataFrame.num_inputs,
            self.vars['input_activation']
        )
        for layer in self.vars['hidden_layers']:
            model.add_layer(
                layer[0],
                layer[1]
            )
        model.add_layer(
            self.DataFrame.num_targets,
            self.vars['output_activation']
        )
        model.connect_layers()
        return model

    def __error_fn(self, arg, y_hat, y):
        '''
        Private method: Parses error argument, calculates corresponding error
        and returns it
        '''

        if arg == 'rmse':
            return ecnet.error_utils.calc_rmse(y_hat, y)
        elif arg == 'r2':
            return ecnet.error_utils.calc_r2(y_hat, y)
        elif arg == 'mean_abs_error':
            return ecnet.error_utils.calc_mean_abs_error(y_hat, y)
        elif arg == 'med_abs_error':
            return ecnet.error_utils.calc_med_abs_error(y_hat, y)
        else:
            raise ValueError('Unknown error function {}'.format(arg))
