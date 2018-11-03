#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/server.py
# v.1.6.0
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains the "Server" class, which handles ECNet project creation, neural
# network model creation, data hand-off to models, and prediction error
# calculation. For example scripts, refer to https://github.com/tjkessler/ecnet
#

# Stdlib imports
from os import listdir, makedirs, path, remove, walk
from zipfile import ZipFile, ZIP_DEFLATED
from pickle import dump as pdump, load as pload

# 3rd party imports
from yaml import dump, load
from numpy import asarray
from ecabc.abc import ABC
from colorlogging import log

# ECNet imports
import ecnet.data_utils
import ecnet.error_utils
import ecnet.model
import ecnet.limit_parameters


class Server:

    def __init__(self, config_filename='config.yml', project_file=None,
                 log_progress=True):
        '''
        Server object: handles data importing, model creation, data to model
        hand-off, error calculations, project saving and loading

        Args:
            config_filename (str): (optional) path of model configuration file
            project_file (str): (optional) path of pre-existing project
            log_progress (bool): whether to use console and file logging or not
        '''

        self.__log = log_progress

        if project_file is not None:
            self.__open_project(project_file)
            return

        self.vars = {}
        if '.yml' not in config_filename:
            config_filename += '.yml'
        try:
            file = open(config_filename, 'r')
            self.vars.update(load(file))
        except:
            if self.__log:
                log('warn', 'Supplied configuration file not found - '
                            'generating default configuration for {}'.format(
                                config_filename
                            ), use_color=False)
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
            dump(config_dict, file)
            self.vars.update(config_dict)

        self.__config_filename = config_filename
        self.__using_project = False

    def create_project(self, project_name, num_builds=1, num_nodes=5,
                       num_trials=10):
        '''
        Creates the folder structure for a project

        Args:
            project_name (str): name of the project
            num_builds (int): number of builds in the project
            num_nodes (int): number of nodes for each build
            num_trials (int): number of trials per node
        '''

        self.__project_name = project_name
        self.__num_builds = num_builds
        self.__num_nodes = num_nodes
        self.__num_trials = num_trials
        if not path.exists(self.__project_name):
            makedirs(self.__project_name)
        for build in range(self.__num_builds):
            path_b = path.join(self.__project_name, 'build_{}'
                               .format(build + 1))
            if not path.exists(path_b):
                makedirs(path_b)
            for node in range(self.__num_nodes):
                path_n = path.join(path_b, 'node_{}'.format(node + 1))
                if not path.exists(path_n):
                    makedirs(path_n)
        self.__using_project = True
        if self.__log:
            log('info', 'Created project {}'.format(project_name),
                use_color=False)

    def import_data(self, data_filename, sort_type='random',
                    data_split=[0.65, 0.25, 0.1]):
        '''
        Imports data from ECNet formatted CSV database

        Args:
            data_filename (str): path to CSV database
            sort_type (str): 'random' or 'explicit' (DB specified) sorting
            data_split (list): [learn%, valid%, test%] if sort_type == True
        '''

        self.DataFrame = ecnet.data_utils.DataFrame(data_filename)
        if sort_type == 'random':
            self.DataFrame.create_sets(random=True, split=data_split)
        elif sort_type == 'explicit':
            self.DataFrame.create_sets(random=False)
        else:
            raise ValueError('Unknown sort_type {}'.format(sort_type))
        self.__sets = self.DataFrame.package_sets()
        if self.__log:
            log('info', 'Imported data from {}'.format(data_filename),
                use_color=False)

    def limit_input_parameters(self, limit_num, output_filename,
                               use_genetic=False, population_size=500,
                               num_generations=25, num_processes=0,
                               shuffle=False, data_split=[0.65, 0.25, 0.1]):
        '''
        Limits the input dimensionality of currently loaded DataFrame; default
        method is an iterative inclusion algorithm, options for using a genetic
        algorithm available.

        Args:
            limit_num (int): desired input dimensionality
            output_filename (str): path to resulting limited database
            use_genetic (bool): whether to use genetic algorithm instead of
                                iterative inclusion algorithm
            population_size (int): if use_genetic, size of genetic population
            num_generations (int): number of generations to run the GA for
            num_processes (int): number of concurrent processes for population
                                 member generation/evaluation
            shuffle (bool): whether to shuffle the data splits for each
                            population member
            data_split (list): [learn%, valid%, test%] for splits if shuffle ==
                               True

        See https://github.com/tjkessler/pygenetics for genetic algorithm
        source code.
        '''

        if use_genetic:
            if self.__log:
                log(
                    'info',
                    'Limiting input parameters using a genetic algorithm',
                    use_color=False
                )
            params = ecnet.limit_parameters.limit_genetic(
                self.DataFrame, limit_num, population_size, num_generations,
                num_processes, shuffle=shuffle, log_progress=self.__log
            )
        else:
            if self.__log:
                log(
                    'info',
                    'Limiting input parameters using iterative inclusion',
                    use_color=False
                )
            params = ecnet.limit_parameters.limit_iterative_include(
                self.DataFrame, limit_num, log_progress=self.__log
            )
        ecnet.limit_parameters.output(self.DataFrame, params, output_filename)
        if self.__log:
            log('info', 'Saved limited database to {}'.format(output_filename),
                use_color=False)

    def tune_hyperparameters(self, target_score=None, num_iterations=50,
                             num_employers=50):
        '''
        Tunes the neural network learning hyperparameters (learning_rate,
        validation_max_epochs, neuron counts for each hidden layer) using an
        artificial bee colony searching algorithm.

        Args:
            target_score (float): fitness required to stop the colony
            num_iterations (int): if !target_score, number of iterations to
                                  run the colony
            num_employers (int): number of employer bees for the colony

        Returns:
            tuple: (learning_rate, validation_max_epochs, neuron count)
                   derived from running the colony (also set as current vals)

        See https://github.com/ecrl/ecabc for ABC source code.
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

        if self.__log:
            log('info', 'Tuning neural network hyperparameters with an ABC',
                use_color=False)

        if target_score is None:
            abc = ABC(
                iterationAmount=num_iterations,
                fitnessFunction=test_neural_network,
                valueRanges=hyperparameters,
                amountOfEmployers=num_employers
            )
        else:
            abc = ABC(
                endValue=target_score,
                fitnessFunction=test_neural_network,
                valueRanges=hyperparameters,
                amountOfEmployers=num_employers
            )

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

        if self.__log:
            log('info', 'Tuned learning rate: {}'.format(
                new_hyperparameters[0]), use_color=False)
            log('info', 'Tuned max validation epochs: {}'.format(
                new_hyperparameters[1]), use_color=False)
            for idx, layer in enumerate(self.vars['hidden_layers'], 2):
                log(
                    'info',
                    'Tuned number of neurons in hidden layer {}: {}'.format(
                        idx - 1, new_hyperparameters[idx]
                    ), use_color=False)

        return new_hyperparameters

    def train_model(self, validate=False, shuffle=None,
                    data_split=[0.65, 0.25, 0.1]):
        '''
        Trains models (fills project if create_project() called, otherwise
        creates one neural network)

        Args:
            validate (bool): whether to use periodic validation to determine
                             learning cutoff
            shuffle (bool): whether to shuffle the data sets for each trial
            data_split (list): [learn%, valid%, test%] if shuffle == True
        '''

        if not self.__using_project:
            if self.__log:
                log('info', 'Training single model', use_color=False)
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
            if self.__log:
                log(
                    'info',
                    'Generating {} x {} x {} neural networks'.format(
                        self.__num_builds, self.__num_nodes, self.__num_trials
                    ),
                    use_color=False)
            for build in range(self.__num_builds):
                path_b = path.join(
                    self.__project_name, 'build_{}'.format(build + 1)
                )
                for node in range(self.__num_nodes):
                    path_n = path.join(
                        path_b, 'node_{}'.format(node + 1)
                    )
                    for trial in range(self.__num_trials):
                        if self.__log:
                            log(
                                'info',
                                'Build {}, Node {}, Trial {}'.format(
                                    build + 1, node + 1, trial + 1
                                ),
                                use_color=False
                            )
                        path_t = path.join(
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

        Args:
            dset (str): which data set performance to use when selecting models
                can choose 'learn', 'valid', 'test', 'train' (learning and
                validation) or None (uses all data sets)
            error_fn (str): which error function to use when measuring model
                            performance
                can choose 'mean_abs_error', 'med_abs_error', 'rmse'
        '''

        if not self.__using_project:
            raise Exception('Project has not been created! (create_project())')
        if not path.exists(
            path.join(
                self.__project_name,
                path.join('build_1', path.join(
                        'node_1',
                        'trial_1.meta'
                    )
                )
            )
        ):
            raise Exception('Models must be trained first! (train_model())')
        if self.__log:
            log('info', 'Selecting best models from each mode for each build',
                use_color=False)
        x_vals = self.__determine_x_vals(dset)
        y_vals = self.__determine_y_vals(dset)
        for build in range(self.__num_builds):
            path_b = path.join(
                self.__project_name, 'build_{}'.format(build + 1)
            )
            for node in range(self.__num_nodes):
                path_n = path.join(
                    path_b, 'node_{}'.format(node + 1)
                )
                error_list = []
                for trial in range(self.__num_trials):
                    path_t = path.join(
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
                path_t_best = path.join(
                    path_n, 'trial_{}'.format(current_min_idx + 1)
                )
                path_best = path.join(
                    path_n, 'model'
                )
                model.load(path_t_best)
                model.save(path_best)

    def use_model(self, dset=None):
        '''
        Uses model(s) to predict for a data set; if using a project, models
        must be selected first

        Args:
            dset (str): which data set performance to use when selecting models
                can choose 'learn', 'valid', 'test', 'train' (learning and
                validation) or None (uses all data sets)

        Returns:
            list: list of lists, where each sublist is a specific item's
                  prediction with a length of the number of DB targets
        '''

        if self.__log:
            log('info', 'Predicting values for {} set'.format(dset),
                use_color=False)
        x_vals = self.__determine_x_vals(dset)
        if not self.__using_project:
            model = ecnet.model.MultilayerPerceptron()
            model.load('./tmp/model')
            return [model.use(x_vals)]
        else:
            if not path.exists(
                path.join(
                    self.__project_name,
                    path.join('build_1', path.join(
                            'node_1',
                            'model.meta'
                        )
                    )
                )
            ):
                raise Exception('Select best performers using select_best()')
            preds = []
            for build in range(self.__num_builds):
                path_b = path.join(
                    self.__project_name, 'build_{}'.format(build + 1)
                )
                build_preds = []
                for node in range(self.__num_nodes):
                    path_n = path.join(
                        path_b, 'node_{}'.format(node + 1)
                    )
                    path_best = path.join(
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
        Calculates errors for data sets

        Args:
            *args (str): any number of error functions
                can choose: 'rmse', 'r2' (r-squared), 'mean_abs_error',
                'med_abs_error'
            dset (str): which data set performance to use when selecting models
                can choose 'learn', 'valid', 'test', 'train' (learning and
                validation) or None (uses all data sets)

        Returns:
            dictionary: dictionary of supplied error functions and their values
        '''

        if self.__log:
            for arg in args:
                log('info', 'Calculating {} for {} set'.format(arg, dset),
                    use_color=False)
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
        Saves results obtained from use_model()

        Args:
            results (list): list of results obtained with use_model()
            filename (string): path to location to save results CSV
        '''

        ecnet.data_utils.save_results(results, self.DataFrame, filename)
        if self.__log:
            log('info', 'Results saved to {}'.format(filename),
                use_color=False)

    def save_project(self, clean_up=True):
        '''
        Saves the current state of the Server (loaded data, model
        configuration) and all created models to a zipped ".project" file

        Args:
            clean_up (bool): whether to remove the project's folder structure
                             after it is zipped up
        '''

        if not self.__using_project:
            raise Exception('Project has not been created! (create_project())')

        if clean_up:
            for build in range(self.__num_builds):
                path_b = path.join(
                    self.__project_name, 'build_{}'.format(build + 1)
                )
                for node in range(self.__num_nodes):
                    path_n = path.join(
                        path_b, 'node_{}'.format(node + 1)
                    )
                    trial_files = [
                        file for file in listdir(path_n) if 'trial' in file
                    ]
                    for file in trial_files:
                        remove(path.join(path_n, file))

        with open(
            path.join(self.__project_name, self.__config_filename),
            'w'
        ) as config_save:
            dump(
                self.vars,
                config_save,
                default_flow_style=False,
                explicit_start=True
            )
        config_save.close()

        with open(
            path.join(self.__project_name, 'data.d'),
            'wb'
        ) as data_save:
            pdump(self.DataFrame, data_save)
        data_save.close()

        zip_file = ZipFile(
            '{}.project'.format(self.__project_name),
            'w',
            ZIP_DEFLATED
        )
        for root, dirs, files in walk(self.__project_name):
            for file in files:
                zip_file.write(path.join(root, file))
        zip_file.close()
        if self.__log:
            log('info', 'Project saved to {}.project'.format(
                    self.__project_name
                ), use_color=False)

    def __open_project(self, project_name):
        '''
        Private method: Opens a .project file, imports saved data and model
        configuration, and unpacks the folder structure containing models

        Args:
            project_name (string): path to .project file
        '''

        self.__project_name = project_name.replace('.project', '')
        if '.project' not in project_name:
            project_name += '.project'

        zip_file = ZipFile(project_name, 'r')
        zip_file.extractall(self.__project_name + '\\..\\')
        zip_file.close()

        self.__num_builds = len(
            [build for build in listdir(
                self.__project_name
            ) if path.isdir(path.join(self.__project_name, build))]
        )
        self.__num_nodes = len(
            [node for node in listdir(
                path.join(self.__project_name, 'build_1')
            ) if path.isdir(path.join(
                self.__project_name,
                path.join('build_1', node))
            )]
        )

        for root, dirs, files in walk(self.__project_name):
            for file in files:
                if '.yml' in file:
                    self.__config_filename = file
                    break

        with open(
            path.join(self.__project_name, self.__config_filename),
            'r'
        ) as config_file:
            self.vars = {}
            self.vars.update(load(config_file))
        config_file.close()

        with open(
            path.join(self.__project_name, 'data.d'),
            'rb'
        ) as data_file:
            self.DataFrame = pload(data_file)
        data_file.close()

        self.__sets = self.DataFrame.package_sets()
        self.__using_project = True
        if self.__log:
            log('info', 'Opened project {}.project'.format(project_name),
                use_color=False)

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
            return asarray(x_vals)
        elif dset is None:
            x_vals = []
            for val in self.__sets.learn_x:
                x_vals.append(val)
            for val in self.__sets.valid_x:
                x_vals.append(val)
            for val in self.__sets.test_x:
                x_vals.append(val)
            return asarray(x_vals)
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
            return asarray(y_vals)
        elif dset is None:
            y_vals = []
            for val in self.__sets.learn_y:
                y_vals.append(val)
            for val in self.__sets.valid_y:
                y_vals.append(val)
            for val in self.__sets.test_y:
                y_vals.append(val)
            return asarray(y_vals)
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
