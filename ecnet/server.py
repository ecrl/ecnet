#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/server.py
# v.2.0.0
# Developed in 2018 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains the "Server" class, which handles ECNet project creation, neural
# network model creation, data hand-off to models, and prediction error
# calculation. For example scripts, refer to https://github.com/tjkessler/ecnet
#

# Stdlib imports
from os import listdir, makedirs, name, path, walk
from zipfile import ZipFile, ZIP_DEFLATED
from pickle import dump as pdump, load as pload
from multiprocessing import Pool, set_start_method
from shutil import rmtree

# 3rd party imports
from yaml import dump, load
from numpy import asarray
from ecabc.abc import ABC
from colorlogging import ColorLogger

# ECNet imports
import ecnet.data_utils
import ecnet.error_utils
import ecnet.model
import ecnet.limit_parameters
from ecnet.fitness_functions import tune_hyperparameters


class Server:

    def __init__(self, config_filename='config.yml', project_file=None,
                 log_level='info', log_dir=None, num_processes=1):
        '''Server object: handles data importing, neural network creation, data
        to neural network hand-off, error calculations, project saving and
        loading

        Args:
            config_filename (str): (optional) path of neural network
                configuration file
            project_file (str): (optional) path of pre-existing project
            log_level (str): 'disable', 'debug', 'info', 'warn', 'error',
                'crit'
            log_dir (None or str): directory to save logs; defaults to not
                saving logs
            num_processes (int): number of concurrent processes to run various
                methods with, including training, input dimensionality
                reduction, and hyperparameter tuning
        '''

        self._logger = ColorLogger(stream_level=log_level)
        self.log_dir = log_dir
        self.log_level = log_level
        self.num_processes = num_processes

        if name != 'nt':
            set_start_method('spawn', force=True)

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
            self._logger.log(
                'warn',
                'Supplied configuration file not found',
                call_loc={'call_loc': 'INIT'}
            )
            self._logger.log(
                'warn',
                'Generating default configuration for {}'.format(
                    config_filename
                ),
                call_loc={'call_loc': 'INIT'}
            )
            config_dict = {
                'learning_rate': 0.1,
                'keep_prob': 0.0,
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

    @property
    def log_level(self):
        '''tuple: (stream log level, file log level)
        '''

        return (self._logger.stream_level, self._logger.file_level)

    @log_level.setter
    def log_level(self, level):
        '''Args:
            level (str): 'disable', 'debug', 'info', 'warn', 'error', 'crit'
        '''

        self._logger.stream_level = level
        if self.log_dir is not None:
            self._logger.file_level = level

    @property
    def log_dir(self):
        '''str or None: log directory, or None to disable file logging
        '''

        if self._logger.file_level == 'disable':
            return None
        return self._logger.log_dir

    @log_dir.setter
    def log_dir(self, log_dir):
        '''Args:
            log_dir (str or None): location for file logging; if None, turns
                off file logging
        '''

        if log_dir is None:
            self._logger.file_level = 'disable'
        else:
            self._logger.log_dir = log_dir
            self._logger.file_level = self.log_level[0]

    @property
    def num_processes(self):
        '''Returns int: number of processors Server will utilize for training,
            tuning, and input dim reduction
        '''

        return self.__num_processes

    @num_processes.setter
    def num_processes(self, num):
        '''Args:
            num (int): number of processes to utilize for training, tuning,
                and input dim reduction
        '''

        assert type(num) is int, \
            'Invalid process number type: {}'.format(type(num))
        self.__num_processes = num

    @property
    def size(self):
        '''(# builds, # nodes, # candidates) if using project, else None
        '''

        if self.__using_project:
            return (self.__num_builds, self.__num_nodes, self.__num_candidates)
        else:
            return None

    def create_project(self, project_name, num_builds=1, num_nodes=5,
                       num_candidates=10):
        '''Creates the folder structure for a project

        Args:
            project_name (str): name of the project
            num_builds (int): number of builds in the project
            num_nodes (int): number of nodes for each build
            num_candidates (int): number of candidate neural networks per node
        '''

        assert type(project_name) is str, \
            'Invalid project_name type: {}'.format(type(project_name))
        assert type(num_builds) is int and type(num_nodes) is int \
            and type(num_candidates) is int, \
            'Invalid project structure: {}*{}*{}'.format(
                num_builds, num_nodes, num_candidates
            )
        self.__project_name = project_name
        self.__num_builds = num_builds
        self.__num_nodes = num_nodes
        self.__num_candidates = num_candidates
        for build in range(self.__num_builds):
            for node in range(self.__num_nodes):
                for candidate in range(self.__num_candidates):
                    folder = path.join(
                        self.__project_name,
                        'build_{}'.format(build + 1),
                        'node_{}'.format(node + 1),
                        'candidate_{}'.format(candidate + 1)
                    )
                    if not path.exists(folder):
                        makedirs(folder)
        self.__using_project = True
        self._logger.log(
            'info',
            'Created project {}'.format(project_name),
            call_loc={'call_loc': 'PROJECT'}
        )

    def import_data(self, data_filename, sort_type='random',
                    data_split=[0.65, 0.25, 0.1]):
        '''Imports data from ECNet formatted CSV database

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
            raise ValueError('Invalid sort_type {}'.format(sort_type))
        self.__sets = self.DataFrame.package_sets()
        self._logger.log(
            'info',
            'Imported data from {}'.format(data_filename),
            call_loc={'call_loc': 'IMPORT'}
        )

    def limit_input_parameters(self, limit_num, output_filename=None,
                               use_genetic=False, population_size=500,
                               num_generations=25, shuffle=False,
                               data_split=[0.7, 0.2, 0.1]):
        '''Limits the input dimensionality of currently loaded data; default
        method is an iterative inclusion algorithm, options for using a genetic
        algorithm available.

        Args:
            limit_num (int): desired input dimensionality
            output_filename (str): path to resulting limited database
            use_genetic (bool): whether to use genetic algorithm instead of
                iterative inclusion algorithm
            population_size (int): if use_genetic, size of genetic population
            num_generations (int): number of generations to run the GA for
            shuffle (bool): whether to shuffle the data splits for each
                population member
            data_split (list): [learn%, valid%, test%] for splits if shuffle ==
                True

        See https://github.com/tjkessler/pygenetics for genetic algorithm
        source code.
        '''

        assert type(limit_num) is int, \
            'Invalid limit_num type: {}'.format(type(limit_num))
        assert type(use_genetic) is bool, \
            'Invalid use_genetic type: {}'.format(type(use_genetic))
        assert type(shuffle) is bool, \
            'Invalid shuffle type: {}'.format(type(shuffle))

        self._logger.log('debug', 'Limiting input dimensionality to {}'.format(
            limit_num
        ), call_loc={'call_loc': 'LIMIT'})

        if use_genetic:
            self._logger.log(
                'info',
                'Limiting input dimensionality using a genetic algorithm',
                call_loc={'call_loc': 'LIMIT'}
            )
            params = ecnet.limit_parameters.limit_genetic(
                self.DataFrame, limit_num, self.vars, population_size,
                num_generations, self.__num_processes, shuffle=shuffle,
                data_split=data_split, logger=self._logger
            )
        else:
            self._logger.log(
                'info',
                'Limiting input dimensionality using iterative inclusion',
                call_loc={'call_loc': 'LIMIT'}
            )
            params = ecnet.limit_parameters.limit_iterative_include(
                self.DataFrame, limit_num, self.vars, logger=self._logger
            )
        self.DataFrame.set_inputs(params)
        self.__sets = self.DataFrame.package_sets()
        if output_filename is not None:
            self.DataFrame.save(output_filename)
            self._logger.log(
                'info',
                'Saved limited database to {}'.format(output_filename),
                call_loc={'call_loc': 'LIMIT'}
            )
        return params

    def tune_hyperparameters(self, target_score=None, num_iterations=50,
                             num_employers=50, shuffle=False, data_split=None):
        '''Tunes the neural network learning hyperparameters (learning_rate,
        validation_max_epochs, keep_prob, neuron counts for each hidden
        layer) using an artificial bee colony search algorithm

        Args:
            target_score (float or None): fitness required to stop the colony
            num_iterations (int): if !target_score, number of iterations to
                run the colony
            num_employers (int): number of employer bees for the colony
            shuffle (bool): True == shuffle all data for each bee
            data_split (list): [learn%, valid%, test%] for splits if shuffle ==
                True

        Returns:
            tuple: (learning_rate, validation_max_epochs, neuron counts)
                derived from running the colony (also set as current vals)

        See https://github.com/ecrl/ecabc for ABC source code.
        '''

        assert type(target_score) is (int or float) or target_score is None, \
            'Invalid target_score type: {}'.format(target_score)
        assert type(num_iterations) is int, \
            'Invalid num_iterations type: {}'.format(num_iterations)
        assert type(shuffle) is bool, \
            'Invalid shuffle type: {}'.format(type(shuffle))

        hyperparameters = [
            ('float', (0.01, 0.2)),
            ('int', (1000, 25000)),
            ('float', (0.0, 1.0))
        ]
        for _ in range(len(self.vars['hidden_layers'])):
            hyperparameters.append(('int', (8, 32)))

        cost_fn_args = {
            'DataFrame': self.DataFrame,
            'packaged_data': self.__sets,
            'shuffle': shuffle,
            'data_split': data_split,
            'num_processes': self.__num_processes,
            'hidden_layers': self.vars['hidden_layers'],
            'input_activation': self.vars['input_activation'],
            'output_activation': self.vars['output_activation']
        }

        abc = ABC(
            tune_hyperparameters,
            hyperparameters,
            print_level=self.log_level[0],
            file_logging=self.log_level[1],
            processes=self.__num_processes,
            args=cost_fn_args
        )
        abc.num_employers = num_employers

        self._logger.log(
            'info',
            'Tuning neural network hyperparameters with an ABC',
            call_loc={'call_loc': 'TUNE'}
        )

        abc.create_employers()
        if target_score is None:
            for _ in range(num_iterations):
                abc.run_iteration()
        else:
            while True:
                abc.run_iteration()
                if (abc.best_performer[0] <= target_score):
                    break

        new_hyperparameters = abc.best_performer[1]

        self.vars['learning_rate'] = new_hyperparameters[0]
        self.vars['validation_max_epochs'] = new_hyperparameters[1]
        self.vars['keep_prob'] = new_hyperparameters[2]
        for idx, layer in enumerate(self.vars['hidden_layers'], 3):
            layer[0] = new_hyperparameters[idx]

        self._logger.log(
            'debug',
            'Tuned learning rate: {}'.format(new_hyperparameters[0]),
            call_loc={'call_loc': 'TUNE'}
        )
        self._logger.log(
            'debug',
            'Tuned max validation epochs: {}'.format(new_hyperparameters[1]),
            call_loc={'call_loc': 'TUNE'}
        )
        self._logger.log(
            'debug',
            'Tuned neuron keep probability: {}'.format(new_hyperparameters[2]),
            call_loc={'call_loc': 'TUNE'}
        )
        for idx, layer in enumerate(self.vars['hidden_layers'], 3):
            self._logger.log(
                'debug',
                'Tuned number of neurons in hidden layer {}: {}'.format(
                    idx - 2, new_hyperparameters[idx]
                ),
                call_loc={'call_loc': 'TUNE'}
            )

        return new_hyperparameters

    def train_model(self, validate=False, shuffle=None,
                    data_split=[0.65, 0.25, 0.1]):
        '''Trains neural network(s): if a project was created, trains
        (builds * nodes * candidates) neural networks, else 1

        Args:
            validate (bool): whether to use periodic validation to determine
                learning cutoff
            shuffle (str): 'train' to shuffle learning and validation sets,
                'all' to shuffle learning, validation and test sets
            data_split (list): [learn%, valid%, test%] if shuffle == True
        '''

        assert type(validate) is bool, \
            'Invalid validate type: {}'.type(validate)

        if not self.__using_project:
            self._logger.log(
                'info',
                'Training single model',
                call_loc={'call_loc': 'TRAIN'}
            )
            ecnet.model.train_model(
                validate,
                self.__sets,
                self.vars,
                save_path='./tmp/model'
            )

        else:
            self._logger.log(
                'info',
                'Generating {} x {} x {} neural networks'.format(
                    self.__num_builds, self.__num_nodes,
                    self.__num_candidates
                ),
                call_loc={'call_loc': 'TRAIN'}
            )

            if self.__num_processes > 1:
                train_pool = Pool(processes=self.__num_processes)

            for build in range(self.__num_builds):
                for node in range(self.__num_nodes):
                    for candidate in range(self.__num_candidates):
                        model_path = path.join(
                            self.__project_name,
                            'build_{}'.format(build + 1),
                            'node_{}'.format(node + 1),
                            'candidate_{}'.format(candidate + 1),
                            'model'
                        )
                        if self.__num_processes > 1:
                            train_pool.apply_async(
                                ecnet.model.train_model,
                                [
                                    validate,
                                    self.__sets,
                                    self.vars,
                                    model_path
                                ]
                            )
                        else:
                            self._logger.log(
                                'debug',
                                'Build {}, Node {}, candidate {}'.format(
                                    build + 1, node + 1, candidate + 1
                                ),
                                call_loc={'call_loc': 'TRAIN'}
                            )
                            ecnet.model.train_model(
                                validate,
                                self.__sets,
                                self.vars,
                                save_path=model_path
                            )
                        if shuffle is not None:
                            self.DataFrame.shuffle(
                                shuffle, split=data_split
                            )

            if self.__num_processes > 1:
                train_pool.close()
                train_pool.join()

    def select_best(self, dset=None, error_fn='mean_abs_error'):
        '''Selects the best performing neural network candidate from each node for
        each build (requires create_project())

        Args:
            dset (str): which data set performance to use when selecting models
                can choose 'learn', 'valid', 'test', 'train' (learning and
                validation) or None (uses all data sets)
            error_fn (str): which error function to use when measuring model
                performance; 'mean_abs_error', 'med_abs_error', 'rmse'
        '''

        if not self.__using_project:
            raise Exception('Project has not been created! (create_project())')
        if not path.exists(
            path.join(self.__project_name, path.join('build_1', path.join(
                        'node_1',
                        'candidate_1',
                        'model.meta'
            )))
        ):
            raise Exception('Models must be trained first! (train_model())')
        if error_fn != 'mean_abs_error' and error_fn != 'med_abs_error' \
           and error_fn != 'rmse':
            raise ValueError(
                '{} is not a valid error function'.format(error_fn)
            )
        self._logger.log(
            'info',
            'Selecting best models from each node for each build',
            call_loc={'call_loc': 'SELECTION'}
        )
        x_vals = self.__determine_x_vals(dset)
        y_vals = self.__determine_y_vals(dset)

        for build in range(self.__num_builds):
            for node in range(self.__num_nodes):
                min_error = None
                best_candidate = None
                for candidate in range(self.__num_candidates):
                    model_path = path.join(
                        self.__project_name,
                        'build_{}'.format(build + 1),
                        'node_{}'.format(node + 1),
                        'candidate_{}'.format(candidate + 1),
                        'model'
                    )
                    model = self.__create_model()
                    model.load(model_path, use_arch_file=False)
                    error = self.__error_fn(
                        error_fn,
                        model.use(x_vals),
                        y_vals
                    )
                    if min_error is None or error < min_error:
                        min_error = error
                        best_candidate = model_path
                model = self.__create_model()
                model.load(best_candidate, use_arch_file=False)
                model_path = path.join(
                    self.__project_name,
                    'build_{}'.format(build + 1),
                    'node_{}'.format(node + 1),
                    'model'
                )
                model.save(model_path)

    def use_model(self, dset=None, output_filename=None):
        '''Uses model(s) to predict for a data set; if using a project, models
        must be selected first with select_best()

        Args:
            dset (str): which data set performance to use when selecting models
                can choose 'learn', 'valid', 'test', 'train' (learning and
                validation) or None (uses all data sets)
            output_filename (str): path to save results, if not None

        Returns:
            list: list of lists, where each sublist is a specific item's
                prediction with a length of the number of DB targets
        '''

        if dset is None:
            self._logger.log(
                'info',
                'Predicting values for all data',
                call_loc={'call_loc': 'USE'}
            )
        else:
            self._logger.log(
                'info',
                'Predicting values for {} set'.format(dset),
                call_loc={'call_loc': 'USE'}
            )
        results = self.__use(dset)
        if output_filename is not None:
            ecnet.data_utils.save_results(
                results,
                dset,
                self.DataFrame,
                output_filename
            )
        return results

    def calc_error(self, *args, dset=None):
        '''Calculates errors for data sets

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

        for arg in args:
            if dset is None:
                self._logger.log(
                    'info',
                    'Calculating {} for all data'.format(arg),
                    call_loc={'call_loc': 'METRICS'}
                )
            else:
                self._logger.log(
                    'info',
                    'Calculating {} for {} set'.format(arg, dset),
                    call_loc={'call_loc': 'METRICS'}
                )
        error_dict = {}
        y_hat = self.__use(dset)
        y = self.__determine_y_vals(dset)
        for arg in args:
            if self.__using_project:
                error_list = []
                for y_build in y_hat:
                    error_list.append(self.__error_fn(arg, y_build, y))
                error_dict[arg] = error_list
            else:
                error_dict[arg] = self.__error_fn(arg, y_hat, y)
            self._logger.log(
                'debug',
                '{} : {}'.format(arg, error_dict[arg]),
                call_loc={'call_loc': 'METRICS'}
            )
        return error_dict

    def save_project(self, filename=None, clean_up=False):
        '''Saves the current state of the Server (loaded data, model
        configuration) and all models built to a ".prj"

        Args:
            filename (str): path to .prj file location; if None, saves to
                %PROJECT_NAME%.prj in the current working directory
            clean_up (bool): whether to remove the project's folder structure
                after it is zipped up
        '''

        if not self.__using_project:
            raise FileExistsError(
                'Project has not been created with create_project()'
            )

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

        if filename is not None:
            if '.prj' not in filename:
                filename += '.prj'
            save_file = filename
        else:
            save_file = '{}.prj'.format(self.__project_name)
        zip_file = ZipFile(save_file, 'w', ZIP_DEFLATED)
        for root, _, files in walk(self.__project_name):
            for file in files:
                zip_file.write(path.join(root, file))
        zip_file.close()

        if clean_up:
            rmtree(self.__project_name)
            if path.exists('./tmp/'):
                rmtree('./tmp/')

        self._logger.log(
            'info',
            'Project saved to {}'.format(save_file),
            call_loc={'call_loc': 'PROJECT'}
        )

    def __open_project(self, project_name):
        '''Private method: Opens a .prj file, imports data and model
        configuration, and populates project folder with saved models

        Args:
            project_name (string): path to .prj file
        '''

        self.__project_name = project_name.replace('.prj', '')
        if '.prj' not in project_name:
            project_name += '.prj'

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

        for _, _, files in walk(self.__project_name):
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
        self._logger.log(
            'info',
            'Opened project {}'.format(project_name),
            call_loc={'call_loc': 'PROJECT'}
        )

    def __determine_x_vals(self, dset):
        '''Private method: Helper function for determining which data set will
        be passed to functions

        Args:
            dset (str): 'learn', 'valid', 'train', 'test', None (all)

        Returns:
            numpy array: all data points from the specified set
        '''

        assert dset in ['learn', 'valid', 'train', 'test', None], \
            'Invalid dset argument: {}'.format(dset)

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
        '''Private method: Helper function for determining which data set target
        data will be passed to functions

        Args:
            dset (str): 'learn', 'valid', 'train', 'test', None (all)

        Returns:
            numpy array: all data points from the specified set
        '''

        assert dset in ['learn', 'valid', 'train', 'test', None], \
            'Invalid dset argument: {}'.format(dset)

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
        '''Private method: Helper function for creating a neural network
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

    def __use(self, dset):
        '''Private method: used to obtain predictions for a set in loaded data

        Args:
            dset (str): 'learn', 'valid', 'train', 'test', None

        Returns list: results for each set element, each being a list of
            length equal to number of targets (size of neural network output
            layer)
        '''

        x_vals = self.__determine_x_vals(dset)
        if not self.__using_project:
            model = self.__create_model()
            model.load('./tmp/model', use_arch_file=False)
            return [model.use(x_vals)]
        else:
            if not path.exists(path.join(
               self.__project_name, 'build_1', 'node_1', 'model.meta')):
                raise FileNotFoundError(
                    'Select best performers using select_best()'
                )
            preds = []
            for build in range(self.__num_builds):
                build_preds = []
                for node in range(self.__num_nodes):
                    model_path = path.join(
                        self.__project_name,
                        'build_{}'.format(build + 1),
                        'node_{}'.format(node + 1),
                        'model'
                    )
                    model = self.__create_model()
                    model.load(model_path, use_arch_file=False)
                    build_preds.append(model.use(x_vals))
                ave_build_preds = []
                for pred in range(len(build_preds[0])):
                    node_preds = []
                    for node in range(len(build_preds)):
                        node_preds.append(build_preds[node][pred])
                    ave_build_preds.append(sum(node_preds)/len(node_preds))
                preds.append(list(ave_build_preds))
            return preds

    @staticmethod
    def __error_fn(fn, y_hat, y):
        '''Private, static method: Parses error argument, calculates
        corresponding error and returns it

        Args:
            fn (str): 'rmse', 'mean_abs_error', 'med_abs_error', 'r2'

        Returns:
            float: calculated error
        '''

        assert fn in ['rmse', 'mean_abs_error', 'med_abs_error', 'r2'], \
            'Invalid error function: {}'.format(fn)

        if fn == 'rmse':
            return ecnet.error_utils.calc_rmse(y_hat, y)
        elif fn == 'r2':
            return ecnet.error_utils.calc_r2(y_hat, y)
        elif fn == 'mean_abs_error':
            return ecnet.error_utils.calc_mean_abs_error(y_hat, y)
        elif fn == 'med_abs_error':
            return ecnet.error_utils.calc_med_abs_error(y_hat, y)
        else:
            raise ValueError('Unknown error function {}'.format(fn))
