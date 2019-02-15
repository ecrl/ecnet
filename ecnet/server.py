#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/server.py
# v.3.0.0
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains the "Server" class, which handles ECNet project creation, neural
# network model creation, data hand-off to models, prediction error
# calculation, input parameter selection, hyperparameter tuning.
#
# For example scripts, refer to https://github.com/ecrl/ecnet/examples
#

# ECNet imports
from ecnet.tasks.limit_inputs import limit_rforest
from ecnet.tasks.remove_outliers import remove_outliers
from ecnet.tasks.tuning import tune_hyperparameters
from ecnet.utils.data_utils import DataFrame, save_results
from ecnet.utils.logging import logger
from ecnet.utils.server_utils import default_config, get_candidate_path,\
    get_error, get_y, open_config, open_df, resave_df, resave_model,\
    save_config, save_df, train_model, use_model

# Stdlib imports
from os import listdir, makedirs, path, walk
from operator import itemgetter
from multiprocessing import Pool
from shutil import rmtree
from zipfile import ZipFile, ZIP_DEFLATED


class Server:

    def __init__(self, model_config='config.yml', prj_file=None,
                 num_processes=1):
        '''Server object: handles data loading, model creation, data-to-model
        hand-off, data input parameter selection, hyperparameter tuning

        Args:
            model_config (str): path to multilayer perceptron .yml config file;
                if not found, default config is generated
            prj_file (str): path to pre-existing ECNet .prj file, if using for
                retraining/new predictions
            num_processes (int): number of parallel processes to utilize for
                training and tuning processes
        '''

        logger.log('debug', 'Arguments:\n\t| model_config:\t\t{}\n\t|'
                   ' prj_file:\t\t{}\n\t| num_processes:\t{}'.format(
                       model_config, prj_file, num_processes
                    ), call_loc='INIT')

        self._num_processes = num_processes

        if prj_file is not None:
            self._open_project(prj_file)
            return

        self._cf_file = model_config
        self._prj_name = None

        self._vars = {}
        try:
            self._vars.update(open_config(self._cf_file))
        except FileNotFoundError:
            logger.log('warn', '{} not found, generating default config'
                       .format(model_config), call_loc='INIT')
            self._vars = default_config()
            save_config(self._vars, self._cf_file)

    def load_data(self, filename, random=False, split=None):
        '''Loads data from an ECNet-formatted CSV database

        Args:
            filename (str): path to CSV database
            random (bool): if True, random set assignments (learn, validate,
                test); if False, uses DB-specified assignmenets
            split (list): if random is True, [learn%, valid%, test%]
        '''

        logger.log('info', 'Loading data from {}'.format(filename),
                   call_loc='LOAD')
        self._df = DataFrame(filename)
        self._df.create_sets(random, split)
        self._sets = self._df.package_sets()

    def create_project(self, project_name, num_pools=1, num_candidates=1):
        '''Creates folder hierarchy for a new project

        Args:
            project_name (str): name of the project, and top-level dir name
            num_pools (int): number of candidate pools for the project
            num_candidates (int): number of candidates per pool
        '''

        self._prj_name = project_name
        self._num_pools = num_pools
        self._num_candidates = num_candidates
        for pool in range(self._num_pools):
            for candidate in range(self._num_candidates):
                pc_dir = get_candidate_path(self._prj_name, pool, candidate)
                if not path.exists(pc_dir):
                    makedirs(pc_dir)
        logger.log('info', 'Created project: {}'.format(project_name),
                   call_loc='PROJECT')
        logger.log('debug', 'Number of pools: {}'.format(num_pools),
                   call_loc='PROJECT')
        logger.log('debug', 'Number of candidates/pool: {}'.format(
                   num_candidates), call_loc='PROJECT')

    def remove_outliers(self, leaf_size=30, output_filename=None):
        '''Removes any outliers from the currently-loaded data using
            unsupervised outlier detection using local outlier factor

        Args:
            leaf_size (int): used by nearest-neighbor algorithm as the number
                of points at which to switch to brute force
            output_filename (str): if not None, database w/o outliers is saved
                here
        '''

        logger.log('info', 'Removing outliers', call_loc='OUTLIERS')
        logger.log('debug', 'Leaf size: {}'.format(leaf_size),
                   call_loc='OUTLIERS')
        self._df = remove_outliers(self._df, leaf_size, self._num_processes)
        if output_filename is not None:
            self._df.save(output_filename)
            logger.log('info', 'Resulting database saved to {}'.format(
                       output_filename), call_loc='OUTLIERS')

    def limit_inputs(self, limit_num, num_estimators=1000,
                     output_filename=None):
        '''Selects `limit_num` influential input parameters using random
        forest regression

        Args:
            limit_num (int): desired number of inputs
            num_estimators (int): number of trees in the RFR algorithm
            output_filename (str): if not None, new limited database is saved
                here
        '''

        logger.log('info', 'Finding {} most influential input parameters'
                   .format(limit_num), call_loc='LIMIT')
        logger.log('debug', 'Number of estimators: {}'.format(num_estimators),
                   call_loc='LIMIT')
        self._df = limit_rforest(
            self._df,
            limit_num,
            num_estimators,
            self._num_processes
        )
        if output_filename is not None:
            self._df.save(output_filename)
            logger.log('info', 'Resulting database saved to {}'.format(
                       output_filename), call_loc='LIMIT')

    def tune_hyperparameters(self, num_employers, num_iterations,
                             shuffle=False, split=None, eval_set=None,
                             eval_fn='rmse'):
        '''Tunes neural network learning hyperparameters using an artificial
        bee colony algorithm; tuned hyperparameters are saved to Server's
        model configuration file

        Args:
            num_employers (int): number of employer bees
            num_iterations (int): number of search iterations for the colony
            shuffle (bool): if True, L/V/T sets are shuffled for each bee and
                their evaluations
            split (list): if shuffle is True, [learn%, valid%, test%]
            eval_set (str): set to evaluate bee fitness; `learn`, `valid`,
                `train`, `test`, None (all sets)
            eval_fn (str): error function used to evaluate bee fitness;
                `rmse`, `mean_abs_error`, `med_abs_error`
        '''

        logger.log('info', 'Tuning architecture/learning hyperparameters',
                   call_loc='TUNE')
        logger.log('debug', 'Arguments:\n\t| num_employers:\t{}\n\t| '
                   'num_iterations:\t{}\n\t| shuffle:\t\t{}\n\t| split:'
                   '\t\t{}\n\t'.format(
                       num_employers, num_iterations, shuffle, split
                   ), call_loc='TUNE')
        params = tune_hyperparameters(
            self._df,
            self._vars,
            num_employers,
            num_iterations,
            self._num_processes,
            shuffle,
            split,
            eval_set,
            eval_fn
        )
        self._vars['beta_1'] = params['beta_1']
        self._vars['beta_2'] = params['beta_2']
        self._vars['decay'] = params['decay']
        self._vars['epsilon'] = params['epsilon']
        self._vars['hidden_layers'][0][0] = params['num_hidden_1']
        self._vars['hidden_layers'][1][0] = params['num_hidden_2']
        self._vars['learning_rate'] = params['learning_rate']
        save_config(self._vars, self._cf_file)

    def train(self, shuffle=None, split=None, retrain=False,
              validate=False, selection_set=None, selection_fn='rmse'):
        '''Trains neural network(s) using currently-loaded data; single NN if
        no project is created, all candidates if created

        Args:
            shuffle (str): `all` to shuffle all sets for each candidate,
                `train` to shuffle learning/validation data for each candidate
            split (list): if shuffle == `all`||`train`, [learn%, valid%, test%]
            retrain (bool): if True, uses existing project models for
                additional training
            validate (bool): if True, uses a validation set to determine
                learning cutoff
            selection_set (str): best candidates/pool are selected using this
                set; `learn`, `valid`, `train`, `test`, None (all data)
            selection_fn (str): candidates are selected based on this error
                metric; `rmse`, `mean_abs_error`, `med_abs_error`
        '''

        if self._prj_name is None:
            logger.log('info', 'Training single model', call_loc='TRAIN')
            train_model(
                self._sets,
                self._vars,
                selection_set,
                selection_fn,
                retrain,
                validate=validate
            )

        else:
            logger.log('info', 'Training {}x{} models'.format(
                       self._num_pools, self._num_candidates),
                       call_loc='TRAIN')
            logger.log('debug', 'Arguments:\n\t| shuffle:\t\t{}\n\t| split:'
                       '\t\t{}\n\t| retrain:\t\t{}\n\t| selection_set:\t{}\n\t'
                       '| selection_fn:\t\t{}'.format(
                           shuffle, split, retrain, selection_set,
                           selection_fn
                        ), call_loc='TRAIN')

            pool_errors = [[] for _ in range(self._num_pools)]
            if self._num_processes > 1:
                train_pool = Pool(processes=self._num_processes)

            for pool in range(self._num_pools):

                for candidate in range(self._num_candidates):

                    filename = get_candidate_path(
                        self._prj_name,
                        pool,
                        candidate,
                        model=True
                    )
                    save_df(self._df, filename.replace('model.h5', 'data.d'))

                    if self._num_processes > 1:
                        pool_errors[pool].append(train_pool.apply_async(
                            train_model,
                            [self._sets, self._vars, selection_set,
                             selection_fn, retrain, filename, validate]
                        ))

                    else:
                        pool_errors[pool].append(train_model(
                            self._sets,
                            self._vars,
                            selection_set,
                            selection_fn,
                            retrain,
                            filename,
                            validate
                        ))

                    if shuffle is not None:
                        self._df.shuffle(sets=shuffle, split=split)
                        self._sets = self._df.package_sets()

            if self._num_processes > 1:
                train_pool.close()
                train_pool.join()
                for p_id, pool in enumerate(pool_errors):
                    pool_errors[p_id] = [e.get() for e in pool]

            logger.log('debug', 'Pool errors: {}'.format(pool_errors),
                       call_loc='TRAIN')

            for p_id, pool in enumerate(pool_errors):
                candidate_fp = get_candidate_path(
                    self._prj_name, p_id, min(
                        enumerate(pool), key=itemgetter(1)
                    )[0], model=True
                )
                pool_fp = get_candidate_path(self._prj_name, p_id, p_best=True)
                resave_model(candidate_fp, pool_fp)
                resave_df(
                    candidate_fp.replace('model.h5', 'data.d'),
                    pool_fp.replace('model.h5', 'data.d')
                )

    def use(self, dset=None, output_filename=None):
        '''Uses trained neural network(s) to predict for specified set; single
        NN if no project created, best pool candidates if created

        Args:
            dset (str): set to predict for; `learn`, `valid`, `train`, `test`,
                None (all sets)
            output_filename (str): if supplied, saves results to this CSV file

        Returns:
            numpy.array: array of results for specified set
        '''

        if self._prj_name is None:
            results = use_model(self._sets, dset)

        else:
            res = []
            for pool in range(self._num_pools):
                res.append(use_model(
                    self._sets,
                    dset,
                    filename=get_candidate_path(
                        self._prj_name, pool, p_best=True
                    )
                ))
            results = sum(res) / len(res)

        if output_filename is not None:
            save_results(results, dset, self._df, output_filename)
            logger.log('info', 'Results saved to {}'.format(output_filename),
                       call_loc='USE')
        return results

    def errors(self, *args, dset=None):
        '''Obtains various errors for specified set

        Args:
            *args (str): one or more error functions; `rmse`, `mean_abs_error`,
                `med_abs_error`, `r2`
            dset (str): set to obtain errors for; `learn`, `valid`, `train`,
                `test`, None (all sets)

        Returns:
            dict: {'error_fn', value ...} with supplied errors
        '''

        for err in args:
            logger.log('debug', 'Calculating {} for {} set'.format(err, dset),
                       call_loc='ERRORS')
        preds = self.use(dset)
        y_vals = get_y(self._sets, dset)
        errors = {}
        for err in args:
            errors[err] = get_error(preds, y_vals, err)
        logger.log('debug', 'Errors: {}'.format(errors), call_loc='ERRORS')
        return errors

    def save_project(self, filename=None, clean_up=True, del_candidates=False):
        '''Saves current state of project to a .prj file

        Args:
            filename (str): if None, uses name supplied in project creation;
                else, saves the project here
            clean_up (bool): if True, removes project folder structure after
                .prj file created
            del_candidates (bool): if True, deletes all non-chosen candidate
                neural networks
        '''

        if self._prj_name is None:
            raise RuntimeError('A project has not been created')
        save_config(self._vars, self._cf_file)
        save_df(self._df, path.join(self._prj_name, 'data.d'))
        save_path = self._prj_name
        if filename is not None:
            save_path = filename
        if '.prj' not in save_path:
            save_path += '.prj'
        prj_save = ZipFile(save_path, 'w', ZIP_DEFLATED)
        for root, dirs, files in walk(self._prj_name):
            for file in files:
                prj_save.write(path.join(root, file))
            if del_candidates:
                for d in dirs:
                    if 'candidate_' in d:
                        rmtree(path.join(root, d))
        prj_save.close()
        if clean_up:
            rmtree(self._prj_name)
        logger.log('info', 'Project saved to {}'.format(save_path),
                   call_loc='PROJECT')

    def _open_project(self, prj_file):
        '''Private method: if project file specified on Server.__init__, loads
        the project

        Args:
            prj_file (str): path to .prj file
        '''

        self._prj_name = prj_file.replace('.prj', '')
        if '.prj' not in prj_file:
            prj_file += '.prj'
        prj_save = ZipFile(prj_file, 'r')
        prj_save.extractall()
        prj_save.close()
        self._num_pools = len(
            [pool for pool in listdir(self._prj_name)
             if path.isdir(path.join(self._prj_name, pool))]
        )
        for _, _, files in walk(self._prj_name):
            for file in files:
                if '.yml' in files:
                    self._cf_file = file
                    self._vars = {}
                    self._vars.update(open_config(file))
        self._df = open_df(path.join(self._prj_name, 'data.d'))
        self._sets = self._df.package_sets()
        logger.log('info', 'Opened project {}'.format(prj_file),
                   call_loc='PROJECT')
