#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/server.py
# v.3.2.3
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains the "Server" class, which handles ECNet project creation, neural
# network model creation, data hand-off to models, prediction error
# calculation, input parameter selection, hyperparameter tuning.
#
# For example scripts, refer to https://ecnet.readthedocs.io/en/latest/
#

# ECNet imports
from ecnet.tasks.limit_inputs import limit_rforest
from ecnet.tasks.training import train_project
from ecnet.tasks.tuning import tune_hyperparameters
from ecnet.utils.data_utils import DataFrame, save_results
from ecnet.utils.logging import logger
from ecnet.utils.server_utils import check_config, create_project,\
    default_config, get_candidate_path, get_error, get_y, open_config,\
    open_df, open_project, resave_df, resave_model, save_config, save_df,\
    save_project, train_model, use_model, use_project


class Server:

    def __init__(self, model_config: str='config.yml', prj_file: str=None,
                 num_processes: int=1):
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
            self._prj_name, self._num_pools, self._num_candidates, self._df,\
                self._cf_file, self._vars = open_project(prj_file)
            check_config(self._vars)
            self._sets = self._df.package_sets()
            logger.log('info', 'Opened project {}'.format(prj_file),
                       call_loc='INIT')
            return

        self._cf_file = model_config
        self._prj_name = None

        self._vars = {}
        try:
            self._vars.update(open_config(self._cf_file))
            check_config(self._vars)
        except FileNotFoundError:
            logger.log('warn', '{} not found, generating default config'
                       .format(model_config), call_loc='INIT')
            self._vars = default_config()
            save_config(self._vars, self._cf_file)

    def load_data(self, filename: str, random: bool=False, split: list=None,
                  normalize: bool=False):
        '''Loads data from an ECNet-formatted CSV database

        Args:
            filename (str): path to CSV database
            random (bool): if True, random set assignments (learn, validate,
                test); if False, uses DB-specified assignmenets
            split (list): if random is True, [learn%, valid%, test%]
            normalize (bool): if true, uses min-max normalization to normalize
                input parameters between 0 and 1
        '''

        logger.log('info', 'Loading data from {}'.format(filename),
                   call_loc='LOAD')
        self._df = DataFrame(filename)
        if normalize:
            self._df.normalize()
        self._df.create_sets(random, split)
        self._sets = self._df.package_sets()

    def create_project(self, project_name: str, num_pools: int=1,
                       num_candidates: int=1):
        '''Creates folder hierarchy for a new project

        Args:
            project_name (str): name of the project, and top-level dir name
            num_pools (int): number of candidate pools for the project
            num_candidates (int): number of candidates per pool
        '''

        self._prj_name = project_name
        self._num_pools = num_pools
        self._num_candidates = num_candidates
        create_project(project_name, num_pools, num_candidates)
        logger.log('info', 'Created project: {}'.format(project_name),
                   call_loc='PROJECT')
        logger.log('debug', 'Number of pools: {}'.format(num_pools),
                   call_loc='PROJECT')
        logger.log('debug', 'Number of candidates/pool: {}'.format(
                   num_candidates), call_loc='PROJECT')

    def limit_inputs(self, limit_num: int, num_estimators: int=None,
                     eval_set: str='learn', output_filename: str=None,
                     **kwargs) -> list:
        '''Selects `limit_num` influential input parameters using random
        forest regression

        Args:
            limit_num (int): desired number of inputs
            num_estimators (int): number of trees in the RFR algorithm;
                defaults to the total number of inputs
            output_filename (str): if not None, new limited database is saved
                here
            eval_set (str): set to perform RFR on (`learn`, `valid`, `train`,
                `test`, None (all)) (default: `learn`)
            **kwargs: any argument accepted by
                sklearn.ensemble.RandomForestRegressor

        Returns:
            list: [(feature, importance), ..., (feature, importance)]
        '''

        result = limit_rforest(
            self._df,
            limit_num,
            num_estimators,
            self._num_processes,
            eval_set,
            **kwargs
        )
        self._df.set_inputs([r[0] for r in result])
        self._sets = self._df.package_sets()
        if output_filename is not None:
            self._df.save(output_filename)
            logger.log('info', 'Resulting database saved to {}'.format(
                       output_filename), call_loc='LIMIT')
        return result

    def tune_hyperparameters(self, num_employers: int, num_iterations: int,
                             shuffle: bool=None, split: list=None,
                             validate: bool=True, eval_set: str=None,
                             eval_fn: str='rmse', epochs: int=300):
        '''Tunes neural network learning hyperparameters using an artificial
        bee colony algorithm; tuned hyperparameters are saved to Server's
        model configuration file

        Args:
            num_employers (int): number of employer bees
            num_iterations (int): number of search iterations for the colony
            shuffle (str): `all` to shuffle all sets for each bee, `train` to
                shuffle learning/validation data for each bee
            split (list): if shuffle is True, [learn%, valid%, test%]
            validate (bool): if True, uses periodic validation; otherwise, no
            eval_set (str): set to evaluate bee fitness; `learn`, `valid`,
                `train`, `test`, None (all sets)
            eval_fn (str): error function used to evaluate bee fitness;
                `rmse`, `mean_abs_error`, `med_abs_error`
            epochs (int): number of training epochs per bee ANN (def: 300)
        '''

        self._vars = tune_hyperparameters(
            self._df,
            self._vars,
            num_employers,
            num_iterations,
            self._num_processes,
            shuffle,
            split,
            validate,
            eval_set,
            eval_fn,
            epochs
        )
        save_config(self._vars, self._cf_file)

    def train(self, shuffle: str=None, split: list=None, retrain: bool=False,
              validate: bool=False, selection_set: str=None,
              selection_fn: str='rmse', model_filename: str='model.ecnet'):
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
            model_filename (str): if project not created, saves `.ecnet` file
                here
        '''

        if self._prj_name is None:
            logger.log('info', 'Training single model', call_loc='TRAIN')
            train_model(
                self._sets,
                self._vars,
                selection_set,
                selection_fn,
                retrain,
                model_filename,
                validate
            )

        else:
            train_project(
                self._prj_name,
                self._num_pools,
                self._num_candidates,
                self._df,
                self._sets,
                self._vars,
                shuffle,
                split,
                retrain,
                validate,
                selection_set,
                selection_fn,
                self._num_processes
            )

    def use(self, dset: str=None, output_filename: str=None,
            model_filename: str='model.ecnet') -> list:
        '''Uses trained neural network(s) to predict for specified set; single
        NN if no project created, best pool candidates if created

        Args:
            dset (str): set to predict for; `learn`, `valid`, `train`, `test`,
                None (all sets)
            output_filename (str): if supplied, saves results to this CSV file
            model_filename (str): if supplied, use specified .ecnet model file

        Returns:
            list: list of results for specified set
        '''

        if self._prj_name is None:
            results = use_model(self._sets, dset, model_filename)

        else:
            results = use_project(
                self._prj_name,
                self._num_pools,
                dset,
                self._sets
            )
        if output_filename is not None:
            save_results(results, dset, self._df, output_filename)
            logger.log('info', 'Results saved to {}'.format(output_filename),
                       call_loc='USE')
        return results

    def errors(self, *args, dset: str=None,
               model_filename: str='model.ecnet') -> dict:
        '''Obtains various errors for specified set

        Args:
            *args (str): one or more error functions; `rmse`, `mean_abs_error`,
                `med_abs_error`, `r2`
            dset (str): set to obtain errors for; `learn`, `valid`, `train`,
                `test`, None (all sets)
            model_filename (str): if specified, uses .ecnet model file for error
                calculations

        Returns:
            dict: {'error_fn', value ...} with supplied errors
        '''

        for err in args:
            logger.log('debug', 'Calculating {} for {} set'.format(err, dset),
                       call_loc='ERRORS')
        preds = self.use(dset, model_filename=model_filename)
        y_vals = get_y(self._sets, dset)
        errors = {}
        for err in args:
            errors[err] = get_error(preds, y_vals, err)
        logger.log('debug', 'Errors: {}'.format(errors), call_loc='ERRORS')
        return errors

    def save_project(self, filename: str=None, clean_up: bool=True,
                     del_candidates: bool=False):
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
        save_path = save_project(
            self._prj_name,
            filename,
            self._cf_file,
            self._df,
            self._vars,
            clean_up,
            del_candidates
        )
        logger.log('info', 'Project saved to {}'.format(save_path),
                   call_loc='PROJECT')
