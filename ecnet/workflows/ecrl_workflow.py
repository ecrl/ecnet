#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/workflows/ecrl_workflow.py
# v.3.3.0
# Developed in 2020 by Travis Kessler <travis.j.kessler@gmail.com>
#
# General workflow used by the UMass Lowell Energy and Combustion Research
# Laboratory
#

# Stdlib imports
from datetime import datetime

# 3rd party imports
from matplotlib import pyplot as plt

# ECNet imports
from ecnet import Server
from ecnet.tasks.tuning import tune_hyperparameters
from ecnet.tools.database import create_db
from ecnet.tools.plotting import ParityPlot
from ecnet.utils.data_utils import DataFrame
from ecnet.utils.logging import logger
from ecnet.utils.server_utils import default_config, save_config
from ecnet.workflows.workflow_utils import find_optimal_num_inputs,\
    prop_range_from_split


def create_model(prop_abvr: str, smiles: list = None, targets: list = None,
                 db_name: str = None, qspr_backend: str = 'padel',
                 create_plots: bool = True, data_split: list = [0.7, 0.2, 0.1],
                 log_level: str = 'info', log_to_file: bool = True,
                 num_processes: int = 1):
    ''' create_model: ECRL's database/model creation workflow for all
    publications

    Args:
        prop_abvr (str): abbreviation for the property name (e.g. CN)
        smiles (list): if supplied with targets, creates a new database
        targets (list): if supplied with smiles, creates a new database
        db_name (str): you may supply an existing ECNet-formatted database
        qspr_backend (str): if creating new database, generation software to
            use (`padel`, `alvadesc`)
        create_plots (bool): if True, creates plots for median absolute error
            vs. number of descriptors as inputs, parity plot for all sets
        data_split (list): [learn %, valid %, test %] for all supplied data
        log_level (str): `debug`, `info`, `warn`, `error`, `crit`
        log_to_file (bool): if True, saves workflow logs to a file in `logs`
            directory
        num_processes (int): number of concurrent processes to use for various
            tasks
    '''

    # Initialize logging
    logger.stream_level = log_level
    if log_to_file:
        logger.file_level = log_level

    # If database not supplied, create database from supplied SMILES, targets
    if db_name is None:
        if smiles is None or targets is None:
            raise ValueError('Must supply SMILES and target values')
        db_name = datetime.now().strftime('{}_model_%Y%m%d.csv'.format(
            prop_abvr
        ))
        logger.log('info', 'Creating database {}...'.format(db_name),
                   'WORKFLOW')
        create_db(smiles, db_name, targets, prop_abvr, backend=qspr_backend)
        logger.log('info', 'Created database {}'.format(db_name), 'WORKFLOW')

    # Create database split, each subset has proportionally equal number of
    #   compounds based on range of experimental/target values
    logger.log('info', 'Creating optimal data split...', 'WORKFLOW')
    prop_range_from_split(db_name, data_split)
    logger.log('info', 'Created optimal data split', 'WORKFLOW')
    df = DataFrame(db_name)
    df.create_sets()
    logger.log('info', '\tLearning set: {}'.format(len(df.learn_set)),
               'WORKFLOW')
    logger.log('info', '\tValidation set: {}'.format(len(df.valid_set)),
               'WORKFLOW')
    logger.log('info', '\tTest set: {}'.format(len(df.test_set)), 'WORKFLOW')

    # Find optimal number of QSPR input variables
    logger.log('info', 'Finding optimal number of inputs...', 'WORKFLOW')
    errors, desc = find_optimal_num_inputs(db_name, 'valid', num_processes)
    df = DataFrame(db_name)
    df.set_inputs(desc)
    df.save(db_name.replace('.csv', '_opt.csv'))
    logger.log('info', 'Found optimal number of inputs', 'WORKFLOW')
    logger.log('info', '\tNumber of inputs: {}'.format(len(df._input_names)),
               'WORKFLOW')

    # Plot the curve of MAE vs. num. desc. added, if desired
    if create_plots:
        logger.log('info', 'Creating plot of MAE vs. descriptors...',
                   'WORKFLOW')
        num_add = [e[0] for e in errors]
        maes = [e[1] for e in errors]
        opt_num = len(desc)
        plt.clf()
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.plot(num_add, maes, c='blue')
        plt.axvline(x=opt_num, c='red', linestyle='--')
        plt.xlabel('Number of Descriptors as ANN Input Variables')
        plt.ylabel('Median Absolute Error of {} Predictions'.format(prop_abvr))
        plt.savefig(db_name.replace('.csv', '_desc_curve.png'))
        logger.log('info', 'Created plot of MAE vs. descriptors', 'WORKFLOW')

    # Tune ANN hyperparameters according to validation set performance
    logger.log('info', 'Tuning ANN hyperparameters...', 'WORKFLOW')
    config = default_config()
    config = tune_hyperparameters(df, config, 25, 10, num_processes,
                                  eval_set='valid', eval_fn='med_abs_error',
                                  epochs=300, validate=False)
    config['epochs'] = default_config()['epochs']
    config_filename = db_name.replace('.csv', '.yml')
    save_config(config, config_filename)
    logger.log('info', 'Tuned ANN hyperparameters', 'WORKFLOW')
    logger.log('info', '\tLearning rate: {}'.format(config['learning_rate']),
               'WORKFLOW')
    logger.log('info', '\tLR decay: {}'.format(config['decay']), 'WORKFLOW')
    logger.log('info', '\tBatch size: {}'.format(config['batch_size']),
               'WORKFLOW')
    logger.log('info', '\tPatience: {}'.format(config['patience']), 'WORKFLOW')
    logger.log('info', '\tHidden layers: {}'.format(config['hidden_layers']),
               'WORKFLOW')

    # Create Model
    logger.log('info', 'Generating ANN...', 'WORKFLOW')
    sv = Server(db_name.replace('.csv', '.yml'), num_processes=num_processes)
    sv.load_data(db_name.replace('.csv', '_opt.csv'))
    sv.create_project(db_name.replace('.csv', ''), 5, 25)
    sv.train(validate=True, selection_set='valid',
             selection_fn='med_abs_error')
    logger.log('info', 'ANN Generated', 'WORKFLOW')
    logger.log('info', 'Measuring ANN performance...', 'WORKFLOW')
    preds_learn = sv.use(dset='learn')
    preds_valid = sv.use(dset='valid')
    preds_test = sv.use(dset='test')
    learn_errors = sv.errors('r2', 'med_abs_error', dset='learn')
    valid_errors = sv.errors('r2', 'med_abs_error', dset='valid')
    test_errors = sv.errors('r2', 'med_abs_error', dset='test')
    logger.log('info', 'Measured ANN performance', 'WORKFLOW')
    logger.log('info', '\tLearning set:\t R2: {}\t MAE: {}'.format(
        learn_errors['r2'], learn_errors['med_abs_error']), 'WORKFLOW')
    logger.log('info', '\tValidation set:\t R2: {}\t MAE: {}'.format(
        valid_errors['r2'], valid_errors['med_abs_error']), 'WORKFLOW')
    logger.log('info', '\tTesting set:\t R2: {}\t MAE: {}'.format(
        test_errors['r2'], test_errors['med_abs_error']), 'WORKFLOW')
    sv.save_project(del_candidates=True)

    if create_plots:
        logger.log('info', 'Creating parity plot...', 'WORKFLOW')
        plt.clf()
        parity_plot = ParityPlot(
            '',
            'Experimental {} Value'.format(prop_abvr),
            'Predicted {} Value'.format(prop_abvr)
        )
        parity_plot.add_series(sv._sets.learn_y, preds_learn, 'Learning Set',
                               'blue')
        parity_plot.add_series(sv._sets.valid_y, preds_valid, 'Validation Set',
                               'green')
        parity_plot.add_series(sv._sets.test_y, preds_test, 'Test Set', 'red')
        parity_plot.add_error_bars(test_errors['med_abs_error'], 'Test MAE')
        parity_plot._add_label('Test $R^2$', test_errors['r2'])
        parity_plot._add_label('Validation MAE', valid_errors['med_abs_error'])
        parity_plot._add_label('Validation $R^2$', valid_errors['r2'])
        parity_plot._add_label('Learning MAE', learn_errors['med_abs_error'])
        parity_plot._add_label('Learning $R^2$', learn_errors['r2'])
        parity_plot.save(db_name.replace('.csv', '_parity.png'))
        logger.log('info', 'Created parity plot', 'WORKFLOW')
