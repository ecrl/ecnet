#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/utils/server_utils.py
# v.3.3.0
# Developed in 2020 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions used by ecnet.Server
#

# Stdlib imports
from os import listdir, makedirs, path, walk
from pickle import dump as pdump, load as pload
from shutil import rmtree
from zipfile import ZipFile, ZIP_DEFLATED

# 3rd party imports
from numpy import asarray, array
from yaml import dump, load, FullLoader

# ECNet imports
from ecnet.utils.data_utils import DataFrame, PackagedData
from ecnet.utils.error_utils import calc_rmse, calc_mean_abs_error,\
    calc_med_abs_error, calc_r2
from ecnet.models.mlp import MultilayerPerceptron


def check_config(config: dict) -> dict:
    '''Compares loaded YML configuration to expected variables, sets any
    missing variables to ECNet's default values

    Args:
        config (dict): currently loaded YML configuration

    Returns:
        dict: currently loaded YML configuration, with default values for
            missing variables
    '''

    expected_vars = default_config()
    for key in list(expected_vars.keys()):
        if key not in list(config.keys()):
            config[key] = expected_vars[key]
    return config


def create_project(prj_name: str, num_pools: int, num_candidates: int):
    '''Creates an ECNet project folder structure

    Args:
        prj_name (str): name of the project
        num_pools (int): number of pools in the project
        num_candidates (int): number of candidates in each pool
    '''

    for pool in range(num_pools):
        for candidate in range(num_candidates):
            pc_dir = get_candidate_path(prj_name, pool, candidate)
            if not path.exists(pc_dir):
                makedirs(pc_dir)


def default_config() -> dict:
    '''Returns default NN architecture/learning parameters'''

    return {
        'epochs': 3000,
        'learning_rate': 0.01,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-8,
        'decay': 0.0,
        'hidden_layers': [
            [32, 'relu'],
            [32, 'relu']
        ],
        'output_activation': 'linear',
        'batch_size': 32,
        'patience': 128
    }


def get_candidate_path(prj: str, pool: int, candidate: int = None,
                       model: bool = False, p_best: bool = False) -> str:
    '''Get path to various states of model.h5 files

    Args:
        prj (str): project name
        pool (int): pool number
        candidate (int): candidate number
        model (bool): if True, appends `model.h5` to path
        p_best (bool): if True, removes candidate folder from path

    Returns:
        str: path
    '''

    if model:
        return path.join(
            prj,
            'pool_{}'.format(pool),
            'candidate_{}'.format(candidate),
            'model.h5'
        )
    elif p_best:
        return path.join(
            prj,
            'pool_{}'.format(pool),
            'model.h5'
        )
    else:
        return path.join(
            prj,
            'pool_{}'.format(pool),
            'candidate_{}'.format(candidate)
        )


def get_error(y_hat: array, y: array, fn: str) -> float:
    '''Obtains specified error for supplied results/targets

    Args:
        y_hat (numpy.array): predicted data
        y (numpy.array): target data
        fn (str): error function; `rmse`, `mean_abs_error`, `med_abs_error`,
            `r2`

    Returns:
        float: error
    '''

    if fn == 'rmse':
        return calc_rmse(y_hat, y)
    elif fn == 'r2':
        return calc_r2(y_hat, y)
    elif fn == 'mean_abs_error':
        return calc_mean_abs_error(y_hat, y)
    elif fn == 'med_abs_error':
        return calc_med_abs_error(y_hat, y)
    else:
        raise ValueError('Unknown error function: {}'.format(fn))


def get_x(sets: PackagedData, dset: str) -> array:
    '''Obtains input values for specified data set

    Args:
        sets (ecnet.utils.data_utils.PackagedData): all sets
        dset (str): `learn`, `valid`, `train`, `test`, None (all sets)

    Returns:
        numpy.array: input data for specified set
    '''

    if dset == 'test':
        return sets.test_x
    elif dset == 'valid':
        return sets.valid_x
    elif dset == 'learn':
        return sets.learn_x
    elif dset == 'train':
        x_vals = [list(val) for val in sets.learn_x]
        x_vals.extend([list(val) for val in sets.valid_x])
        return asarray(x_vals)
    elif dset is None:
        x_vals = [list(val) for val in sets.learn_x]
        x_vals.extend([list(val) for val in sets.valid_x])
        x_vals.extend([list(val) for val in sets.test_x])
        return asarray(x_vals)
    else:
        raise ValueError('Unknown dset argument: {}'.format(dset))


def get_y(sets: PackagedData, dset: str) -> array:
    '''Obtains target values for specified data set

    Args:
        sets (ecnet.utils.data_utils.PackagedData): all sets
        dset (str): `learn`, `valid`, `train`, `test`, None (all sets)

    Returns:
        numpy.array: target data for specified set
    '''

    if dset == 'test':
        return sets.test_y
    elif dset == 'valid':
        return sets.valid_y
    elif dset == 'learn':
        return sets.learn_y
    elif dset == 'train':
        y_vals = [val for val in sets.learn_y]
        y_vals.extend([val for val in sets.valid_y])
        return y_vals
    elif dset is None:
        y_vals = [val for val in sets.learn_y]
        y_vals.extend([val for val in sets.valid_y])
        y_vals.extend([val for val in sets.test_y])
        return y_vals
    else:
        raise ValueError('Unknown dset argument: {}'.format(dset))


def open_config(filename: str) -> dict:
    '''Returns contents of YML model configuration file

    Args:
        filename (str): path to YML configuration file

    Returns:
        dict: variable names and values
    '''

    with open(filename, 'r') as cf_file:
        return load(cf_file, FullLoader)


def open_df(filename: str) -> DataFrame:
    '''Opens pickled DataFrame object

    Args:
        filename (str): path to pickled file

    Returns:
        DataFrame: opened DataFrame
    '''

    with open(filename, 'rb') as data_file:
        return pload(data_file)


def open_project(prj_file: str) -> tuple:
    '''Opens an ECNet .prj file, returns project size, last-used DataFrame,
    YML configuration file

    Args:
        prj_file (str): path to .prj file

    Returns:
        tuple: (prj_name, num_pools, num_candidates, DataFrame,
            config_filename, config)
    '''

    prj_name = prj_file.replace('.prj', '')
    if '.prj' not in prj_file:
        prj_file += '.prj'
    prj_save = ZipFile(prj_file, 'r')
    prj_save.extractall()
    prj_save.close()
    num_pools = len(
        [pool for pool in listdir(prj_name)
         if path.isdir(path.join(prj_name, pool))]
    )
    num_candidates = len(
        [cand for cand in listdir(path.join(prj_name, 'pool_0'))
         if path.isdir(path.join(prj_name, 'pool_0', cand))]
    )
    for root, _, files in walk(prj_name):
        for file in files:
            if '.yml' in file:
                vars = open_config(path.join(root, file))
                return (
                    prj_name,
                    len([pool for pool in listdir(prj_name)
                        if path.isdir(path.join(prj_name, pool))]),
                    len([cand for cand in listdir(path.join(
                            prj_name, 'pool_0'
                        )) if path.isdir(path.join(prj_name, 'pool_0', cand))]
                        ),
                    open_df(path.join(prj_name, 'data.d')),
                    file,
                    vars
                )
    raise FileNotFoundError('Unable to locate configuration file in {}'.format(
        prj_file
    ))


def resave_df(old: str, new: str):
    '''Resaves picked DataFrame to new location

    Args:
        old (str): path to existing file
        new (str): path to new file
    '''

    save_df(open_df(old), new)


def resave_model(old: str, new: str):
    '''Resaves .h5 model file

    Args:
        old (str): path to existing file
        new (str): path to new file
    '''

    model = MultilayerPerceptron(filename=old)
    model.load()
    model.save(new)


def save_config(vars: dict, filename: str):
    '''Saves model architecture/learning variables to YML file

    Args:
        filename (str): path to save location
    '''

    with open(filename, 'w') as cf_file:
        dump(vars, cf_file, default_flow_style=False, explicit_start=True)
    cf_file.close()


def save_df(df: DataFrame, filename: str):
    '''Saves DataFrame to pickled file

    Args:
        filename (str): path to save location
    '''

    with open(filename, 'wb') as data_file:
        pdump(df, data_file)
    data_file.close()


def save_project(prj_name: str, filename: str, config_filename: str,
                 df: DataFrame, vars: dict, clean_up: bool,
                 del_candidates: bool) -> str:
    '''Saves an ECNet project folder structure, currently loaded data and
    configuration to a .prj file

    Args:
        prj_name (str): name of the project
        filename (str): (optional) alternate save path for .prj file
        config_filename (str): name of the configuration file
        df (DataFrame): currently loaded data
        vars (dict): current learning/architecture variables
        clean_up (bool): if True, removes project folder structure after .prj
            file created
        del_candidates (bool): if True, deletes all non-chosen candidate
            neural networks

    Returns:
        str: path where project was saved
    '''

    save_config(vars, path.join(prj_name, config_filename))
    save_df(df, path.join(prj_name, 'data.d'))
    save_path = prj_name
    if filename is not None:
        save_path = filename
    if '.prj' not in save_path:
        save_path += '.prj'
    prj_save = ZipFile(save_path, 'w', ZIP_DEFLATED)
    for root, dirs, files in walk(prj_name):
        for file in files:
            prj_save.write(path.join(root, file))
        if del_candidates:
            for d in dirs:
                if 'candidate_' in d:
                    rmtree(path.join(root, d))
    prj_save.close()
    if clean_up:
        rmtree(prj_name)
    return save_path


def train_model(sets: PackagedData, vars: dict, eval_set: str, eval_fn: str,
                retrain: bool = False, filename: str = 'model.h5',
                validate: bool = True, save: bool = True,
                verbose: int = 0) -> tuple:
    '''Trains neural network

    Args:
        sets (ecnet.utils.data_utils.PackagedData): data sets
        vars (dict): Server._vars variables
        eval_set (str): set used to evaluate the model; `learn`, `valid`,
            `train`, `test`, None (all sets)
        eval_fn (str): error function to evaluate the model; `rmse`,
            `mean_abs_error`, `med_abs_error`
        retrain (bool): if True, opens model for additional training
        filename (str): path to .h5 model file, loads from this if retraining
        save (bool): if True, saves model to `filename`
        verbose (int): 1 to display loss at each epoch, 0 otherwise (single
                model only)

    Returns:
        tuple: (error of evaluated set, tuple of learn/valid losses)
    '''

    model = MultilayerPerceptron(filename=filename)
    if retrain:
        model.load()
    else:
        for idx, layer in enumerate(vars['hidden_layers']):
            if idx == 0:
                model.add_layer(layer[0], layer[1], len(sets.learn_x[0]))
            else:
                model.add_layer(layer[0], layer[1])
        model.add_layer(len(sets.learn_y[0]), vars['output_activation'])
    if validate:
        losses = model.fit(
            sets.learn_x,
            sets.learn_y,
            sets.valid_x,
            sets.valid_y,
            epochs=vars['epochs'],
            lr=vars['learning_rate'],
            beta_1=vars['beta_1'],
            beta_2=vars['beta_2'],
            epsilon=vars['epsilon'],
            decay=vars['decay'],
            batch_size=vars['batch_size'],
            v=verbose,
            patience=vars['patience']
        )
    else:
        losses = model.fit(
            sets.learn_x,
            sets.learn_y,
            epochs=vars['epochs'],
            lr=vars['learning_rate'],
            beta_1=vars['beta_1'],
            beta_2=vars['beta_2'],
            epsilon=vars['epsilon'],
            decay=vars['decay'],
            batch_size=vars['batch_size'],
            v=verbose,
            patience=vars['patience']
        )
    if save:
        model.save()
    return (get_error(
        model.use(get_x(sets, eval_set)),
        get_y(sets, eval_set),
        eval_fn
    ), losses)


def use_model(sets: PackagedData, dset: str,
              filename: str = 'model.h5') -> array:
    '''Uses existing model to predict data

    Args:
        sets (ecnet.utils.data_utils.PackagedData): data sets
        dset (str): set to predict for; `learn`, `valid`, `train`, `test`,
            None (all sets)
        filename (str): path to existing .h5 model file

    Returns:
        numpy.array: predicted data
    '''

    model = MultilayerPerceptron(filename=filename)
    model.load()
    return model.use(get_x(sets, dset))


def use_project(prj_name: str, num_pools: int, dset: str,
                sets: PackagedData) -> list:
    '''Uses existing ECNet project to predict values for specified set

    Args:
        prj_name (str): name of the project
        num_pools (int): number of pools in the project
        dset (str): dataset to predict for
        sets (PackagedData): currently loaded data sets

    Returns:
        list: list of results for specified set
    '''

    res = []
    for pool in range(num_pools):
        res.append(use_model(
            sets,
            dset,
            get_candidate_path(prj_name, pool, p_best=True)
        ))
    return sum(res) / len(res)
