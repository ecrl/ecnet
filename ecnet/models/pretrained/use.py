#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/models/pretrained/use.py
# v.3.3.0
# Developed in 2020 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains function for predicting properties using pre-trained .prj files
# bundled with ECNet
#

# Stdlib imports
from os.path import abspath, dirname, join

# ECNet imports
from ecnet.tools.project import TrainedProject

_BASEDIR = join(dirname(abspath(__file__)), 'prj')

_TRAINED_MODELS = {
    'CN_alvadesc': join(_BASEDIR, 'cn_model_v2.0.prj'),
    'CN_padel': join(_BASEDIR, 'cn_model_v2.1.prj')
}

_TEST_MED_ABS_ERRORS = {
    'CN_alvadesc': 5.0,
    'CN_padel': 6.0
}


def predict(smiles: list, prop: str, backend: str = 'padel') -> tuple:
    ''' predict: uses a pretrained model, packaged with ECNet, to predict values
    for supplied molecules; returns predicted values, approximate error based
    on model test set median absolute error

    Args:
        smiles (list): SMILES strings of supplied molecules
        prop (str): abbreviation of property to predict:
            CN - cetane number
            CP - cloud point
            KV - kinematic viscosity
            MON - motor octane number
            PP - pour point
            RON - research octane number
            OS - octane sensitivity
            YSI - yield sooting index
        backend (str): the QSPR generation software used to generate the model;
            `padel` or `alvadesc`; default = `padel`; alvadesc requires valid
            license

    Returns:
        tuple: (predicted values, median absolute error of model test set)
    '''

    to_use = '{}_{}'.format(prop, backend)
    available_models = list(_TRAINED_MODELS.keys())
    if to_use not in available_models:
        raise IndexError('Model for `{}` trained with `{}` descriptors not '
                         'found in available models: {}'.format(
                             prop, backend, available_models
                        ))

    prj = TrainedProject(_TRAINED_MODELS[to_use])
    return (prj.use(smiles, backend), _TEST_MED_ABS_ERRORS[to_use])
