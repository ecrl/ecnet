#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tools/project.py
# v.3.3.0
# Developed in 2020 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions for predicting data using pre-existing .prj files
#

# Stdlib imports
from os import walk
from os.path import join
from re import compile, IGNORECASE
from tempfile import TemporaryDirectory
from warnings import warn
from zipfile import ZipFile

# 3rd party imports
from alvadescpy import smiles_to_descriptors
from numpy import asarray, mean
from padelpy import from_smiles

# ECNet imports
from ecnet.models.mlp import MultilayerPerceptron
from ecnet.utils.server_utils import open_config, open_df

CONFIG_RE = compile(r'^.*\.yml$', IGNORECASE)
MODEL_RE = compile(r'^.*\.h5$', IGNORECASE)


class TrainedProject:

    def __init__(self, filename: str):
        ''' TrainedProject: loads a trained ECNet project, including last-used
        DataFrame, configuration .yml file, and all trained models

        Args:
            filename (str): name/path of the trained .prj file
        '''

        self._df = None
        self._config = None
        self._models = []

        with ZipFile(filename, 'r') as zf:
            prj_zip = zf.namelist()
            if '{}/data.d'.format(filename.replace('.prj', '')) not in prj_zip:
                raise Exception('`data.d` not found in .prj file')
            with TemporaryDirectory() as tmpdirname:
                zf.extractall(tmpdirname)
                prj_dirname = join(tmpdirname, filename.replace('.prj', ''))
                self._df = open_df(join(prj_dirname, 'data.d'))
                for root, _, files in walk(prj_dirname):
                    for f in files:
                        if MODEL_RE.match(f) is not None:
                            _model = MultilayerPerceptron(join(root, f))
                            _model.load()
                            self._models.append(_model)
                        elif CONFIG_RE.match(f) is not None:
                            self._config = open_config(join(root, f))

    def use(self, smiles: list, backend: str = 'padel'):
        ''' use: uses the trained project to predict values for supplied
        molecules

        Args:
            smiles (list): list of SMILES strings to predict for
            backend (str): backend software to use for QSPR generation; `padel`
                or `alvadesc`; default = `padel`; alvadesc requries valid
                license

        Returns:
            numpy.array: predicted values
        '''

        if backend == 'alvadesc':
            mols = [smiles_to_descriptors(s) for s in smiles]
        elif backend == 'padel':
            mols = [from_smiles(s) for s in smiles]
        else:
            raise ValueError('Unknown backend software: {}'.format(backend))
        return mean([model.use(asarray(
            [[float(mol[name]) for name in self._df._input_names]
             for mol in mols]
        )) for model in self._models], axis=0)


def predict(smiles: list, prj_file: str, results_file: str = None,
            backend: str = 'padel') -> list:
    ''' predict: predicts values for supplied molecules (SMILES strings) using
    pre-existing ECNet project (.prj) file

    Args:
        smiles (str): SMILES strings for molecules
        prj_file (str): path to ECNet .prj file
        results_file (str): if not none, saves results to this CSV file
            (WARNING: depricated, no longer saves to file)
        backend (str): `padel` (default) or `alvadesc`, depending on the data
            your project was trained with

    Returns:
        list: predicted values
    '''

    if results_file is not None:
        class NotImplementedWarning(UserWarning):
            pass
        warn('`predict` no longer saves directly to a file, results are only'
             ' returned to the user', NotImplementedWarning)

    project = TrainedProject(prj_file)
    return project.use(smiles, backend)
