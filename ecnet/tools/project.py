#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tools/project.py
# v.3.1.2
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions for predicting data using pre-existing .prj files
#

# Stdlib imports
from datetime import datetime
from os import remove
from shutil import rmtree

# ECNet imports
from ecnet import Server
from ecnet.utils.data_utils import DataFrame
from ecnet.utils.logging import logger
from ecnet.tools.database import create_db


def predict(smiles: list, prj_file: str, results_file: str=None,
            backend: str='padel') -> list:
    ''' predict: predicts values for supplied molecules (SMILES strings) using
    pre-existing ECNet project (.prj) file

    Args:
        smiles (str): SMILES strings for molecules
        prj_file (str): path to ECNet .prj file
        results_file (str): if not none, saves results to this CSV file
        backend (str): `padel` (default) or `alvadesc`, depending on the data
            your project was trained with

    Returns:
        list: predicted values
    '''

    sv = Server(prj_file=prj_file)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    create_db(smiles, '{}.csv'.format(timestamp), backend=backend)
    new_data = DataFrame('{}.csv'.format(timestamp))
    new_data.set_inputs(sv._df._input_names)
    new_data.create_sets()
    sv._df = new_data
    sv._sets = sv._df.package_sets()
    results = sv.use(output_filename=results_file)
    remove('{}.csv'.format(timestamp))
    rmtree(prj_file.replace('.prj', ''))
    return results
