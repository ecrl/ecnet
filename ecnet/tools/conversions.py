#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tools/conversions.py
# v.3.1.0
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions for converting various chemical file formats
#

# Stdlib imports
from csv import DictReader
from os import devnull
from os.path import abspath, dirname, isfile, join
from re import compile, IGNORECASE
from shutil import which
from subprocess import call

# 3rd party imports
from pubchempy import get_compounds

_PADEL_PATH = join(
    dirname(abspath(__file__)),
    'PaDEL-Descriptor',
    'PaDEL-Descriptor.jar'
)


def get_smiles(name):
    '''Queries PubChemPy for SMILES string for supplied molecule

    Args:
        name (str): name of the molecule (IUPAC, CAS, etc.)

    Returns:
        str or None: if molecule found, returns first idenitifying SMILES,
            else None
    '''

    return [m.isomeric_smiles for m in get_compounds(name, 'name')]


def smiles_to_mdl(smiles_file, mdl_file):
    '''Invoke Open Babel to generate an MDL file containing all supplied
    molecules; requires Open Babel to be installed externally

    Args:
        smiles_file (str): path to .smi file (text file) with SMILES strings
        mdl_file (str): path to resulting MDL file
    '''

    if which('obabel') is None:
        raise ReferenceError('Open Babel installation not found')

    is_mdl = compile(r'.*\.mdl$', IGNORECASE)
    if is_mdl.match(mdl_file) is None:
        raise ValueError('Output file must have an MDL extension: {}'.format(
            mdl_file
        ))

    dn = open(devnull, 'w')
    for attempt in range(3):
        try:
            call([
                'obabel',
                '-i',
                'smi',
                smiles_file,
                '-o',
                'mdl',
                '-O',
                mdl_file,
                '--gen3D'
            ], stdout=dn, stderr=dn, timeout=3600)
            break
        except Exception as e:
            if attempt == 2:
                raise e
            else:
                continue


def mdl_to_descriptors(mdl_file, descriptors_csv, fingerprints=False):
    '''Generates QSPR descriptors from supplied MDL file using
    PaDEL-Descriptor

    Args:
        mdl_file (str): path to source MDL file
        descriptors_csv (str): path to resulting CSV file w/ descriptors
        fingerprints (bool): if True, generates molecular fingerprints instead
            of QSPR descriptors

    Returns:
        list: list of dicts, where each dict is a molecule populated with
            descriptor names (keys) and values
    '''

    if which('java') is None:
        raise ReferenceError(
            'Java JRE 6+ not found (required for PaDEL-Descriptor)'
        )

    dn = open(devnull, 'w')
    for attempt in range(3):
        try:
            if fingerprints:
                call([
                    'java',
                    '-jar',
                    _PADEL_PATH,
                    '-fingerprints',
                    '-retainorder',
                    '-retain3d',
                    '-dir',
                    mdl_file,
                    '-file',
                    descriptors_csv
                ], stdout=dn, stderr=dn, timeout=600)
                break
            else:
                call([
                    'java',
                    '-jar',
                    _PADEL_PATH,
                    '-2d',
                    '-3d',
                    '-retainorder',
                    '-retain3d',
                    '-dir',
                    mdl_file,
                    '-file',
                    descriptors_csv
                ], stdout=dn, stderr=dn, timeout=600)
                break
        except Exception as e:
            if attempt == 2:
                raise e
            else:
                continue

    with open(descriptors_csv, 'r', encoding='utf-8') as desc_file:
        reader = DictReader(desc_file)
        return [row for row in reader]
