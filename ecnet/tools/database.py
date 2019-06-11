#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ecnet/tools/database.py
# v.3.2.0
# Developed in 2019 by Travis Kessler <travis.j.kessler@gmail.com>
#
# Contains functions for creating ECNet-formatted databases
#

# Stdlib imports
from csv import writer
from datetime import datetime
from os import remove
from warnings import warn

# 3rd party imports
from alvadescpy import smiles_to_descriptors
from padelpy import from_smiles


class _Molecule:

    def __init__(self, id):

        self.id = id
        self.assignment = 'L'
        self.strings = {'Compound Name': ''}
        self.target = 0
        self.inputs = None


def create_db(smiles: list, db_name: str, targets: list=None,
              id_prefix: str='', extra_strings: dict={}, backend: str='padel'):
    ''' create_db: creates an ECNet-formatted database from SMILES strings
    using either PaDEL-Descriptor or alvaDesc software; using alvaDesc
    requires a valid installation/license of alvaDesc

    Args:
        smiles (list): list of SMILES strings
        db_name (str): name/path of database being created
        targets (list): target (experimental) values, align with SMILES
            strings; if None, all TARGETs set to 0
        id_prefix (str): prefix of molecule DATAID, if desired
        extra_strings (dict): extra STRING columns, label = name, value = list
            with length equal to number of SMILES strings
        backend (str): software used to calculate QSPR descriptors, 'padel' or
            'alvadesc'
    '''

    if targets is not None:
        if len(targets) != len(smiles):
            raise ValueError('Must supply same number of targets as SMILES '
                             'strings: {}, {}'.format(
                                 len(targets), len(smiles)
                             ))

    for string in list(extra_strings.keys()):
        if len(extra_strings[string]) != len(smiles):
            raise ValueError('Extra string values for {} not equal in length '
                             'to supplied SMILES: {}, {}'.format(
                                 len(extra_strings[string]), len(smiles)
                             ))

    mols = []
    if backend == 'alvadesc':
        for mol in smiles:
            mols.append(smiles_to_descriptors(mol))
    elif backend == 'padel':
        for idx, mol in enumerate(smiles):
            try:
                mols.append(from_smiles(mol))
            except RuntimeError:
                warn('Could not calculate descriptors for {}, omitting'.format(
                     mol), RuntimeWarning)
                del smiles[idx]
                if targets is not None:
                    del targets[idx]
                for string in list(extra_strings.keys()):
                    del extra_strings[string][idx]
    else:
        raise ValueError('Unknown backend software: {}'.format(backend))

    rows = []
    type_row = ['DATAID', 'ASSIGNMENT', 'STRING', 'STRING']
    title_row = ['DATAID', 'ASSIGNMENT', 'Compound Name', 'SMILES']
    strings = list(extra_strings.keys())
    for string in strings:
        if string != 'Compound Name':
            type_row.append('STRING')
            title_row.append(string)
    type_row.append('TARGET')
    title_row.append('TARGET')
    descriptor_keys = list(mols[0].keys())
    for key in descriptor_keys:
        type_row.append('INPUT')
        title_row.append(key)

    mol_rows = []
    for idx, desc in enumerate(mols):
        for key in descriptor_keys:
            if desc[key] == 'na' or desc[key] == '':
                desc[key] = 0
        mol = _Molecule('{}'.format(id_prefix) + '%04d' % (idx + 1))
        for string in strings:
            mol.strings[string] = extra_strings[string][idx]
        if targets is not None:
            mol.target = targets[idx]
        mol.inputs = desc
        mol_rows.append(mol)

    with open(db_name, 'w', encoding='utf-8') as db_file:
        wr = writer(db_file, delimiter=',', lineterminator='\n')
        wr.writerow(type_row)
        wr.writerow(title_row)
        for idx, mol in enumerate(mol_rows):
            row = [mol.id, mol.assignment, mol.strings['Compound Name'],
                   smiles[idx]]
            for string in strings:
                if string != 'Compound Name':
                    row.append(mol.strings[string])
            row.append(mol.target)
            for key in descriptor_keys:
                row.append(mol.inputs[key])
            wr.writerow(row)
    db_file.close()
