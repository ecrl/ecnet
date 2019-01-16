#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# tools/names_to_smiles.py
#
# Houses functions to obtain SMILES strings from molecule names
#

from argparse import ArgumentParser
from csv import DictReader, DictWriter
from sys import exit, argv

try:
    from pubchempy import get_compounds
except ImportError:
    print('Error: PubChemPy is not installed!')
    print('Use `pip install pubchempy`')
    exit([1])


def get_names(csv_file, column_name):
    '''Reads molecule names from supplied CSV file

    Args:
        csv_file (str): path to CSV file
        column_name (str): column header for names

    Returns:
        list: list of names, strings
    '''

    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = DictReader(file)
        return [row[column_name] for row in reader]


def get_smiles(names):
    '''Queries PubChem for SMILES strings for supplied names

    Args:
        names (list): list of molecule names

    Returns:
        list: list of corresponding SMILES strings
    '''

    smiles = []
    for name in names:
        smi = [m.isomeric_smiles for m in get_compounds(name, 'name')]
        if len(smi) == 0:
            print('Warning: SMILES for {} not found'.format(name))
            smi = ''
        else:
            smi = smi[0]
        smiles.append(smi)
    return smiles


def save_file(names, smiles, output_filename, names_header='Compound Name',
              smiles_header='SMILES'):
    '''Saves names and SMILES to CSV file

    Args:
        names (list): list of molecule names
        smiles (list): list of SMILES strings (equal length to names)
        output_filename (str): path to resulting file
        names_header (str): names column header (defaults to 'Compound Name')
        smiles_header (str): SMILES column header (defaults to 'SMILES')
    '''

    with open(output_filename, 'w', encoding='utf-8') as file:
        writer = DictWriter(
            file,
            delimiter=',',
            lineterminator='\n',
            fieldnames=[names_header, smiles_header]
        )
        writer.writeheader()
        for idx, name in enumerate(names):
            writer.writerow({
                names_header: name,
                smiles_header: smiles[idx]
            })


def parse_args():
    '''Run from command line

    Returns:
        dict: {
            'input_file',
            'output_file',
            'names_header',
            'smiles_header'
        }
    '''

    ap = ArgumentParser()
    ap.add_argument(
        'input_file',
        type=str,
        help='Path to CSV file with molecule names'
    )
    ap.add_argument(
        '--output_file',
        type=str,
        help='Name of the resulting file containing names, SMILES',
        default=None
    )
    ap.add_argument(
        '--names_header',
        type=str,
        help='Header for molecule names column',
        default='Compound Name'
    )
    ap.add_argument(
        '--smiles_header',
        type=str,
        help='Header for SMILES column of output file',
        default='SMILES'
    )

    return vars(ap.parse_args(argv[1:]))


def main(args):
    '''Run from command line'''

    names = get_names(args['input_file'], args['names_header'])
    smiles = get_smiles(names)
    if args['output_file'] is None:
        args['output_file'] = args['input_file'].replace(
            '.csv',
            '_smiles.csv'
        )
    save_file(
        names,
        smiles,
        args['output_file'],
        names_header=args['names_header'],
        smiles_header=args['smiles_header']
    )


if __name__ == '__main__':
    '''Run from command line'''

    main(parse_args())
